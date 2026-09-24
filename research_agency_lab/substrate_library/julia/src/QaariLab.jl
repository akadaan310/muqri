"""
    QaariLab

Julia research core for qaari-eval: reference-reciter calibration and model discovery.

* `calibrate`   — robust (median/MAD) reference bands per rule key, anchored on Al-Hussary and
                  widened to his ijaazah peers' consensus; leave-one-peer-out validation;
                  category-weight optimisation (Optim.jl) separating peers from imams.
* `Discovery`   — tempo dynamics ODE (OrdinaryDiffEq.jl, fitted with Optim + ForwardDiff) and
                  sparse regression (STLSQ, SINDy-style) for the duration law of a harakah count.

Input rows are the JSONL written by `benchmarks/run_benchmark.py` (raw, uncalibrated verdicts).
"""
module QaariLab

using JSON3
using Statistics
using StatsBase: quantile, mad
using LinearAlgebra
using Optim

export load_rows, calibrate, write_json

const MADK = 1.4826  # MAD → σ for a normal distribution

# ------------------------------------------------------------------ data ------------------------
struct Obs
    reciter::String
    mode::String
    surah::Int
    ayah::Int
    key::String
    rule_type::String
    status::String
    score::Float64
    metrics::Dict{String,Float64}
end

"""Load benchmark rows (one JSON object per line) into flat per-diagnostic observations."""
function load_rows(paths::Vector{String}; modes = ("studio",))
    obs = Obs[]
    rows = Dict{String,Any}[]
    for p in paths, line in eachline(p)
        isempty(strip(line)) && continue
        r = JSON3.read(line)
        String(r.mode) in modes || continue
        push!(rows, Dict("reciter" => String(r.reciter), "mode" => String(r.mode), "surah" => r.surah,
                         "ayah" => r.ayah, "haraka_ms" => get(r, :haraka_ms, NaN),
                         "duration_s" => get(r, :duration_s, NaN)))
        for d in r.diagnostics
            m = Dict{String,Float64}()
            if haskey(d, :metrics)
                for (k, v) in pairs(d.metrics)
                    v isa Number && (m[String(k)] = Float64(v))
                end
            end
            key = haskey(d, :key) ? String(d.key) : String(d.rule_type)
            sc = haskey(d, :score) && d.score !== nothing ? Float64(d.score) : NaN
            push!(obs, Obs(String(r.reciter), String(r.mode), r.surah, r.ayah, key, String(d.rule_type),
                           String(d.status), sc, m))
        end
    end
    return obs, rows
end

# ------------------------------------------------------------ robust stats ----------------------
struct Robust
    n::Int
    median::Float64
    sigma::Float64   # 1.4826 · MAD
    p5::Float64
    p25::Float64
    p75::Float64
    p95::Float64
end

function robust(v::AbstractVector{<:Real})
    x = filter(isfinite, Float64.(v))
    isempty(x) && return Robust(0, NaN, NaN, NaN, NaN, NaN, NaN)
    q = quantile(x, [0.05, 0.25, 0.75, 0.95])
    Robust(length(x), median(x), MADK * mad(x; normalize = false), q...)
end

"Robust coefficient of variation: how noisy a ruler is relative to what it measures."
rcv(r::Robust) = r.sigma / max(abs(r.median), 1e-9)

function scale_floor(metric::String, floors)
    for (k, v) in pairs(floors)
        k = String(k)
        (metric == k || endswith(metric, k)) && return Float64(v)
    end
    return 0.0
end

# --------------------------------------------------------------- bands ---------------------------
struct Band
    metric::String
    side::String
    lo::Float64
    hi::Float64
    scale::Float64
end

function zscore(b::Band, x::Float64)
    d = x < b.lo ? (b.side == "upper" ? 0.0 : b.lo - x) :
        x > b.hi ? (b.side == "lower" ? 0.0 : x - b.hi) : 0.0
    return d / b.scale
end

"Identical to app.calibration.verdict (Python): |z|≤2 PASS, ≤3 WARNING, else FAIL."
function verdict(z::Float64)
    z <= 2.0 && return ("PASS", 1.0)
    z <= 3.0 && return ("WARNING", 1.0 - 0.6 * (z - 2.0))
    return ("FAIL", max(0.0, 0.4 * (1 - (z - 3.0) / 3.0)))
end

reliable(o::Obs, conf_min, collapsed_ms, duration_rule) =
    get(o.metrics, "align_conf", 1.0) >= conf_min &&
    get(o.metrics, "align_min_ms", Inf) >= collapsed_ms &&
    (!duration_rule || get(o.metrics, "core_ms", 1.0) > 0.0)

values_of(obs, reciters, key, metric) =
    [o.metrics[metric] for o in obs if o.reciter in reciters && o.key == key && haskey(o.metrics, metric)]

"""
    build_bands(obs, anchor, peers, spec; conf_min)

For every rule key: pick the ruler (lowest anchor rCV among `alts`), centre the band on the hull of
the anchor median and the peer consensus (median of per-peer medians), and scale it by the pooled
within-reciter σ (median over anchor and peers), floored per unit.
"""
function build_bands(obs::Vector{Obs}, anchor::String, peers::Vector{String}, spec; conf_min::Float64)
    rules = spec.rules
    floors = spec.scale_floor
    dur = Set(String.(spec.duration_rules))
    collapsed = Float64(spec.collapsed_unit_ms)
    keys_by_type = Dict{String,Set{String}}()
    for o in obs
        push!(get!(keys_by_type, o.rule_type, Set{String}()), o.key)
    end
    usable = [o for o in obs if reliable(o, conf_min, collapsed, o.rule_type in dur)]
    bands = Dict{String,Vector{Band}}()
    report = Dict{String,Any}()
    for (rt, groups) in pairs(rules)
        rt = String(rt)
        for key in get(keys_by_type, rt, Set{String}())
            kb = Band[]
            krep = Any[]
            for g in groups
                best = nothing
                cands = Dict{String,Any}()
                for m in String.(g.alts)
                    ra = robust(values_of(usable, (anchor,), key, m))
                    ra.n >= spec.min_anchor_n || continue
                    cands[m] = Dict("n" => ra.n, "median" => ra.median, "rcv" => rcv(ra))
                    if best === nothing || rcv(ra) < rcv(best[2])
                        best = (m, ra)
                    end
                end
                best === nothing && continue
                m, ra = best
                peer_meds = Float64[]
                sigmas = [ra.sigma]
                pstats = Dict{String,Any}()
                for p in peers
                    rp = robust(values_of(usable, (p,), key, m))
                    rp.n >= spec.min_peer_n || continue
                    push!(peer_meds, rp.median)
                    push!(sigmas, rp.sigma)
                    pstats[p] = Dict("n" => rp.n, "median" => rp.median, "sigma" => rp.sigma)
                end
                consensus = isempty(peer_meds) ? ra.median : median(peer_meds)
                lo, hi = minmax(ra.median, consensus)
                s = max(median(sigmas), scale_floor(m, floors), 1e-6)
                push!(kb, Band(m, String(g.side), lo, hi, s))
                push!(krep, Dict("metric" => m, "side" => g.side, "candidates" => cands,
                                 "anchor" => Dict("n" => ra.n, "median" => ra.median, "sigma" => ra.sigma,
                                                  "p5" => ra.p5, "p25" => ra.p25, "p75" => ra.p75, "p95" => ra.p95),
                                 "peers" => pstats, "peer_consensus" => consensus))
            end
            isempty(kb) || (bands[key] = kb; report[key] = krep)
        end
    end
    return bands, report
end

# ------------------------------------------------------------- scoring -------------------------
"""Re-score observations with bands (unbanded rules keep their raw verdict); per-reciter index."""
function rescore(obs, bands, taxonomy; weights = nothing, conf_min = 0.0, collapsed = 40.0, dur = Set{String}())
    catw = weights === nothing ? Dict(String(k) => Float64(v) for (k, v) in pairs(taxonomy.category_weights)) : weights
    rc = Dict(String(k) => String(v) for (k, v) in pairs(taxonomy.rule_category))
    per = Dict{String,Dict{String,Vector{Float64}}}()
    for o in obs
        o.status in ("SKIPPED", "VALID_NECESSARY_PAUSE") && continue
        sc = o.score
        if haskey(bands, o.key)
            reliable(o, conf_min, collapsed, o.rule_type in dur) || continue
            zs = [zscore(b, o.metrics[b.metric]) for b in bands[o.key] if haskey(o.metrics, b.metric)]
            isempty(zs) || (sc = verdict(maximum(zs))[2])
        end
        isfinite(sc) || continue
        cat = get(rc, o.rule_type, "other")
        push!(get!(get!(per, o.reciter, Dict{String,Vector{Float64}}()), cat, Float64[]), sc)
    end
    out = Dict{String,Any}()
    for (r, cats) in per
        ahk = [c for c in keys(cats) if c != "sifaat" && haskey(catw, c)]
        w = sum(catw[c] * length(cats[c]) for c in ahk; init = 0.0)
        perf = w > 0 ? 100 * sum(catw[c] * sum(cats[c]) for c in ahk) / w : NaN
        sif = haskey(cats, "sifaat") ? 100 * mean(cats["sifaat"]) : NaN
        out[r] = Dict("perfection" => perf, "sifaat" => sif,
                      "categories" => Dict(c => 100 * mean(v) for (c, v) in cats))
    end
    return out
end

# ----------------------------------------------------------- calibrate --------------------------
"""
    calibrate(paths; anchor, peers, imams, spec, taxonomy) -> Dict

1. count scale: 2.0 / anchor median of madd_tabii counts (the chosen ruler), so Husary's natural madd reads 2.
2. alignment reliability: posterior floor = max(0.2, anchor 2nd percentile); collapsed units < 40 ms.
3. bands from the anchor + all peers; leave-one-peer-out bands to score each held-out peer.
4. category weights: maximise mean(peers) − mean(imams) with each weight bounded to ×/÷2 of its
   default and a ridge pull (Optim.jl, Nelder–Mead); kept only if separation gains ≥ 1 point.
   The ≥ 95 peer target is reported, never optimised for.
"""
function calibrate(paths::Vector{String}; anchor::String, peers::Vector{String}, imams::Vector{String},
                   spec, taxonomy, target_peer::Float64 = 95.0)
    obs, rows = load_rows(paths; modes = ("studio",))
    dur = Set(String.(spec.duration_rules))
    confs = [o.metrics["align_conf"] for o in obs if o.reciter == anchor && haskey(o.metrics, "align_conf")]
    conf_min = isempty(confs) ? 0.2 : max(0.2, quantile(confs, 0.02))
    collapsed = Float64(spec.collapsed_unit_ms)

    bands, report = build_bands(obs, anchor, peers, spec; conf_min = conf_min)

    # count scale from the natural madd, on whichever ruler won
    madd_metric = haskey(bands, "madd_tabii") ? bands["madd_tabii"][1].metric : "counts"
    mt = robust(values_of(obs, (anchor,), "madd_tabii", madd_metric))
    count_scale = mt.n > 0 ? 2.0 / mt.median : 1.0

    # leave-one-peer-out
    loo = Dict{String,Any}()
    for p in peers
        others = [q for q in peers if q != p]
        b, _ = build_bands(obs, anchor, others, spec; conf_min = conf_min)
        s = rescore([o for o in obs if o.reciter == p], b, taxonomy; conf_min = conf_min, collapsed = collapsed, dur = dur)
        loo[p] = get(s, p, Dict("perfection" => NaN))
    end
    full = rescore(obs, bands, taxonomy; conf_min = conf_min, collapsed = collapsed, dur = dur)
    raw = rescore(obs, Dict{String,Vector{Band}}(), taxonomy; conf_min = 0.0, collapsed = 0.0)

    weights = optimise_weights(obs, bands, taxonomy, peers, imams; conf_min = conf_min, collapsed = collapsed,
                               dur = dur, target = target_peer)

    return Dict(
        "version" => 1, "anchor" => anchor, "peers" => peers, "imams" => imams,
        "count_scale" => count_scale, "count_scale_ruler" => madd_metric,
        "reliability" => Dict("align_conf_min" => conf_min, "align_min_ms" => collapsed),
        "rules" => Dict(k => [Dict("metric" => b.metric, "side" => b.side, "lo" => b.lo, "hi" => b.hi,
                                   "scale" => b.scale) for b in v] for (k, v) in bands),
        "category_weights" => weights["weights"],
        "validation" => Dict("raw_textbook" => raw, "calibrated_in_sample" => full, "leave_one_peer_out" => loo,
                             "weight_search" => weights),
        "evidence" => report,
        "n_observations" => length(obs), "n_rows" => length(rows),
    )
end

function optimise_weights(obs, bands, taxonomy, peers, imams; conf_min, collapsed, dur, target)
    cats = [String(k) for k in keys(taxonomy.category_weights) if String(k) != "sifaat"]
    w0 = Dict(String(k) => Float64(v) for (k, v) in pairs(taxonomy.category_weights))
    present = Set(vcat(peers, imams))
    sub = [o for o in obs if o.reciter in present]
    has_imams = any(o.reciter in imams for o in sub)
    base = log.([w0[c] for c in cats])
    # Weights may move at most ×/÷ 2 from the defaults (bounded by tanh): a category that everyone
    # passes (e.g. wasl) must not be inflated to lift scores — the first, unbounded search did exactly
    # that (wasl ×30), raising the imams as much as the peers.
    toweights(x) = base .+ log(2.0) .* tanh.(x)
    function evaluate(x)
        w = copy(w0)
        for (i, c) in enumerate(cats)
            w[c] = exp(toweights(x)[i])
        end
        s = rescore(sub, bands, taxonomy; weights = w, conf_min = conf_min, collapsed = collapsed, dur = dur)
        pp = [s[p]["perfection"] for p in peers if haskey(s, p)]
        ii = [s[i]["perfection"] for i in imams if haskey(s, i)]
        return pp, ii, w
    end
    separation(pp, ii) = isempty(pp) || isempty(ii) ? 0.0 : mean(pp) - mean(ii)
    # Objective: separation of the reference reciters from the imams only (the ≥ target score is a
    # reported check, never optimised for), with a ridge toward the defaults.
    function objective(x)
        pp, ii, _ = evaluate(x)
        return -separation(pp, ii) + 0.5 * sum(abs2, x)
    end
    x0 = zeros(length(cats))
    xbest = x0
    if has_imams
        try
            res = Optim.optimize(objective, x0, Optim.NelderMead(), Optim.Options(iterations = 300))
            xbest = Optim.minimizer(res)
        catch err
            @warn "Weight optimisation skipped" err
        end
    end
    pp0, ii0, _ = evaluate(x0)
    pp, ii, w = evaluate(xbest)
    gain = separation(pp, ii) - separation(pp0, ii0)
    accepted = has_imams && gain >= 1.0 && minimum(pp) >= minimum(pp0) - 0.5
    return Dict("weights" => accepted ? w : w0, "tuned_weights" => w, "default_weights" => w0,
                "accepted" => accepted,
                "rule" => "accept only if peer–imam separation gains ≥ 1 point and no peer drops > 0.5",
                "separation_default" => separation(pp0, ii0), "separation_tuned" => separation(pp, ii),
                "peers_mean_default" => isempty(pp0) ? NaN : mean(pp0),
                "imams_mean_default" => isempty(ii0) ? NaN : mean(ii0),
                "peers_mean_tuned" => isempty(pp) ? NaN : mean(pp),
                "imams_mean_tuned" => isempty(ii) ? NaN : mean(ii),
                "peers_min_default" => isempty(pp0) ? NaN : minimum(pp0),
                "peers_min_tuned" => isempty(pp) ? NaN : minimum(pp),
                "peers_reaching_target" => count(>=(target), pp0),
                "had_imams" => has_imams)
end

function write_json(path::String, obj)
    open(path, "w") do io
        JSON3.pretty(io, JSON3.write(sanitize(obj)))
    end
end

sanitize(x::AbstractFloat) = isfinite(x) ? x : nothing
sanitize(x::AbstractDict) = Dict(String(k) => sanitize(v) for (k, v) in x)
sanitize(x::AbstractVector) = [sanitize(v) for v in x]
sanitize(x::Tuple) = [sanitize(v) for v in x]
sanitize(x) = x

include("Discovery.jl")

end # module
