# Frontier calibration layer (included by QaariLab.jl) — frontier_math_roadmap.md items A1–A4, B1, C2.
#
# Rule-agnostic: every rule family (madd, ghunnah, noon/meem, qalqalah, idgham, tafkheem/tarqeeq,
# sifaat, sakt, hamzat al-wasl) is treated the same way — a reciter's instances of one rule key form
# a cloud of multi-metric feature vectors, and the question "does this reciter realise the rule the
# way the ijazah reference does?" becomes a comparison of distributions.
#
#  1. Features per key: the spec's judged metrics first, then every other numeric metric the anchor
#     measures on ≥ 90 % of its reliable instances, kept only if it adds rank (no exact linear
#     redundancy such as f2_minus_f1_hz = f2_hz − f1_hz), at most `dmax`.
#  2. Robust whitening by the pooled *within-reciter* scatter (OGK, Maronna & Zamar 2002 — the
#     multivariate 1.4826·MAD), so all units (ms, counts, Hz, dB) become commensurate.
#  3. Each reciter ≈ N(mᵢ, Cᵢ) (OGK + shrinkage to the pooled scatter = I in whitened space).
#  4. Reference = anchor-weighted Bures–Wasserstein barycenter (fixed point of Álvarez-Esteban et al.
#     2016, which converges for any start).
#  5. Distance = location ‖m − m̄‖² (wrong length / formant / energy) + β · AIRM²(C, C̄) (unit-invariant
#     shape: inconsistent or over-dispersed realisation — tasawi at the ijazah level); the Bures shape
#     term is reported too.
#  6. Thresholds are conformal: leave-one-peer-out distances are the null distribution, PASS/WARN
#     cuts are their ⌈(P+1)(1−α)⌉-th order statistics (Vovk et al.), both per reciter and per instance
#     (Mahalanobis² of single instances against the LOO reference).
#
# Everything here is plain linear algebra so it is mirrored exactly in Octave
# (substrate_library/octave/fr_*.m) and cross-checked by frontier_crosscheck.jl / .m.

using Distributions: Chisq, quantile as dquantile

export ogk, bw_barycenter, w2_gauss, airm_dist, le_vec, le_mean, karcher_mean, conformal_threshold,
       sinkhorn_divergence, frontier_calibrate, beat_normalization, gpd_fit, pot_threshold

# ------------------------------------------------------------ SPD matrix functions ---------------
function psd_fun(C::AbstractMatrix, f)
    F = eigen(Symmetric(Matrix{Float64}(C)))
    λ = max.(F.values, 0.0)
    return Symmetric(F.vectors * Diagonal(f.(λ)) * F.vectors')
end
psd_sqrt(C) = psd_fun(C, sqrt)
psd_invsqrt(C) = psd_fun(C, x -> 1 / sqrt(max(x, 1e-300)))
logm_spd(C) = psd_fun(C, x -> log(max(x, 1e-300)))
expm_sym(S) = let F = eigen(Symmetric(Matrix{Float64}(S)))
    Symmetric(F.vectors * Diagonal(exp.(F.values)) * F.vectors')
end

# ------------------------------------------------------------ robust location / scatter ----------
"1.4826·MAD with fallbacks (IQR/1.349, then the SD) so a scale is never zero."
function rscale(v::AbstractVector)
    s = MADK * median(abs.(v .- median(v)))
    s > 1e-12 && return s
    q = quantile(v, [0.25, 0.75])
    s = (q[2] - q[1]) / 1.349
    s > 1e-12 && return s
    s = length(v) > 1 ? std(v) : 0.0
    return max(s, 1e-9)
end

"""
    ogk(X; reweight=true) -> (μ, Σ)

Orthogonalised Gnanadesikan–Kettenring estimator (Maronna & Zamar, Technometrics 44, 2002) with one
hard-rejection reweighting step at χ²_d(0.975). Rows of `X` are observations.
"""
function ogk(X::AbstractMatrix; reweight::Bool = true)
    n, d = size(X)
    med = [median(X[:, j]) for j in 1:d]
    s = [rscale(X[:, j]) for j in 1:d]
    Y = (X .- med') ./ s'
    U = Matrix{Float64}(I, d, d)
    for j in 1:d, k in j+1:d
        U[j, k] = U[k, j] = (rscale(Y[:, j] .+ Y[:, k])^2 - rscale(Y[:, j] .- Y[:, k])^2) / 4
    end
    E = eigen(Symmetric(U)).vectors
    Z = Y * E
    γ = [rscale(Z[:, l])^2 for l in 1:d]
    μz = [median(Z[:, l]) for l in 1:d]
    D = Diagonal(s)
    Σ = D * E * Diagonal(γ) * E' * D
    μ = D * (E * μz) .+ med
    if reweight && n > d + 1
        Si = inv(Symmetric(Σ + 1e-12I))
        d2 = [dot(X[i, :] .- μ, Si * (X[i, :] .- μ)) for i in 1:n]
        Σ *= median(d2) / dquantile(Chisq(d), 0.5)          # consistency at the normal
        Si = inv(Symmetric(Σ + 1e-12I))
        d2 = [dot(X[i, :] .- μ, Si * (X[i, :] .- μ)) for i in 1:n]
        keep = d2 .<= dquantile(Chisq(d), 0.975)
        if count(keep) > d + 1
            Xk = X[keep, :]
            μ = vec(mean(Xk; dims = 1))
            Σ = cov(Xk; dims = 1)
        end
    end
    return μ, Matrix(Symmetric(Σ))
end

# ------------------------------------------------------------ Gaussian geometry ------------------
"""
    bw_barycenter(ms, Cs, w) -> (m̄, C̄, iterations)

Bures–Wasserstein barycenter of N(mᵢ, Cᵢ): m̄ = Σ wᵢ mᵢ and C̄ the fixed point
C̄ = Σ wᵢ (C̄^½ Cᵢ C̄^½)^½ (Agueh & Carlier 2011), iterated as
S ← S^{-½} (Σ wᵢ (S^½ Cᵢ S^½)^½)² S^{-½} (Álvarez-Esteban et al. 2016).
"""
function bw_barycenter(ms::Vector{Vector{Float64}}, Cs::Vector{Matrix{Float64}}, w::Vector{Float64};
                       iters::Int = 200, tol::Float64 = 1e-12)
    w = w ./ sum(w)
    m = sum(w[i] * ms[i] for i in eachindex(ms))
    S = Matrix(Symmetric(sum(w[i] * Cs[i] for i in eachindex(Cs))))
    it = 0
    for k in 1:iters
        it = k
        Sh, Shi = psd_sqrt(S), psd_invsqrt(S)
        T = sum(w[i] * Matrix(psd_sqrt(Sh * Cs[i] * Sh)) for i in eachindex(Cs))
        Sn = Matrix(Symmetric(Shi * T * T * Shi))
        δ = norm(Sn - S) / max(norm(S), 1e-300)
        S = Sn
        δ < tol && break
    end
    return m, S, it
end

"Closed-form W₂² between Gaussians, split into (location², Bures shape², total²)."
function w2_gauss(m1, C1, m2, C2)
    loc = sum(abs2, m1 .- m2)
    C2h = psd_sqrt(C2)
    shape = max(tr(C1) + tr(C2) - 2tr(psd_sqrt(C2h * C1 * C2h)), 0.0)
    return loc, shape, loc + shape
end

"Affine-invariant (Fisher–Rao for zero-mean Gaussians) distance ‖log(C₁^{-½} C₂ C₁^{-½})‖_F."
function airm_dist(C1, C2)
    Ci = psd_invsqrt(C1)
    λ = eigvals(Symmetric(Ci * C2 * Ci))
    return sqrt(sum(abs2, log.(max.(λ, 1e-300))))
end

"Log-Euclidean embedding: vech(log C) with √2 on off-diagonals (an isometry onto ℝ^{d(d+1)/2})."
function le_vec(C)
    L = logm_spd(C)
    d = size(L, 1)
    return [i == j ? L[i, j] : sqrt(2) * L[i, j] for j in 1:d for i in j:d]
end

le_mean(Cs, w) = Matrix(expm_sym(sum(w[i] * Matrix(logm_spd(Cs[i])) for i in eachindex(Cs)) / sum(w)))

"Affine-invariant Fréchet (Karcher) mean by Riemannian gradient descent from the log-Euclidean mean."
function karcher_mean(Cs, w; iters::Int = 100, tol::Float64 = 1e-12)
    w = w ./ sum(w)
    M = le_mean(Cs, w)
    for _ in 1:iters
        Mh, Mhi = psd_sqrt(M), psd_invsqrt(M)
        G = sum(w[i] * Matrix(logm_spd(Mhi * Cs[i] * Mhi)) for i in eachindex(Cs))
        M = Matrix(Symmetric(Mh * expm_sym(G) * Mh))
        norm(G) < tol && break
    end
    return M
end

"""
    conformal_threshold(scores, α) -> (threshold, attainable)

Split-conformal cut: the ⌈(n+1)(1−α)⌉-th smallest score. When that index exceeds n the finite-sample
guarantee is unattainable with this many calibration scores; the maximum is returned and flagged.
"""
function conformal_threshold(scores::AbstractVector, α::Real)
    s = sort(filter(isfinite, collect(Float64, scores)))
    n = length(s)
    n == 0 && return (NaN, false)
    k = ceil(Int, (n + 1) * (1 - α))
    return k <= n ? (s[k], true) : (s[end], false)
end

# ------------------------------------------------------------ optimal transport ------------------
"Log-domain entropic OT cost ⟨P, M⟩ between histograms a, b with ground cost M."
function sinkhorn_cost(a, b, M; ε::Float64 = 0.05, iters::Int = 2000, tol::Float64 = 1e-10)
    f, g = zeros(length(a)), zeros(length(b))
    la, lb = log.(max.(a, 1e-300)), log.(max.(b, 1e-300))
    lse(v) = (mx = maximum(v); mx + log(sum(exp.(v .- mx))))
    for _ in 1:iters
        fo = copy(f)
        f = [-ε * lse((g .- M[i, :]) ./ ε .+ lb) for i in eachindex(a)]
        g = [-ε * lse((f .- M[:, j]) ./ ε .+ la) for j in eachindex(b)]
        maximum(abs.(f .- fo)) < tol && break
    end
    P = [exp((f[i] + g[j] - M[i, j]) / ε + la[i] + lb[j]) for i in eachindex(a), j in eachindex(b)]
    return sum(P .* M)
end

"""
Debiased Sinkhorn divergence S_ε(a,b) = OT_ε(a,b) − ½OT_ε(a,a) − ½OT_ε(b,b) (Feydy et al. 2019).
With `M` an articulatory ground metric over letters (makharij/sifaat distance), comparing a produced
letter-posterior histogram with the target makes near-miss substitutions (ظ for ض) cheaper than
far ones (ب for ض).
"""
sinkhorn_divergence(a, b, M; ε = 0.05) =
    sinkhorn_cost(a, b, M; ε) - (sinkhorn_cost(a, a, M; ε) + sinkhorn_cost(b, b, M; ε)) / 2

# ------------------------------------------------------------ feature selection ------------------
const _BOOKKEEPING = Set(["align_conf", "align_min_ms", "haraka_ms", "expected_ratio"])

function select_features(obs_key::Vector{Obs}, judged::Vector{String}; dmax::Int = 4, coverage::Float64 = 0.9)
    isempty(obs_key) && return String[]
    counts = Dict{String,Int}()
    for o in obs_key, (k, v) in o.metrics
        isfinite(v) && (counts[k] = get(counts, k, 0) + 1)
    end
    n = length(obs_key)
    extra = sort([k for (k, c) in counts if c >= coverage * n && !(k in _BOOKKEEPING) &&
                  !endswith(k, "_flag") && !(k in judged)])
    chosen = String[]
    for m in vcat([j for j in judged if get(counts, j, 0) >= coverage * n], extra)
        length(chosen) >= dmax && break
        cand = vcat(chosen, [m])
        rows = [o for o in obs_key if all(c -> isfinite(get(o.metrics, c, NaN)), cand)]
        length(rows) >= length(cand) + 2 || continue
        X = [o.metrics[c] for o in rows, c in cand]
        Z = (X .- [median(X[:, j]) for j in 1:size(X, 2)]') ./ [rscale(X[:, j]) for j in 1:size(X, 2)]'
        sv = svdvals(Z)
        sv[end] > 1e-6 * sv[1] * sqrt(size(Z, 1)) && push!(chosen, m)
    end
    return chosen
end

cloud(obs_key, reciter, feats) =
    let rows = [o for o in obs_key if o.reciter == reciter && all(f -> isfinite(get(o.metrics, f, NaN)), feats)]
        isempty(rows) ? zeros(0, length(feats)) : [o.metrics[f] for o in rows, f in feats]
    end

# ------------------------------------------------------------ reference model --------------------
struct GaussRef
    feats::Vector{String}
    center::Vector{Float64}
    W::Matrix{Float64}          # whitening: z = W (x − center)
    m::Vector{Float64}          # barycenter (whitened)
    C::Matrix{Float64}
    iters::Int
end

"Fit N(m, C) in whitened space with shrinkage (n C + k I)/(n + k), k = d + 1, toward the pooled scatter."
function fit_gauss(Z::AbstractMatrix)
    n, d = size(Z)
    μ, Σ = n >= 2d + 3 ? ogk(Z) : ([median(Z[:, j]) for j in 1:d], n > 1 ? cov(Z; dims = 1) : zeros(d, d))
    k = d + 1
    return μ, Matrix(Symmetric((n * Σ + k * I(d)) / (n + k)))
end

function build_ref(clouds::Dict{String,Matrix{Float64}}, anchor::String, peers::Vector{String};
                   anchor_weight::Float64 = 0.5)
    refs = [r for r in vcat([anchor], peers) if haskey(clouds, r)]
    haskey(clouds, anchor) || return nothing
    pooled = reduce(vcat, [clouds[r] .- [median(clouds[r][:, j]) for j in 1:size(clouds[r], 2)]' for r in refs])
    _, Sw = ogk(pooled)
    center = [median(clouds[anchor][:, j]) for j in 1:size(pooled, 2)]
    W = Matrix(psd_invsqrt(Sw + 1e-12I))
    ms, Cs, ws = Vector{Float64}[], Matrix{Float64}[], Float64[]
    np = length(refs) - 1
    for r in refs
        m, C = fit_gauss((clouds[r] .- center') * W')
        push!(ms, m); push!(Cs, C)
        push!(ws, r == anchor ? (np == 0 ? 1.0 : anchor_weight) : (1 - anchor_weight) / np)
    end
    m̄, C̄, it = bw_barycenter(ms, Cs, ws)
    return GaussRef(String[], center, W, m̄, C̄, it)
end

whiten(ref::GaussRef, X) = (X .- ref.center') * ref.W'

function distance(ref::GaussRef, X; β::Float64 = 1.0)
    m, C = fit_gauss(whiten(ref, X))
    loc, shape, _ = w2_gauss(m, C, ref.m, ref.C)
    a = airm_dist(ref.C, C)
    return Dict("loc2" => loc, "bures_shape2" => shape, "airm" => a, "total2" => loc + β * a^2)
end

inst_d2(ref::GaussRef, X) = let Z = whiten(ref, X), Ci = inv(Symmetric(ref.C))
    [dot(Z[i, :] .- ref.m, Ci * (Z[i, :] .- ref.m)) for i in 1:size(Z, 1)]
end

# ------------------------------------------------------------ calibration ------------------------
"""
    frontier_calibrate(obs, anchor, peers, others, spec; ...) -> Dict

Distributional reference per rule key with conformal peer-LOO thresholds at reciter level (α_pass,
α_warn) and instance level (α_inst_pass, α_inst_warn). `others` (imams, EveryAyah-only reciters, …)
are scored against the full reference. Returns the per-key model and per-reciter verdicts.
"""
function frontier_calibrate(obs::Vector{Obs}, anchor::String, peers::Vector{String}, others::Vector{String},
                            spec; conf_min::Float64 = 0.2, β::Float64 = 1.0, anchor_weight::Float64 = 0.5,
                            α_pass::Float64 = 0.2, α_warn::Float64 = 0.1, α_inst_pass::Float64 = 0.05,
                            α_inst_warn::Float64 = 0.01, dmax::Int = 4)
    dur = Set(String.(spec.duration_rules))
    collapsed = Float64(spec.collapsed_unit_ms)
    judged = Dict(String(rt) => unique(vcat([String.(g.alts) for g in groups]...)) for (rt, groups) in pairs(spec.rules))
    usable = [o for o in obs if o.rule_type in keys(judged) && reliable(o, conf_min, collapsed, o.rule_type in dur)]
    bykey = Dict{String,Vector{Obs}}()
    for o in usable
        push!(get!(bykey, o.key, Obs[]), o)
    end
    min_n = Int(spec.min_peer_n)
    everyone = unique(vcat([anchor], peers, others))
    out = Dict{String,Any}()
    for (key, ok) in sort(collect(bykey); by = first)
        rt = ok[1].rule_type
        feats = select_features([o for o in ok if o.reciter == anchor], judged[rt]; dmax)
        isempty(feats) && continue
        clouds = Dict{String,Matrix{Float64}}()
        for r in everyone
            X = cloud(ok, r, feats)
            size(X, 1) >= max(min_n, length(feats) + 2) && (clouds[r] = X)
        end
        haskey(clouds, anchor) && size(clouds[anchor], 1) >= Int(spec.min_anchor_n) || continue
        ps = [p for p in peers if haskey(clouds, p)]
        ref = build_ref(clouds, anchor, ps; anchor_weight)
        ref === nothing && continue
        # leave-one-peer-out null distributions
        loo_d, loo_inst = Dict{String,Float64}(), Float64[]
        for p in ps
            rl = build_ref(clouds, anchor, [q for q in ps if q != p]; anchor_weight)
            loo_d[p] = distance(rl, clouds[p]; β)["total2"]
            append!(loo_inst, inst_d2(rl, clouds[p]))
        end
        tp, okp = conformal_threshold(collect(values(loo_d)), α_pass)
        tw, okw = conformal_threshold(collect(values(loo_d)), α_warn)
        ip, _ = conformal_threshold(loo_inst, α_inst_pass)
        iw, _ = conformal_threshold(loo_inst, α_inst_warn)
        verdicts = Dict{String,Any}()
        for (r, X) in clouds
            dd = distance(ref, X; β)
            # held-out peers are judged by their LOO distance (the in-sample one is optimistic)
            t = haskey(loo_d, r) ? loo_d[r] : dd["total2"]
            v = isempty(loo_d) ? "UNCALIBRATED" : t <= tp ? "PASS" : t <= tw ? "WARNING" : "FAIL"
            d2 = inst_d2(ref, X)
            sc = isnan(ip) ? NaN : mean(x -> x <= ip ? 1.0 : x <= iw ? 0.5 : 0.0, d2)
            verdicts[r] = merge(dd, Dict("n" => size(X, 1), "judged_total2" => t, "verdict" => v,
                                         "instance_score" => sc, "is_reference" => r == anchor || r in ps))
        end
        out[key] = Dict("rule_type" => rt, "features" => feats, "d" => length(feats),
                        "barycenter_iters" => ref.iters, "center" => ref.center, "whitening" => ref.W,
                        "mean" => ref.m, "cov" => ref.C, "n_peers" => length(ps),
                        "thresholds" => Dict("pass" => tp, "warn" => tw, "pass_attainable" => okp,
                                             "warn_attainable" => okw, "instance_pass" => ip, "instance_warn" => iw),
                        "loo" => loo_d, "reciters" => verdicts)
    end
    return out
end

"Per-reciter summary over rule keys: share of PASS/WARN/FAIL and mean instance score."
function frontier_summary(model::Dict)
    agg = Dict{String,Dict{String,Any}}()
    for (_, km) in model, (r, v) in km["reciters"]
        a = get!(agg, r, Dict{String,Any}("PASS" => 0, "WARNING" => 0, "FAIL" => 0, "UNCALIBRATED" => 0, "scores" => Float64[]))
        a[v["verdict"]] += 1
        isfinite(v["instance_score"]) && push!(a["scores"], v["instance_score"])
    end
    return Dict(r => Dict("keys_pass" => a["PASS"], "keys_warn" => a["WARNING"], "keys_fail" => a["FAIL"],
                          "keys_uncalibrated" => a["UNCALIBRATED"],
                          "instance_index" => isempty(a["scores"]) ? NaN : 100 * mean(a["scores"])) for (r, a) in agg)
end

# ------------------------------------------------------------ B1: beat normalisation ------------
"""
    beat_normalization(obs; madd_keys, ref_key="madd_tabii") -> Dict

Falsifiable test of roadmap B1: does measuring a held sound against the *local* beat (the natural
madd of the same ayah, which by definition is 2 counts) tighten each madd class more than the ayah-
level harakah does? For every key reports the robust CV of core_ms (raw), core_ms/haraka_ms (global
counts) and core_ms/(½·median tabii core_ms in the same ayah, excluding the instance itself) (local
counts), and the Fisher separation of log-counts across keys.
"""
function beat_normalization(obs::Vector{Obs}; madd_keys = ["madd_tabii", "madd_badal", "madd_muttasil",
                            "madd_munfasil", "madd_lazim", "madd_arid_lissukun", "ghunnah"], ref_key = "madd_tabii")
    ayah(o) = (o.reciter, o.surah, o.ayah)
    tabii = Dict{Tuple{String,Int,Int},Vector{Tuple{Obs,Float64}}}()
    for o in obs
        o.key == ref_key && get(o.metrics, "core_ms", 0.0) > 0 &&
            push!(get!(tabii, ayah(o), Tuple{Obs,Float64}[]), (o, o.metrics["core_ms"]))
    end
    series = Dict(k => Dict("raw" => Float64[], "global" => Float64[], "local" => Float64[], "reciter" => String[]) for k in madd_keys)
    for o in obs
        o.key in madd_keys || continue
        c, h = get(o.metrics, "core_ms", NaN), get(o.metrics, "haraka_ms", NaN)
        (isfinite(c) && c > 0 && isfinite(h) && h > 0) || continue
        peers_t = [v for (q, v) in get(tabii, ayah(o), Tuple{Obs,Float64}[]) if q !== o]
        isempty(peers_t) && continue
        s = series[o.key]
        push!(s["raw"], c); push!(s["global"], c / h); push!(s["local"], c / (median(peers_t) / 2))
        push!(s["reciter"], o.reciter)
    end
    rcv_(v) = isempty(v) ? NaN : rscale(v) / abs(median(v))
    function fisher(kind)
        groups = [log.(series[k][kind]) for k in madd_keys if length(series[k][kind]) >= 5]
        length(groups) < 2 && return NaN
        allv = vcat(groups...)
        μ = mean(allv)
        between = sum(length(g) * (mean(g) - μ)^2 for g in groups)
        within = sum(sum(abs2, g .- mean(g)) for g in groups)
        return between / max(within, 1e-12)
    end
    per = Dict(k => Dict("n" => length(s["raw"]), "rcv_raw" => rcv_(s["raw"]), "rcv_global" => rcv_(s["global"]),
                         "rcv_local" => rcv_(s["local"]),
                         "median_global" => isempty(s["global"]) ? NaN : median(s["global"]),
                         "median_local" => isempty(s["local"]) ? NaN : median(s["local"])) for (k, s) in series)
    return Dict("per_key" => per, "fisher_raw" => fisher("raw"), "fisher_global" => fisher("global"),
                "fisher_local" => fisher("local"), "series" => series)
end

# ------------------------------------------------------------ extreme-value tail thresholds -------
# Roadmap A6. PASS/WARN/FAIL are *tail* events, and the tail is not Gaussian: at z = 3 a normal puts
# 0.13 % of mass beyond the cut but a t(ν=5) puts 1.5 % — an order of magnitude of miscalibration that
# drifts from rule to rule. Peaks-Over-Threshold (Pickands–Balkema–de Haan) says the excesses over a
# high threshold u converge to a Generalised Pareto distribution, so we fit that tail directly and read
# the cut off it at a *controlled exceedance risk* rather than a nominal sigma count.
#
# Fitting uses Grimshaw's reparameterisation θ = ξ/σ, which collapses the two-parameter MLE onto a
# one-dimensional profile likelihood (numerically far better behaved than optimising (ξ, σ) jointly):
#     ξ(θ) = (1/n) Σ log(1 + θ yᵢ),   σ(θ) = ξ(θ)/θ,   ℓ(θ) = −n log σ(θ) − (1 + 1/ξ(θ)) Σ log(1 + θ yᵢ)
# The quantile then follows Siffer et al. (SPOT, KDD 2017 doi:10.1145/3097983.3098144):
#     q_z = u + (σ/ξ)[(z·n/N_u)^{−ξ} − 1]
# ξ itself is a *reportable diagnostic*: ξ > 0 means a heavy right tail (durations stretched
# deliberately — madd elongation), ξ ≈ 0 exponential, ξ < 0 a bounded tail.

"""
    gpd_fit(y) -> (ξ, σ)

MLE of the Generalised Pareto shape ξ and scale σ for strictly positive excesses `y`, via Grimshaw's
one-dimensional profile likelihood. Falls back to the exponential fit (ξ = 0, σ = mean(y)) when the
profile is degenerate.
"""
function gpd_fit(y::AbstractVector)
    v = sort(filter(x -> isfinite(x) && x > 0, collect(Float64, y)))
    n = length(v)
    n < 5 && return (0.0, n == 0 ? NaN : mean(v))
    ymax, ymin = v[end], v[1]

    # ℓ(θ) on the admissible range 1 + θ y > 0  ⇒  θ > −1/ymax
    function profile(θ)
        (θ == 0 || 1 + θ * ymax <= 1e-12) && return -Inf
        s = 0.0
        @inbounds for x in v
            u = 1 + θ * x
            u <= 1e-12 && return -Inf
            s += log(u)
        end
        ξ = s / n
        (abs(ξ) < 1e-12 || ξ / θ <= 0) && return -Inf
        return -n * log(ξ / θ) - (1 + 1 / ξ) * s
    end

    lo, hi = -1 / ymax + 1e-9, 2 / max(ymin, 1e-9)
    grid = vcat(range(lo, -1e-9; length = 200), range(1e-9, hi; length = 400))
    best, θbest = -Inf, 0.0
    for θ in grid
        l = profile(θ)
        l > best && (best = l; θbest = θ)
    end
    best == -Inf && return (0.0, mean(v))

    # golden-section refine around the grid winner
    step = (hi - lo) / 600
    a, b = θbest - step, θbest + step
    φ = (sqrt(5) - 1) / 2
    c, d = b - φ * (b - a), a + φ * (b - a)
    for _ in 1:80
        if profile(c) > profile(d)
            b, d = d, c
            c = b - φ * (b - a)
        else
            a, c = c, d
            d = a + φ * (b - a)
        end
    end
    θ = (a + b) / 2
    profile(θ) == -Inf && return (0.0, mean(v))
    ξ = sum(log1p(θ * x) for x in v) / n
    σ = ξ / θ
    (isfinite(ξ) && isfinite(σ) && σ > 0) || return (0.0, mean(v))
    return (ξ, σ)
end

"""
    pot_threshold(scores, z; u_quantile=0.95) -> (q, ξ, σ, u, n_exceed)

Peaks-Over-Threshold cut at exceedance risk `z` (e.g. 1e-2 = one false FAIL per hundred instances).
`u` is the empirical `u_quantile` of `scores`; the GPD is fitted to the excesses above it. Returns the
threshold together with the fitted tail so callers can report ξ as a non-Gaussianity diagnostic.
"""
function pot_threshold(scores::AbstractVector, z::Real; u_quantile::Float64 = 0.95)
    s = sort(filter(isfinite, collect(Float64, scores)))
    n = length(s)
    n < 10 && return (n == 0 ? NaN : s[end], 0.0, NaN, NaN, 0)
    u = quantile(s, u_quantile)
    exc = [x - u for x in s if x > u]
    nu = length(exc)
    nu < 5 && return (s[end], 0.0, NaN, u, nu)
    ξ, σ = gpd_fit(exc)
    r = z * n / nu
    q = abs(ξ) < 1e-8 ? u - σ * log(r) : u + (σ / ξ) * (r^(-ξ) - 1)
    return (isfinite(q) ? q : s[end], ξ, σ, u, nu)
end
