# The timing calculus: every stretching as a multiple of the reciter's own count, learned from the
# masters' own recitations.
#
# Per verse (research_agency_lab/experiments/timing/stretch_T300.jsonl, app/stretch.py): the plain
# voweled letters (consonant + short vowel, nothing held) with their times and durations, and every
# stretching instance (madd, ghunnah, ikhfa, ...) with its time and duration. The reciter's count unit
# u(t) is estimated from the plain letters; a stretching's STRETCH is rho = seconds / u(t).
#
#  1. Unit estimators: the verse median, the geometric mean, and LOCAL windows (median of the k
#     plain letters nearest in time). The best estimator makes a fixed-length stretching -- madd
#     tabii, 2 counts everywhere -- most even within each reciter (smallest within-reciter spread of
#     log rho) and independent of tempo (rho uncorrelated with u across reciters).
#  2. Norms: for every rule and level, the typical stretch rho_ref and the spread sigma (within-reciter
#     SD of log rho, pooled) -- the human looseness a balanced reading keeps. Munfasil / silah kubra take
#     the verse's wajh (qasr 2 / tawassut 4-5); madd 'arid and leen take each reciter's own level (2, 4
#     or 6, placed against that reciter's tabii).
#  3. Evenness: the reciters ranked by how evenly they hold the tabii and how exactly their long
#     stretchings are multiples of it.
#  4. Leave-one-reciter-out: each reciter graded by norms fitted on the other 40, at |z| > Z, against
#     the engine's current windows.
#
#   ~/julia-1.11.5/bin/julia --project=research_agency_lab/substrate_library/julia \
#       research_agency_lab/substrate_library/julia/stretch.jl

using JSON3, Statistics

const ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
const DATA = joinpath(ROOT, "research_agency_lab/experiments/timing/stretch_T300.jsonl")
const TIERS = joinpath(ROOT, "research_agency_lab/experiments/timing/tiers.json")
const OUT = joinpath(ROOT, "research_agency_lab/experiments/timing/stretch_model.json")
const Z = 2.5
const FIXED = Dict("madd_tabii" => 2.0, "madd_badal" => 2.0, "madd_silah_sughra" => 2.0, "madd_iwad" => 2.0,
                   "madd_muttasil" => 4.5, "madd_lazim" => 6.0, "tawassut" => 1.5, "ghunnah" => 2.0,
                   "ikhfa" => 2.0, "ikhfa_shafawi" => 2.0, "idgham_shafawi" => 2.0, "iqlab" => 2.0,
                   "idgham_ghunnah" => 2.0)
const WAJH = ("madd_munfasil", "madd_silah_kubra")
const CHOSEN = ("madd_arid_lissukun", "madd_leen")          # the reciter's own level: 2, 4 or 6
const ONE_SIDED = ("madd_arid_lissukun",)                     # at a stop: graded short only

struct Inst
    speaker::String
    rule::String
    t::Float64
    secs::Float64
    wajh::String
    status::String
    plain_t::Vector{Float64}
    plain_d::Vector{Float64}
end

function load()
    tiers = JSON3.read(read(TIERS, String))
    insts = Inst[]
    for line in eachline(DATA)
        r = JSON3.read(line)
        haskey(r, :error) && continue
        pt = Float64[p[1] for p in r.plain]
        pd = Float64[p[2] for p in r.plain]
        length(pd) < 3 && continue
        for s in r.stretch
            w = s.wajh === nothing ? "" : (occursin("qasr", String(s.wajh)) ? "qasr" : "tawassut")
            push!(insts, Inst(String(r.speaker), String(s.rule), s.t, s.seconds, w, String(s.status), pt, pd))
        end
    end
    return insts, Dict(String(k) => String(v) for (k, v) in pairs(tiers))
end

gmean(x) = exp(mean(log.(x)))
function local_median(pt, pd, t, k)
    k >= length(pd) && return median(pd)
    idx = partialsortperm(abs.(pt .- t), 1:k)
    return median(pd[idx])
end

const ESTIMATORS = [("verse median", (i) -> median(i.plain_d)),
                    ("verse geometric mean", (i) -> gmean(i.plain_d)),
                    ("local k=3", (i) -> local_median(i.plain_t, i.plain_d, i.t, 3)),
                    ("local k=5", (i) -> local_median(i.plain_t, i.plain_d, i.t, 5)),
                    ("local k=7", (i) -> local_median(i.plain_t, i.plain_d, i.t, 7)),
                    ("local k=9", (i) -> local_median(i.plain_t, i.plain_d, i.t, 9))]

"Within-reciter SD of log rho, pooled over reciters (the evenness a unit estimator allows)."
function pooled_within_sd(sp::Vector{String}, lr::Vector{Float64})
    ss, df = 0.0, 0
    for s in unique(sp)
        x = lr[sp .== s]
        length(x) < 5 && continue
        ss += sum((x .- mean(x)) .^ 2)
        df += length(x) - 1
    end
    return sqrt(ss / df)
end

function level_of(i::Inst, tabii_ref::Dict{String, Float64}, rho::Float64)
    haskey(FIXED, i.rule) && return FIXED[i.rule]
    if i.rule in WAJH
        return i.wajh == "qasr" ? 2.0 : 4.5
    end
    if i.rule in CHOSEN            # the level this reciter chose: rho against their own tabii
        rel = 2 * rho / get(tabii_ref, i.speaker, rho)
        return rel < 3.0 ? 2.0 : (rel < 5.0 ? 4.0 : 6.0)
    end
    return NaN
end

function main()
    insts, tiers = load()
    println(length(insts), " stretching instances, ", length(unique(i.speaker for i in insts)), " reciters")

    # 1. unit estimators, judged on the madd tabii
    tab = [i for i in insts if i.rule == "madd_tabii"]
    sp = [i.speaker for i in tab]
    println("\n1. unit estimators on ", length(tab), " madd tabii (lower spread = more even; corr ~ 0 = tempo-free)")
    println("   seconds, no unit:            within-reciter sd(log) ", round(pooled_within_sd(sp, log.([i.secs for i in tab])); digits = 4))
    results = []
    for (name, f) in ESTIMATORS
        u = [f(i) for i in tab]
        lr = log.([i.secs for i in tab] ./ u)
        sd = pooled_within_sd(sp, lr)
        # across reciters: does their median stretch depend on their tempo?
        rs = unique(sp)
        mu = [median(u[sp .== s]) for s in rs]
        mr = [median(lr[sp .== s]) for s in rs]
        c = cor(log.(mu), mr)
        push!(results, (name, sd, c))
        println("   ", rpad(name, 22), "      within-reciter sd(log) ", round(sd; digits = 4), "   corr(tempo, stretch) ", round(c; digits = 3))
    end
    best = results[argmin([r[2] for r in results])][1]
    unit = Dict(ESTIMATORS)[best]
    println("   -> ", best)

    # 2. stretches and norms
    rho = [i.secs / unit(i) for i in insts]
    tabii_ref = Dict{String, Float64}()
    for s in unique(i.speaker for i in insts)
        x = [rho[j] for (j, i) in enumerate(insts) if i.speaker == s && i.rule == "madd_tabii"]
        isempty(x) || (tabii_ref[s] = median(x))
    end
    lvl = [level_of(i, tabii_ref, rho[j]) for (j, i) in enumerate(insts)]
    keys_ = unique([(i.rule, lvl[j]) for (j, i) in enumerate(insts) if !isnan(lvl[j])])

    # 3. evenness: how evenly each reciter holds the tabii
    reciters = sort(unique(i.speaker for i in insts))
    even = Dict{String, Float64}()
    for s in reciters
        x = log.([rho[j] for (j, i) in enumerate(insts) if i.speaker == s && i.rule == "madd_tabii"])
        length(x) >= 10 && (even[s] = std(x))
    end
    ranked = sort(collect(even); by = last)
    println("\n3. evenness of the tabii (sd of log stretch within the reciter), most even first:")
    for (s, v) in ranked[1:10]
        println("   ", rpad(s, 44), round(v; digits = 3), "  ", get(tiers, s, ""))
    end
    reference = Set(first.(ranked[1:min(15, length(ranked))]))     # the 15 most even reciters

    function fit(exclude::String)
        norms = Dict{Tuple{String, Float64}, Tuple{Float64, Float64, Int}}()
        for k in keys_
            sel = [j for (j, i) in enumerate(insts) if (i.rule, lvl[j]) == k && i.speaker != exclude]
            ref = [j for j in sel if insts[j].speaker in reference]
            length(ref) < 10 && (ref = sel)
            length(ref) < 10 && continue
            lr = log.(rho[ref])
            spk = [insts[j].speaker for j in ref]
            sd = pooled_within_sd(spk, lr)
            isnan(sd) && (sd = std(lr))
            norms[k] = (exp(median(lr)), sd, length(ref))
        end
        return norms
    end

    norms = fit("")
    println("\n2. norms (the 15 most even reciters): rule, level -> typical stretch in units, within-reciter spread")
    for k in sort(collect(keys(norms)))
        r, s, n = norms[k]
        println("   ", rpad(k[1], 20), rpad(k[2], 5), " stretch ", round(r; digits = 2), "  sd(log) ", round(s; digits = 3), "  n ", n)
    end

    # 2b. tempo: a reciter who slows down lengthens the held sounds more than the plain syllables,
    # so the expected stretch is a function of the reciter's own unit: log rho = a + b log(u / U0)
    U0 = median([unit(i) for i in insts])
    uv = [unit(i) for i in insts]
    function fit_tempo(exclude::String)
        m = Dict{Tuple{String, Float64}, NTuple{5, Float64}}()
        for k in keys_
            sel = [j for (j, i) in enumerate(insts) if (i.rule, lvl[j]) == k && i.speaker != exclude]
            ref = [j for j in sel if insts[j].speaker in reference]
            length(ref) < 30 && (ref = sel)
            length(ref) < 30 && continue
            x = log.(uv[ref] ./ U0); y = log.(rho[ref])
            b = cov(x, y) / var(x); a = mean(y) - b * mean(x)
            res = y .- (a .+ b .* x)
            # two spreads: the upper tail carries very long holds and pause absorbed at stops, which
            # inflate a symmetric SD and hide a genuine halving; the lower half cannot contain them.
            # short: judged on the lower half's spread (RMS below the centre); long: on the SD
            lower = res[res .< 0]
            sd_low = length(lower) >= 10 ? sqrt(mean(lower .^ 2)) : std(res)
            m[k] = (a, b, std(res), sd_low, length(ref))
        end
        return m
    end
    tm = fit_tempo("")
    SD_FLOOR = tm[("madd_tabii", 2.0)][3]
    println("\n2b. tempo-aware norms: log stretch = a + b log(unit / ", round(U0; digits = 3), " s); residual sd (all human spread)")
    for k in sort(collect(keys(tm)))
        a, b, s, sl, n = tm[k]
        println("   ", rpad(k[1], 20), rpad(k[2], 5), " stretch at U0 ", round(exp(a); digits = 2), "  tempo slope ", round(b; digits = 2),
                "  sd ", round(s; digits = 3), "  sd below ", round(sl; digits = 3), "  n ", n)
    end

    # the multiples question: long stretchings as multiples of the reciter's OWN tabii
    println("\n2c. multiples of the reciter's own tabii (median over reciters, with the reciters' IQR):")
    for k in sort(collect(keys(tm)))
        k[1] == "madd_tabii" && continue
        rr = Float64[]
        for s in reciters
            x = [rho[j] for (j, i) in enumerate(insts) if i.speaker == s && (i.rule, lvl[j]) == k]
            length(x) >= 5 && haskey(tabii_ref, s) && push!(rr, median(x) / tabii_ref[s])
        end
        length(rr) >= 5 || continue
        q = quantile(rr, [0.25, 0.5, 0.75])
        println("   ", rpad(k[1], 20), rpad(k[2], 5), " x tabii ", round(q[2]; digits = 2), "  (", round(q[1]; digits = 2), "-", round(q[3]; digits = 2),
                ")   nominal ratio ", round(k[2] / 2; digits = 2), "   reciters ", length(rr))
    end

    # 4. leave-one-reciter-out against the engine's current windows
    println("\n4. leave-one-reciter-out: share flagged (|z| > ", Z, ") vs the current windows")
    flag_new = Dict{String, Vector{Bool}}(); flag_old = Dict{String, Vector{Bool}}(); flag_tmp = Dict{String, Vector{Bool}}()
    for s in reciters
        nm = fit(s)
        nt = fit_tempo(s)
        for (j, i) in enumerate(insts)
            i.speaker == s || continue
            k = (i.rule, lvl[j])
            haskey(nm, k) || continue
            r, sd, _ = nm[k]
            z = (log(rho[j]) - log(r)) / sd
            push!(get!(flag_new, i.rule, Bool[]), abs(z) > Z)
            if i.rule in CHOSEN            # 2, 4 and 6 are all legitimate: far from every level
                # a level's own spread is truncated by assigning levels by thresholds: floor it at the
                # tabii's, the spread any single stretch measurement carries
                zs = [(log(rho[j]) - v[1] - v[2] * log(uv[j] / U0)) / max(v[3], SD_FLOOR) for (kk, v) in nt if kk[1] == i.rule]
                # graded short only for the 'arid: at a stop the madd runs into the pause and the
                # alignment gives it part of the silence (the flagged ones sat at 9-13 units, twice the
                # six-count level), so length past six is not separable from the stop
                zmin = zs[argmin(abs.(zs))]
                isempty(zs) || push!(get!(flag_tmp, i.rule, Bool[]), i.rule in ONE_SIDED ? zmin < -Z : abs(zmin) > Z)
            elseif haskey(nt, k)
                a, b, sdt, sdl, _ = nt[k]
                d = log(rho[j]) - a - b * log(uv[j] / U0)
                push!(get!(flag_tmp, i.rule, Bool[]), d < 0 ? d / sdl < -Z : d / sdt > Z)
            end
            push!(get!(flag_old, i.rule, Bool[]), i.status in ("short", "long"))
        end
    end
    for r in sort(collect(keys(flag_new)))
        println("   ", rpad(r, 20), " n ", rpad(length(flag_new[r]), 6), " flagged now ", round(mean(flag_old[r]); digits = 3),
                "  calculus ", round(mean(flag_new[r]); digits = 3),
                "  tempo-aware ", haskey(flag_tmp, r) ? round(mean(flag_tmp[r]); digits = 3) : "-")
    end

    out = Dict("unit_estimator" => best, "z_threshold" => Z,
               "estimators" => [Dict("name" => r[1], "within_reciter_sd_log" => r[2], "corr_tempo_stretch" => r[3]) for r in results],
               "reference_reciters" => sort(collect(reference)),
               "evenness" => [Dict("reciter" => s, "tabii_sd_log" => v, "tier" => get(tiers, s, "")) for (s, v) in ranked],
               "U0" => U0, "sd_floor_multilevel" => SD_FLOOR, "multilevel_rules" => collect(CHOSEN),
               "one_sided_rules" => collect(ONE_SIDED), "fixed_levels" => FIXED,
               "tempo_norms" => [Dict("rule" => k[1], "level" => k[2], "a" => v[1], "b" => v[2], "sd" => v[3], "sd_low" => v[4], "n" => v[5])
                                 for (k, v) in sort(collect(tm))],
               "norms" => [Dict("rule" => k[1], "level" => k[2], "stretch" => v[1], "sd_log" => v[2], "n" => v[3])
                           for (k, v) in sort(collect(norms))])
    # parity fixture: the z of the first instances of each rule under the final model, for the
    # numpy path (app/stretch.py) to reproduce
    fx = []
    for r in sort(unique(i.rule for i in insts))
        for j in findall(i -> i.rule == r, insts)[1:min(5, count(i -> i.rule == r, insts))]
            i = insts[j]
            push!(fx, Dict("rule" => r, "seconds" => i.secs, "plain" => i.plain_d, "wajh" => i.wajh,
                           "unit" => uv[j], "stretch" => rho[j]))
        end
    end
    open(joinpath(dirname(OUT), "stretch_parity.json"), "w") do io
        JSON3.write(io, fx)
    end
    open(OUT, "w") do io
        JSON3.pretty(io, out)
    end
    println("\n-> ", OUT)
end

main()
