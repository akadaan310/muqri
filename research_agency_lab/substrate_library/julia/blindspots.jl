# Blind spots of the acoustic model: where professional reciters "fail" a check together, the check
# cannot be trusted there -- and the same context elsewhere is suspect too.
#
# Input: research_agency_lab/experiments/quran/checks_T300.json.gz (modal_profile.py::checks): every
# scored check of every T300 recitation (41 reciters, 11,995 verses) aggregated by CONTEXT -- letter,
# check (a characteristic head or letter identity), the two sounds before, the sound after,
# voweled / saakin / doubled / stop, position in the word, ayah-final word -- and by WORD, in two
# reciter halves (fold 0 / 1). A professional's failure is overwhelmingly the model's, not the
# reciter's: that is the premise, and it is checked against the certified reciter's own takes.
#
#  1. base rates per (letter, check), with Wilson bounds;
#  2. per check, a regularised logistic model of failure on the context features, with the
#     letter x previous-sound and letter x next-sound interactions; fitted on one half, judged on the
#     other against the letter-only baseline (held-out log-loss and AUC);
#  3. the dependence map: for each check, the conditional mutual information I(feature; failure |
#     letter) -- how much each context feature explains beyond the letter itself;
#  4. blind-spot contexts (predicted master failure >= TAU) and words the context model leaves
#     unexplained (word-specific residuals);
#  5. validation: on the held-out half, the share of professional failures that suppression removes
#     against the share of checks it gives up.
#
#   ~/julia-1.11.5/bin/julia --project=research_agency_lab/substrate_library/julia \
#       research_agency_lab/substrate_library/julia/blindspots.jl

using JSON3, Optim, Statistics, LinearAlgebra

const ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
const DATA = joinpath(ROOT, "research_agency_lab/experiments/quran/checks_T300.json.gz")
const OUT = joinpath(ROOT, "research_agency_lab/experiments/quran/blindspots.json")
const TAU = 0.25
const LAMBDA = 2.0
const FEATURES = ["sym", "prev2", "prev", "next", "ctx", "wpos", "stop", "sym_prev", "sym_next"]

struct Row
    fold::Int
    f::Vector{String}      # the feature values, in FEATURES order
    n::Float64
    k::Float64
end

function load()
    raw = JSON3.read(read(`gzip -dc $DATA`, String))
    rows = Dict{String, Vector{Row}}()
    for (key, v) in pairs(raw.ctx)
        p = split(String(key), "|")
        fold, sym, head, prev2, prev, nxt, ctx, wpos, stop = p
        f = [String(sym), String(prev2), String(prev), String(nxt), String(ctx), String(wpos), String(stop),
             sym * ">" * prev, sym * "<" * nxt]
        push!(get!(rows, String(head), Row[]), Row(parse(Int, fold), f, Float64(v[1]), Float64(v[2])))
    end
    words = Dict{String, Vector{Float64}}()
    for (key, v) in pairs(raw.word)
        p = split(String(key), "|")
        wk = join(p[2:end], "|")
        a = get!(words, wk, [0.0, 0.0])
        a[1] += v[1]; a[2] += v[2]
    end
    return rows, words
end

sig(z) = 1 / (1 + exp(-z))

"Index maps for each feature's values seen in `rows` (value -> column)."
function design(rows::Vector{Row}, use::Vector{Int})
    cols = Dict{Tuple{Int, String}, Int}()
    for r in rows, j in use
        get!(cols, (j, r.f[j]), length(cols) + 2)      # column 1 = intercept
    end
    X = [Int[get(cols, (j, r.f[j]), 0) for j in use] for r in rows]
    return cols, X
end

"Binomial logistic regression on aggregated counts, L2 on everything but the intercept."
function fitlogit(rows::Vector{Row}, use::Vector{Int})
    cols, X = design(rows, use)
    P = length(cols) + 1
    n = [r.n for r in rows]; k = [r.k for r in rows]
    function fg!(F, G, b)
        z = [b[1] + sum((b[c] for c in x if c > 0); init = 0.0) for x in X]
        p = sig.(z)
        if G !== nothing
            G .= 0.0
            for (i, x) in enumerate(X)
                g = n[i] * p[i] - k[i]
                G[1] += g
                for c in x
                    c > 0 && (G[c] += g)
                end
            end
            @views G[2:end] .+= LAMBDA .* b[2:end]
        end
        if F !== nothing
            ll = sum(k .* log.(p .+ 1e-12) .+ (n .- k) .* log.(1 .- p .+ 1e-12))
            return -ll + LAMBDA / 2 * sum(abs2, @view b[2:end])
        end
        nothing
    end
    b0 = zeros(P)
    b0[1] = log((sum(k) + 0.5) / (sum(n) - sum(k) + 0.5))
    res = optimize(b -> fg!(0.0, nothing, b), (G, b) -> fg!(nothing, G, b), b0, LBFGS(), Optim.Options(iterations = 400))
    return cols, Optim.minimizer(res)
end

predict(cols, b, r::Row, use) = sig(b[1] + sum((b[get(cols, (j, r.f[j]), 1)] * (haskey(cols, (j, r.f[j])) ? 1 : 0)
                                               for j in use); init = 0.0))

"Held-out log-loss per instance and AUC (instance-weighted: failures against passes)."
function heldout(pred::Vector{Float64}, rows::Vector{Row})
    n = [r.n for r in rows]; k = [r.k for r in rows]
    ll = -sum(k .* log.(pred .+ 1e-12) .+ (n .- k) .* log.(1 .- pred .+ 1e-12)) / sum(n)
    o = sortperm(pred)
    pos, neg = k[o], (n .- k)[o]
    # AUC = P(a failure is scored above a pass), ties half
    cumneg, auc, i = 0.0, 0.0, 1
    while i <= length(o)
        j = i
        while j < length(o) && pred[o[j + 1]] == pred[o[i]]
            j += 1
        end
        tp, tn = sum(pos[i:j]), sum(neg[i:j])
        auc += tp * (cumneg + tn / 2)
        cumneg += tn
        i = j + 1
    end
    return ll, auc / (sum(pos) * sum(neg))
end

"I(feature; failure | letter) in bits, from aggregated counts."
function cmi(rows::Vector{Row}, j::Int)
    by = Dict{String, Vector{Row}}()
    for r in rows
        push!(get!(by, r.f[1], Row[]), r)
    end
    N = sum(r.n for r in rows)
    tot = 0.0
    for (_, rs) in by
        Ns = sum(r.n for r in rs); Ks = sum(r.k for r in rs)
        ps = Ks / Ns
        (ps <= 0 || ps >= 1) && continue
        hs = -(ps * log2(ps) + (1 - ps) * log2(1 - ps))
        g = Dict{String, Vector{Float64}}()
        for r in rs
            a = get!(g, r.f[j], [0.0, 0.0]); a[1] += r.n; a[2] += r.k
        end
        hc = 0.0
        for (_, (n, k)) in g
            q = k / n
            (q <= 0 || q >= 1) && continue
            hc += n / Ns * -(q * log2(q) + (1 - q) * log2(1 - q))
        end
        tot += Ns / N * (hs - hc)
    end
    return tot
end

function wilson(k, n; z = 1.96)
    n == 0 && return (0.0, 0.0)
    p = k / n; d = 1 + z^2 / n
    c = (p + z^2 / (2n)) / d; h = z * sqrt(p * (1 - p) / n + z^2 / (4n^2)) / d
    return (c - h, c + h)
end

function main()
    rows, words = load()
    heads = sort(collect(keys(rows)))
    full = collect(1:length(FEATURES))
    report = Dict{String, Any}()
    println("check                 instances  failure   held-out log-loss  (letter only -> context)   AUC (letter -> context)")
    for h in heads
        R = rows[h]
        n = sum(r.n for r in R); k = sum(r.k for r in R)
        k < 30 && continue
        tr = [r for r in R if r.fold == 0]; te = [r for r in R if r.fold == 1]
        cb, bb = fitlogit(tr, [1]); cf, bf = fitlogit(tr, full)
        llb, aucb = heldout([predict(cb, bb, r, [1]) for r in te], te)
        llf, aucf = heldout([predict(cf, bf, r, full) for r in te], te)
        println(rpad(h, 22), lpad(Int(n), 9), "  ", rpad(round(k / n; digits = 4), 8), "   ",
                round(llb; digits = 4), " -> ", round(llf; digits = 4), "                ", round(aucb; digits = 3), " -> ", round(aucf; digits = 3))

        # suppression on the held-out half: checks predicted >= TAU are not scored
        pt = [predict(cf, bf, r, full) for r in te]
        sup = pt .>= TAU
        kept_fail = sum(r.k for (r, s) in zip(te, sup) if !s; init = 0.0)
        tot_fail = sum(r.k for r in te)
        gave_up = sum(r.n for (r, s) in zip(te, sup) if s; init = 0.0) / sum(r.n for r in te)

        # the dependence map
        dep = Dict(FEATURES[j] => round(cmi(R, j); digits = 5) for j in 2:length(FEATURES))

        # fit on everything for serving and for the blind-spot list
        ca, ba = fitlogit(R, full)
        spots = []
        for r in R
            p = predict(ca, ba, r, full)
            if p >= TAU && r.n >= 5
                lo, hi = wilson(r.k, r.n)
                push!(spots, Dict("letter" => r.f[1], "prev2" => r.f[2], "prev" => r.f[3], "next" => r.f[4],
                                  "context" => r.f[5], "word_position" => r.f[6], "ayah" => r.f[7], "n" => r.n,
                                  "failures" => r.k, "rate" => round(r.k / r.n; digits = 3), "wilson" => [round(lo; digits = 3), round(hi; digits = 3)],
                                  "predicted" => round(p; digits = 3)))
            end
        end
        sort!(spots; by = s -> -s["failures"])
        report[h] = Dict("instances" => n, "failures" => k,
                         "heldout" => Dict("logloss_letter" => llb, "logloss_context" => llf, "auc_letter" => aucb, "auc_context" => aucf,
                                           "failures_removed" => 1 - kept_fail / max(tot_fail, 1), "checks_given_up" => gave_up),
                         "dependence_bits" => dep,
                         "coef" => Dict("intercept" => ba[1],
                                        "terms" => Dict(string(FEATURES[j], "=", v) => ba[c] for ((j, v), c) in ca)),
                         "blind_spots" => spots[1:min(60, length(spots))])
    end

    println("\nheld-out suppression at predicted master failure >= ", TAU, ":")
    for h in sort(collect(keys(report)))
        v = report[h]["heldout"]
        println("   ", rpad(h, 22), " professional failures removed ", rpad(round(v["failures_removed"]; digits = 3), 6),
                "  checks given up ", round(v["checks_given_up"]; digits = 4))
    end
    println("\ndependence map: I(feature; failure | letter), bits x 1000")
    for h in sort(collect(keys(report)))
        d = report[h]["dependence_bits"]
        top = sort(collect(d); by = x -> -x[2])[1:3]
        println("   ", rpad(h, 22), join(["$(a) $(round(1000b; digits = 2))" for (a, b) in top], "   "))
    end

    # words the context model does not explain: master failure >= 0.3 over >= 10 checks
    wlist = []
    for (wk, (n, k)) in words
        n >= 10 && k / n >= 0.3 && push!(wlist, Dict("key" => wk, "n" => n, "failures" => k, "rate" => round(k / n; digits = 3)))
    end
    sort!(wlist; by = w -> -w["failures"])
    println("\nwords where masters fail a check >= 30 % (n >= 10): ", length(wlist))
    for w in wlist[1:min(15, length(wlist))]
        println("   ", w["key"], "   ", Int(w["failures"]), "/", Int(w["n"]))
    end
    # parity fixture: predictions of the served model on real contexts, for the numpy path to reproduce
    parity = []
    for h in sort(collect(keys(report)))
        c = report[h]["coef"]
        for r in rows[h][1:min(25, length(rows[h]))]
            z = c["intercept"] + sum(get(c["terms"], string(FEATURES[j], "=", r.f[j]), 0.0) for j in 1:length(FEATURES))
            push!(parity, Dict("check" => h, "features" => r.f, "p" => sig(z)))
        end
    end
    open(OUT, "w") do io
        JSON3.pretty(io, Dict("tau" => TAU, "lambda" => LAMBDA, "features" => FEATURES, "checks" => report, "words" => wlist,
                             "parity" => parity))
    end
    println("\n-> ", OUT)
end

main()
