# The synthesis pilot: an hour of Husary chosen as PASSAGES that cover the most of the Quran's
# phenomena -- every rule, letter context, pair, repetition, waqf form, every one of the 29 named rare
# cases (sakt, ishmam, imala, tashil, ...) and every sound-to-sound transition (diphone) -- instead of
# one surah range. Passages rather than single ayahs, because a synthesizer and its evaluation need
# connected recitation: the tempo held across ayahs, madd lengths kept equal (taswiyah), the stop and
# restart at every ayah end.
#
# A candidate is a run of consecutive ayahs in one surah lasting MIN_S..MAX_S seconds of Husary. Key
# family f has need n_f and weight w_f; key k needs need_k = min(n_f, total_k) and a candidate p gains
#     g(p) = sum_k w_f / need_k * min(c_pk, max(0, need_k - got_k))      per second of audio.
# F(S) is monotone submodular, so the greedy is within (1 - 1/e) of the best set of the same cost
# (for unit costs; per-second greedy is the standard cost-benefit variant). Chosen passages are
# disjoint. Every rare case first gets its best passage; when all needs are met they double (capped
# at the Quran total) and the greedy continues until the budget is spent. A held-out test set is then
# chosen the same way from the ayahs left, with needs of 1.
#
#   ~/julia-1.11.5/bin/julia --project=research_agency_lab/substrate_library/julia datasets/qaari_keys/synth_select.jl

using JSON3

const BUILD = joinpath(@__DIR__, "build")
const SYN = joinpath(BUILD, "synth")
# phase 1: connected passages, long enough to measure tempo and madd consistency across ayahs;
# phase 2: short passages that fill what the long ones left uncovered; then the held-out test set
const LONG = (min_s = 60.0, max_s = 240.0, budget = 2400.0)
const FILL = (min_s = 10.0, max_s = 60.0, budget = 1200.0)
const TEST = (min_s = 20.0, max_s = 90.0, budget = 300.0)
const NEED = Dict("rule" => 6, "letter" => 6, "pair" => 4, "rep" => 6, "special" => 1, "waqf" => 6, "diphone" => 3)
const WEIGHT = Dict("rule" => 1.0, "letter" => 0.5, "pair" => 1.5, "rep" => 1.0, "special" => 2.0, "waqf" => 0.5,
                    "diphone" => 1.0)

function load()
    ay = [parse.(Int, split(l, '\t')) for l in eachline(joinpath(BUILD, "ayahs.tsv"))]
    names = vcat(readlines(joinpath(BUILD, "keys.txt")), readlines(joinpath(SYN, "diphone_keys.txt")))
    nk0 = length(readlines(joinpath(BUILD, "keys.txt")))
    rows = [Dict{Int, Int}() for _ in ay]
    for l in eachline(joinpath(BUILD, "ayah_keys.tsv"))
        a, k, v = parse.(Int, split(l, '\t'))
        rows[a][k] = v
    end
    for l in eachline(joinpath(SYN, "ayah_diphones.tsv"))
        a, k, v = parse.(Int, split(l, '\t'))
        rows[a][nk0 + k] = v
    end
    secs = [parse(Float64, split(l, '\t')[3]) for l in eachline(joinpath(SYN, "seconds.tsv"))]
    return ay, names, rows, secs
end

family(k) = String(split(k, ':')[1])

"Every run of consecutive ayahs in one surah lasting MIN_S..MAX_S seconds: (first, last, seconds, key counts)."
function candidates(ay, rows, secs, MIN_S, MAX_S)
    out = Tuple{Int, Int, Float64, Dict{Int, Int}}[]
    n = length(ay)
    for i in 1:n
        c = Dict{Int, Int}()
        s = 0.0
        for j in i:n
            ay[j][1] == ay[i][1] || break
            s += secs[j]
            s > MAX_S && break
            mergewith!(+, c, rows[j])
            s >= MIN_S && push!(out, (i, j, s, copy(c)))
        end
    end
    return out
end

function greedy(cands, names, total, budget, needs, n_ayahs; used = falses(n_ayahs), got0 = nothing,
                force_special = true)
    fam = family.(names)
    need = [min(get(needs, fam[k], 0), total[k]) for k in eachindex(names)]
    w = [need[k] > 0 ? get(WEIGHT, fam[k], 0.0) / need[k] : 0.0 for k in eachindex(names)]
    got = got0 === nothing ? zeros(Int, length(names)) : copy(got0)
    taken = copy(used)
    chosen = Int[]
    spent = 0.0
    gain(c) = sum((w[k] * min(v, max(0, need[k] - got[k])) for (k, v) in c[4]); init = 0.0)
    free(c) = !any(@view taken[c[1]:c[2]])
    function take!(p)
        c = cands[p]
        push!(chosen, p)
        taken[c[1]:c[2]] .= true
        spent += c[3]
        for (k, v) in c[4]
            got[k] += v
        end
    end
    if force_special                   # every rare case first, in its most efficient passage
        for k in findall(==("special"), fam)
            (total[k] == 0 || got[k] > 0) && continue
            best, br = 0, -Inf
            for (p, c) in enumerate(cands)
                (haskey(c[4], k) && free(c) && spent + c[3] <= budget) || continue
                r = gain(c) / c[3]
                r > br && ((best, br) = (p, r))
            end
            best > 0 && take!(best)
        end
    end
    while true
        best, br = 0, 0.0
        for (p, c) in enumerate(cands)
            (free(c) && spent + c[3] <= budget) || continue
            r = gain(c) / c[3]
            r > br + 1e-12 && ((best, br) = (p, r))
        end
        if best == 0
            # all needs met (or nothing affordable adds coverage): double the needs and go on
            all(got .>= need) || break
            any(need .< total) || break
            need .= min.(2 .* max.(need, 1), total)
            w .= [need[k] > 0 ? get(WEIGHT, fam[k], 0.0) / need[k] : 0.0 for k in eachindex(names)]
            continue
        end
        take!(best)
    end
    return chosen, got, taken, spent
end

function report(name, cands, chosen, got, names, total, ay)
    fam = family.(names)
    cov = Dict{String, Any}()
    for f in unique(fam)
        ks = findall(==(f), fam)
        present = [k for k in ks if total[k] > 0]
        cov[f] = Dict("keys_in_quran" => length(present), "covered" => count(k -> got[k] > 0, present),
                      "min_instances" => minimum(got[present]; init = 0))
    end
    passages = [Dict("surah" => ay[c[1]][1], "from_ayah" => ay[c[1]][2], "to_ayah" => ay[c[2]][2],
                     "ayahs" => c[2] - c[1] + 1, "seconds" => round(c[3]; digits = 1))
                for c in (cands[p] for p in chosen)]
    sort!(passages; by = p -> (p["surah"], p["from_ayah"]))
    missing = [names[k] for k in eachindex(names) if total[k] > 0 && got[k] == 0]
    println(rpad(name, 6), length(passages), " passages, ", sum(p["ayahs"] for p in passages), " ayahs, ",
            round(sum(p["seconds"] for p in passages) / 60; digits = 1), " min")
    for f in sort(collect(Base.keys(cov)))
        println("   ", rpad(f, 8), cov[f]["covered"], " / ", cov[f]["keys_in_quran"], " keys  (min instances ",
                cov[f]["min_instances"], ")")
    end
    return Dict("passages" => passages, "coverage" => cov, "uncovered" => missing)
end

function main()
    ay, names, rows, secs = load()
    total = zeros(Int, length(names))
    for r in rows, (k, v) in r
        total[k] += v
    end
    n = length(ay)
    cl = candidates(ay, rows, secs, LONG.min_s, LONG.max_s)
    cf = candidates(ay, rows, secs, FILL.min_s, FILL.max_s)
    ct = candidates(ay, rows, secs, TEST.min_s, TEST.max_s)
    println(length(cl), " long / ", length(cf), " fill candidates over ", n, " ayahs (", round(sum(secs) / 3600; digits = 1), " h)")
    l, gl, taken, _ = greedy(cl, names, total, LONG.budget, NEED, n)
    f, gf, taken, _ = greedy(cf, names, total, FILL.budget, NEED, n; used = taken, got0 = gl)
    te, gte, _, _ = greedy(ct, names, total, TEST.budget, Dict(k => 1 for k in Base.keys(NEED)), n;
                           used = taken, force_special = false)
    out = Dict("voice" => "Husary_128kbps", "phases" => Dict("long" => LONG, "fill" => FILL, "test" => TEST),
               "train_long" => report("long", cl, l, gl, names, total, ay),
               "train" => report("train", vcat(cl, cf), vcat(l, f .+ length(cl)), gf, names, total, ay),
               "test" => report("test", ct, te, gte, names, total, ay))
    open(joinpath(SYN, "pilot.json"), "w") do io
        JSON3.pretty(io, out)
    end
    println("-> ", joinpath(SYN, "pilot.json"))
end

main()
