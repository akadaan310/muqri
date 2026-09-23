# QaariKeys stage B: nested verse tiers T10 ⊂ T100 ⊂ T300 by weighted greedy coverage.
#
# Input: build/{ayahs.tsv, keys.txt, ayah_keys.tsv} from features.py.
# For a tier with per-family need n_f and weight w_f, key k (family f) needs
#     need_k = min(n_f, total_k)          (rare keys: every instance)
# and ayah a gains
#     g(a) = Σ_k  w_f / need_k · min(c_ak, max(0, need_k − got_k))
# per cost 1 + words_a / λ (short ayahs preferred so a whole reciter is cheap to run).
# F(S) = Σ_k w_f/need_k · min(need_k, got_k(S)) is monotone submodular, so greedy is within
# (1 − 1/e) of the best tier of the same size. When every need is met before the tier is full, all
# needs double (capped at the Quran total) and the greedy continues: repetitions of the same keys,
# which per-rule statistics (tasawi, per-letter GOP bands) need. Ties within a relative 1e-12 go to the lower ayah
# index, which keeps the selection identical to the Octave mirror (select_mirror.m).
#
#   julia --project=research_agency_lab/substrate_library/julia datasets/qaari_keys/select.jl [build_dir]

using JSON3

const FAMILIES = ["rule", "letter", "pair", "rep", "special", "waqf"]

# (name, max ayahs, need per family, weight per family, λ words, force all special-case ayahs)
const TIERS = [
    (name = "T10",  size = 10,  need = Dict("rule" => 1, "letter" => 1, "pair" => 1, "rep" => 1, "special" => 0, "waqf" => 1),
     weight = Dict("rule" => 1.0, "letter" => 0.5, "pair" => 1.5, "rep" => 1.0, "special" => 0.0, "waqf" => 0.5),
     lambda = 12.0, force_special = false),
    (name = "T100", size = 100, need = Dict("rule" => 3, "letter" => 3, "pair" => 2, "rep" => 3, "special" => 1, "waqf" => 3),
     weight = Dict("rule" => 1.0, "letter" => 0.5, "pair" => 1.5, "rep" => 1.0, "special" => 2.0, "waqf" => 0.5),
     lambda = 15.0, force_special = true),
    (name = "T300", size = 300, need = Dict("rule" => 8, "letter" => 8, "pair" => 5, "rep" => 8, "special" => 1, "waqf" => 8),
     weight = Dict("rule" => 1.0, "letter" => 0.5, "pair" => 1.5, "rep" => 1.0, "special" => 2.0, "waqf" => 0.5),
     lambda = 20.0, force_special = true),
]

function load(dir)
    ay = [parse.(Int, split(l, '\t')) for l in eachline(joinpath(dir, "ayahs.tsv"))]
    keys = readlines(joinpath(dir, "keys.txt"))
    C = zeros(Int, length(ay), length(keys))
    for l in eachline(joinpath(dir, "ayah_keys.tsv"))
        a, k, v = parse.(Int, split(l, '\t'))
        C[a, k] = v
    end
    return reduce(hcat, ay)', keys, C   # ayahs: [surah ayah words letters]
end

family(k) = split(k, ':')[1]

"""Greedy tier selection. Returns the chosen ayah indices (in pick order) and the gain of each pick."""
function select_tier(C, words, keys, tier, start::Vector{Int})
    fam = family.(keys)
    total = vec(sum(C; dims = 1))
    need = [min(get(tier.need, f, 0), t) for (f, t) in zip(fam, total)]
    base_need = copy(need)
    w = [need[k] > 0 ? get(tier.weight, fam[k], 0.0) / need[k] : 0.0 for k in eachindex(keys)]
    chosen = copy(start)
    got = length(chosen) > 0 ? vec(sum(C[chosen, :]; dims = 1)) : zeros(Int, length(keys))
    gains = fill(NaN, length(chosen))
    if tier.force_special
        for a in 1:size(C, 1)
            a in chosen && continue
            if any(C[a, k] > 0 for k in eachindex(keys) if fam[k] == "special")
                push!(chosen, a); push!(gains, NaN); got .+= C[a, :]
            end
        end
    end
    taken = falses(size(C, 1)); taken[chosen] .= true
    rounds = 0
    while length(chosen) < tier.size
        best, bestv = 0, 0.0
        for a in 1:size(C, 1)
            taken[a] && continue
            g = 0.0
            for k in eachindex(keys)
                c = C[a, k]
                (c == 0 || w[k] == 0) && continue
                g += w[k] * min(c, max(0, need[k] - got[k]))
            end
            g == 0 && continue
            v = g / (1 + words[a] / tier.lambda)
            if v > bestv * (1 + 1e-12)
                best, bestv = a, v
            end
        end
        if best == 0
            any(need .< total) || break  # nothing left anywhere
            need = [n > 0 ? min(2n, t) : 0 for (n, t) in zip(need, total)]
            w = [need[k] > 0 ? get(tier.weight, fam[k], 0.0) / need[k] : 0.0 for k in eachindex(keys)]
            rounds += 1
            continue
        end
        push!(chosen, best); push!(gains, bestv); taken[best] = true
        got .+= C[best, :]
    end
    return chosen, gains, got, base_need, rounds
end

function main(dir = joinpath(@__DIR__, "build"))
    ay, keys, C = load(dir)
    words = ay[:, 3]
    out = Dict{String, Any}("tiers" => Any[], "keys" => keys)
    prev = Int[]
    tsv = open(joinpath(dir, "tiers_julia.tsv"), "w")
    for t in TIERS
        chosen, gains, got, need, rounds = select_tier(C, words, keys, t, prev)
        missing = [keys[k] for k in eachindex(keys) if got[k] < need[k]]
        push!(out["tiers"], Dict(
            "name" => t.name, "n" => length(chosen), "words" => sum(words[chosen]),
            "verses" => [Dict("surah" => ay[a, 1], "ayah" => ay[a, 2], "words" => ay[a, 3], "gain" => isnan(g) ? nothing : g,
                              "new_in_tier" => !(a in prev), "keys" => [keys[k] for k in eachindex(keys) if C[a, k] > 0])
                         for (a, g) in zip(chosen, gains)],
            "coverage" => Dict(keys[k] => Dict("got" => got[k], "need" => need[k], "in_quran" => sum(C[:, k]))
                               for k in eachindex(keys)),
            "unmet" => missing, "need_doublings" => rounds))
        for a in chosen
            println(tsv, t.name, '\t', a)
        end
        met = count(k -> got[k] >= need[k], eachindex(keys))
        println(rpad(t.name, 5), " ayahs=", length(chosen), " words=", sum(words[chosen]),
                " keys met ", met, "/", length(keys), " need doublings ", rounds, isempty(missing) ? "" : "  unmet: " * join(first(missing, 8), ", "))
        prev = chosen
    end
    close(tsv)
    open(joinpath(dir, "tiers.json"), "w") do io
        JSON3.pretty(io, out)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS...)
end
