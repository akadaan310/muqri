# What every reciter always does, what all of them share, what only the masters share, and which
# departures travel together -- as sets, information and a graph.
#
# A "feature" is one characteristic of one letter in one context: (letter, context, head), e.g.
# (ط, sakin, qalqalah). For each reciter its value is the REALISATION RATE: how often the head heard
# the class the books expect there (calculus/extract.py, 545,592 consonants, 41 reciters).
#
#   INVARIANTS  I_r = {f : rate_r(f) >= 0.98}   what reciter r always does
#               W_r = {f : rate_r(f) <= 0.85}   where r recurrently departs
#   CORES       universal  = ∩_r I_r            what every reciter shares
#               masters    = ∩_{r ∈ M} I_r      what every master shares
#               mastery    = masters \ {f : f ∈ W_r for >= 3 non-masters}
#                            -- what the masters hold and others do not
#   INFORMATION for each feature, I(tier; weak) in bits: how much knowing whether a reciter departs on
#               f tells you about their tier -- the features that carry mastery
#   COMPANIONS  departures that come TOGETHER: over the 11,996 verses, for every pair of departure
#               features the pointwise mutual information of co-occurring in the same verse
#               (log2 P(a,b) / P(a)P(b)), with support -- "if a reciter slips here, they slip there"
#   NEIGHBOURS  reciters as vectors of rates; cosine similarity; each reciter's k nearest; label-
#               propagation communities; per community the departures ALL its members share
#
#   julia --project=research_agency_lab/substrate_library/julia \
#         research_agency_lab/substrate_library/julia/reciter_sets.jl DATA_DIR OUT.json

using JSON3, Statistics, LinearAlgebra

include(joinpath(@__DIR__, "calculus.jl"))        # load, head_cols, context

const MASTER_TIERS = ("anchor", "studio")

function feature_rates(D; min_n = 8)
    cols = head_cols(D.h)
    S = length(D.h.speakers)
    hit = Dict{Tuple{Int, String, Int}, Vector{Int}}()
    soft = Dict{Tuple{Int, String, Int}, Vector{Float64}}()   # summed P(expected class): a graded realisation
    tot = Dict{Tuple{Int, String, Int}, Vector{Int}}()
    clip = Vector{Int32}(undef, size(D.meta, 1)); read!(joinpath(DATA, "clip.i32"), clip)
    departures = Dict{Int32, Set{Tuple{Int, String, Int}}}()     # verse -> the features it departed on
    for i in 1:size(D.meta, 1), k in eachindex(cols)
        e = D.expect[i, k]
        e < 0 && continue
        f = (Int(D.meta[i, 1]), context(D.meta, i), k)
        s = D.speaker[i] + 1
        get!(tot, f, zeros(Int, S))[s] += 1
        get!(soft, f, zeros(S))[s] += D.probs[i, cols[k][e + 1]]
        if argmax(@view D.probs[i, cols[k]]) - 1 == e
            get!(hit, f, zeros(Int, S))[s] += 1
        else
            push!(get!(departures, clip[i], Set{Tuple{Int, String, Int}}()), f)
        end
    end
    feats = sort([f for f in keys(tot) if count(>=(min_n), tot[f]) >= 20])   # measured on >= 20 reciters
    R = fill(NaN, S, length(feats)); Q = fill(NaN, S, length(feats))
    for (j, f) in enumerate(feats), s in 1:S
        n = tot[f][s]
        n >= min_n || continue
        R[s, j] = get(hit, f, zeros(Int, S))[s] / n
        Q[s, j] = soft[f][s] / n
    end
    return feats, R, Q, departures
end

fname(D, (l, c, k)) = string(D.letters[l + 1], " ", c, " ", D.h.heads[k])

H2(p) = p <= 0 || p >= 1 ? 0.0 : -(p * log2(p) + (1 - p) * log2(1 - p))

"""I(tier; weak) for one feature, over the reciters it was measured on (bits)."""
function mi_tier(weak::AbstractVector{Bool}, tier::AbstractVector{String})
    n = length(weak); n == 0 && return 0.0
    pw = mean(weak)
    h = H2(pw)
    cond = 0.0
    for t in unique(tier)
        m = tier .== t
        cond += mean(m) * H2(mean(weak[m]))
    end
    return h - cond
end

"""Deterministic label propagation on a weighted kNN graph: each node takes the label with the most
neighbour weight, ties to the smallest label, until nothing changes."""
function communities(W::Matrix{Float64})
    n = size(W, 1); lab = collect(1:n)
    for _ in 1:100
        changed = false
        for i in 1:n
            score = Dict{Int, Float64}()
            for j in 1:n
                W[i, j] > 0 && (score[lab[j]] = get(score, lab[j], 0.0) + W[i, j])
            end
            isempty(score) && continue
            best = maximum(values(score))
            l = minimum(k for (k, v) in score if v == best)
            l != lab[i] && (lab[i] = l; changed = true)
        end
        changed || break
    end
    return lab
end

function main(dir, out)
    global DATA = dir
    D = load(dir)
    feats, R, Q, departures = feature_rates(D)
    S = size(R, 1)
    names = [fname(D, f) for f in feats]
    tiers = String.(collect(D.h.tiers))
    spk = String.(collect(D.h.speakers))
    master = [t in MASTER_TIERS for t in tiers]
    println("$(length(feats)) features (letter x context x characteristic) measured on >= 20 of $S reciters")

    inv = [Set(j for j in axes(R, 2) if !isnan(R[s, j]) && R[s, j] >= 0.98) for s in 1:S]
    weak = [Set(j for j in axes(R, 2) if !isnan(R[s, j]) && R[s, j] <= 0.85) for s in 1:S]
    universal = reduce(intersect, inv)
    mcore = reduce(intersect, inv[master])
    weak_count = [count(s -> !master[s] && j in weak[s], 1:S) for j in axes(R, 2)]
    mastery = sort([j for j in mcore if weak_count[j] >= 3], by = j -> -weak_count[j])
    println("universal core (every reciter always): $(length(universal)) features; masters' core: $(length(mcore)); ",
            "held by every master but a recurrent departure for >= 3 others: $(length(mastery))")
    for j in mastery[1:min(15, end)]
        println("   ", rpad(names[j], 44), " departs for ", weak_count[j], " non-masters")
    end

    # expectations the masters overrule: where the masters' median realisation is under one half, the
    # books-as-encoded expect something the masters do not do -- the expectation, not the recitation,
    # is what needs revising (e.g. takrir, which must be concealed)
    mrate = [median(filter(!isnan, R[master, j])) for j in axes(R, 2)]
    overruled = sort([j for j in axes(R, 2) if mrate[j] < 0.5], by = j -> mrate[j])
    println("\nEXPECTATIONS THE MASTERS OVERRULE (masters' median realisation < 0.5)")
    for j in overruled
        println("   ", rpad(names[j], 44), " masters realise ", round(mrate[j]; digits = 3))
    end

    # graded mastery: Cliff's delta of the soft realisation, masters vs everyone else, per feature
    function cliff(a, b)
        (isempty(a) || isempty(b)) && return 0.0
        return (sum(x > y for x in a, y in b) - sum(x < y for x in a, y in b)) / (length(a) * length(b))
    end
    delta = [(j = j, d = cliff(filter(!isnan, Q[master, j]), filter(!isnan, Q[.!master, j]))) for j in axes(Q, 2)
             if !(j in overruled)]
    sort!(delta, by = x -> -x.d)
    println("\nWHAT THE MASTERS DO MORE FULLY (Cliff's delta of graded realisation, masters vs the rest)")
    for x in delta[1:min(20, end)]
        println("   ", rpad(names[x.j], 44), " delta ", round(x.d; digits = 3), "   masters ",
                round(median(filter(!isnan, Q[master, x.j])); digits = 3), " vs ",
                round(median(filter(!isnan, Q[.!master, x.j])); digits = 3))
    end

    # information: which features tell tiers apart
    mi = Float64[]
    for j in axes(R, 2)
        ok = .!isnan.(R[:, j])
        push!(mi, mi_tier(R[ok, j] .<= 0.85, tiers[ok]))
    end
    top_mi = sortperm(mi, rev = true)[1:min(15, end)]
    println("\nFEATURES THAT CARRY TIER (I(tier; departs), bits)")
    for j in top_mi
        println("   ", rpad(names[j], 44), round(mi[j]; digits = 3), "   departs for ",
                count(s -> j in weak[s], 1:S), " reciters")
    end

    # companions: departures that co-occur in the same verse
    idx = Dict(f => j for (j, f) in enumerate(feats))
    V = length(departures)
    single = zeros(Int, length(feats)); pair = Dict{Tuple{Int, Int}, Int}()
    for fs in values(departures)
        js = sort([idx[f] for f in fs if haskey(idx, f)])
        for a in js
            single[a] += 1
        end
        for x in 1:length(js), y in x+1:length(js)
            pair[(js[x], js[y])] = get(pair, (js[x], js[y]), 0) + 1
        end
    end
    comp = [(a = names[a], b = names[b], together = c,
             pmi = log2((c / V) / ((single[a] / V) * (single[b] / V))))
            for ((a, b), c) in pair if c >= 25]
    sort!(comp, by = x -> -x.pmi)
    println("\nDEPARTURES THAT TRAVEL TOGETHER (same verse; PMI bits, support >= 25 verses)")
    for x in comp[1:min(15, end)]
        println("   ", rpad(x.a, 34), " + ", rpad(x.b, 34), " PMI ", round(x.pmi; digits = 2), "  n=", x.together)
    end

    # neighbours: cosine over the features every reciter was measured on
    common = [j for j in axes(Q, 2) if all(!isnan, Q[:, j]) && !(j in overruled)]
    X = Q[:, common] .- mean(Q[:, common], dims = 1)
    X ./= max.(1e-9, std(Q[:, common], dims = 1))            # each feature on its own scale
    Xn = X ./ max.(1e-9, sqrt.(sum(abs2, X, dims = 2)))
    C = Xn * Xn'
    k = 4
    W = zeros(S, S)
    for i in 1:S
        for j in sortperm(C[i, :], rev = true)[2:k+1]
            W[i, j] = W[j, i] = max(C[i, j], 0)
        end
    end
    lab = communities(W)
    groups = Dict{Int, Vector{Int}}()
    for (i, l) in enumerate(lab)
        push!(get!(groups, l, Int[]), i)
    end
    println("\nNEIGHBOURHOODS (kNN k=$k on $(length(common)) shared features; label propagation)")
    comms = []
    for (l, mem) in sort(collect(groups), by = kv -> -length(kv[2]))
        shared_weak = setdiff(reduce(intersect, weak[mem]), overruled)
        push!(comms, (members = spk[mem], tiers = tiers[mem], shared_departures = names[collect(shared_weak)]))
        println("   {", join([split(spk[m], "_")[1] * "(" * tiers[m][1:1] * ")" for m in mem], ", "), "}")
        isempty(shared_weak) || println("      all depart on: ", join(names[collect(shared_weak)][1:min(6, end)], "; "))
    end

    open(out, "w") do io
        JSON3.write(io, (features = names, reciters = spk, tiers = tiers,
                         rates = [[isnan(x) ? nothing : round(x; digits = 4) for x in R[s, :]] for s in 1:S],
                         invariants = [names[collect(v)] for v in inv], departures = [names[collect(v)] for v in weak],
                         universal_core = names[collect(universal)], masters_core = names[collect(mcore)],
                         overruled_expectations = [(feature = names[j], masters_realise = mrate[j]) for j in overruled],
                         mastery_graded = [(feature = names[x.j], cliffs_delta = x.d) for x in delta],
                         graded = [[isnan(x) ? nothing : round(x; digits = 4) for x in Q[s, :]] for s in 1:S],
                         mastery_features = [(feature = names[j], nonmasters_departing = weak_count[j]) for j in mastery],
                         tier_information = [(feature = names[j], bits = round(mi[j]; digits = 4)) for j in sortperm(mi, rev = true)],
                         companions = comp, similarity = C, knn = W, communities = comms))
    end
    println("\n-> $out")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS...)
end
