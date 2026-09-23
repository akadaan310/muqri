# Ranking reciters by DISTANCE FROM THE ANCHOR, not by flag rate.
#
# The flag-rate score in datastore/reciter_table.py cannot put the anchors on top, and the reason is
# structural rather than a tuning problem: every threshold is conformal at level α *calibrated on the
# anchors*, so an anchor is flagged at exactly α by construction. Any reciter who happens to be more
# conservative on a metric lands below α and outranks them. Flag rate is a one-sided count against a
# cut; "how close is this recitation to the ijazah reference" is a distance between distributions.
#
# So: describe each reciter by a feature vector of the quantities we have actually validated, whiten
# the 41 × d matrix with the robust OGK scatter (Frontier A1's multivariate 1.4826·MAD, so ms, counts
# and rates become commensurate and correlated features are not double-counted), and take the
# Mahalanobis distance to the anchor centroid. Lower is closer to the reference.
#
# This is the A1/A2 formulation from frontier_math_roadmap.md applied at the reciter level, and unlike
# the flag-rate sum it is symmetric: being *unlike* the anchor in either direction costs.
#
#   julia --project=research_agency_lab/substrate_library/julia \
#         research_agency_lab/substrate_library/julia/reciter_distance.jl KEYS_DIR TIER OUT.json

using QaariLab, JSON3, Statistics, LinearAlgebra

const ANCHORS = Set(["everyayah:Husary_Muallim_128kbps", "everyayah:Husary_128kbps",
                     "everyayah:Husary_128kbps_Mujawwad", "qdc:12", "qdc:6"])

read_json(p) = isfile(p) ? JSON3.read(read(p, String)) : nothing

function features(dir, tier)
    tw = read_json(joinpath(dir, "tasawi_$tier.json"))
    sk = read_json(joinpath(dir, "sukoon_$tier.json"))
    sf = read_json(joinpath(dir, "sifat_$tier.json"))
    rs = read_json(joinpath(dir, "ruleswap_$tier.json"))
    feats = Dict{String, Dict{String, Float64}}()
    add!(spk, k, v) = (isfinite(v) && (get!(feats, String(spk), Dict{String, Float64}())[k] = v))

    if tw !== nothing
        for (cls, cv) in pairs(tw.classes), (spk, v) in pairs(cv.per_speaker)
            add!(spk, "len_$cls", Float64(v.median_counts))     # is the length right
            add!(spk, "cv_$cls", Float64(v.cv))                 # is it equal (tasawi)
        end
    end
    if sk !== nothing
        for r in sk.reciters
            add!(r.spk, "sep", Float64(r.separation))           # sukoon class separation
            add!(r.spk, "rikhw", Float64(r.rikhw))
            add!(r.spk, "shadeed", Float64(r.shadeed))
        end
    end
    if sf !== nothing
        for (lvl, classes) in pairs(sf.levels), (cls, cv) in pairs(classes), (spk, v) in pairs(cv.per_speaker)
            add!(spk, "sif_$(lvl)_$(cls)", Float64(v.flag_rate))
        end
    end
    if rs !== nothing
        for (fam, fv) in pairs(rs.families), (spk, v) in pairs(fv.per_speaker)
            add!(spk, "rule_$fam", Float64(v.flag_rate))
        end
    end
    return feats
end

function main(dir, tier, out)
    feats = features(dir, tier)
    isempty(feats) && error("no feature files for tier $tier in $dir")
    # keep only features every reciter has, so no one is scored on a different basis
    spks = sort(collect(keys(feats)))
    keys_all = sort(collect(intersect((Set(keys(feats[s])) for s in spks)...)))
    length(keys_all) >= 3 || error("only $(length(keys_all)) shared features")
    X = [feats[s][k] for s in spks, k in keys_all]
    anc = [i for (i, s) in enumerate(spks) if s in ANCHORS]
    isempty(anc) && error("no anchors present")

    μ, Σ = ogk(X)                       # robust location/scatter over all reciters
    Σ += 1e-9I
    A = inv(cholesky(Symmetric(Σ)).L)   # whitening
    Z = (X .- μ') * A'
    centre = vec(mean(Z[anc, :], dims = 1))          # the anchors' centroid, in whitened space
    d = [norm(Z[i, :] - centre) for i in 1:length(spks)]

    ord = sortperm(d)
    println("features: ", length(keys_all), " shared, ", length(spks), " reciters, ",
            length(anc), " anchors")
    println(rpad("#", 4), rpad("distance", 10), rpad("ladder", 9), "reciter")
    rows = []
    for (rank, i) in enumerate(ord)
        isanc = spks[i] in ANCHORS
        println(rpad(rank, 4), rpad(round(d[i]; digits = 3), 10),
                rpad(isanc ? "ANCHOR" : "", 9), split(spks[i], ":")[end])
        push!(rows, Dict("rank" => rank, "reciter_id" => spks[i], "distance" => d[i], "anchor" => isanc))
    end
    anc_ranks = [r["rank"] for r in rows if r["anchor"]]
    println("\nanchor ranks: ", anc_ranks, " of ", length(spks))
    open(io -> JSON3.pretty(io, Dict("tier" => tier, "features" => keys_all,
                                     "reciters" => rows, "anchor_ranks" => anc_ranks)), out, "w")
    println("-> $out")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS[1], ARGS[2], ARGS[3])
end
