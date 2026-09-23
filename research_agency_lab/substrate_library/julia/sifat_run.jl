# Does the model's own sifat head agree with the textbook, and can it name the violated attribute?
#
# For every unit of every clip we take the expected sifah from the phonetizer and the model's
# per-frame posterior over that unit's Viterbi frames, and form the log-likelihood ratio of expected
# vs best competitor (Frontier/SifatGop). Thresholds are conformal on the ANCHORS only, per (level,
# expected class): a near-gold reciter may be flagged at most α of the time, by construction.
#
# Reported per level: the anchor threshold, the anchor flag rate (should sit at α), and each
# speaker's flag rate — which for non-anchors is a real quantity, not a false-alarm rate, because
# faster reciters genuinely relax attributes.
#
#   julia --project=research_agency_lab/substrate_library/julia \
#         research_agency_lab/substrate_library/julia/sifat_run.jl DUMP_DIR OUT.json [--alpha 0.01]

using QaariLab, JSON3, Statistics

const ANCHORS = Set(["everyayah:Husary_Muallim_128kbps", "everyayah:Husary_128kbps",
                     "everyayah:Husary_128kbps_Mujawwad", "qdc:12", "qdc:6"])
speaker_of(id) = String(split(String(id), '/')[1])

function main(dir, out; α = 0.01)
    C, c0, W, vocab, blank = load_dump_layout(dir)
    blocks = sifat_levels(dir)
    refs = load_sifat_ref(dir)
    isempty(refs) && error("no sifat.jsonl in $dir — run experiments/learner_eval/sifat_ref.py first")

    # (level, expected class) -> speaker -> LLRs
    obs = Dict{Tuple{String, String}, Dict{String, Vector{Float64}}}()
    n = 0
    t0 = time()
    for line in eachline(joinpath(dir, "index.jsonl"))
        rec = JSON3.read(line)
        haskey(rec, :error) && continue
        id = String(rec.id)
        haskey(refs, id) || continue
        ph = String(rec.ref_ph)
        all(c -> haskey(vocab, c), ph) || continue
        spk = speaker_of(id)
        full = read_dump_clip(dir, rec, C)
        lp = full[:, c0+1:c0+W]
        g = gop_sf(lp, ph, vocab, blank)
        units = ph_units(ph)
        frames = [u.frames for u in g]
        for r in sifat_llr(full, blocks, units, frames, refs[id])
            d = get!(obs, (r.level, r.expected), Dict{String, Vector{Float64}}())
            push!(get!(d, spk, Float64[]), r.llr)
        end
        n += 1
        n % 50 == 0 && println(stderr, "$n clips, $(round(time() - t0; digits = 1)) s")
    end

    res = Dict{String, Any}("alpha" => α, "clips" => n, "levels" => Dict{String, Any}())
    for lvl in SIFAT_LEVELS
        keys_l = sort([k for k in keys(obs) if k[1] == lvl], by = last)
        isempty(keys_l) && continue
        per_class = Dict{String, Any}()
        for key in keys_l
            d = obs[key]
            null = vcat([v for (s, v) in d if s in ANCHORS]...)
            length(null) < 20 && continue
            # a violation is a LOW llr, so the conformal cut is on the negated score
            τneg, ok = conformal_threshold(-null, α)
            τ = -τneg
            per_spk = Dict(s => Dict("n" => length(v), "flag_rate" => mean(v .< τ),
                                     "anchor" => s in ANCHORS, "median_llr" => median(v))
                           for (s, v) in d if length(v) >= 10)
            anchor_rate = mean(vcat([v for (s, v) in d if s in ANCHORS]...) .< τ)
            others = [x["flag_rate"] for (s, x) in per_spk if !x["anchor"]]
            per_class[key[2]] = Dict("threshold" => τ, "threshold_exact" => ok, "anchor_n" => length(null),
                                     "anchor_flag_rate" => anchor_rate,
                                     # JSON has no NaN: an absent statistic is null
                                     "other_median_flag_rate" => isempty(others) ? nothing : median(others),
                                     "other_max_flag_rate" => isempty(others) ? nothing : maximum(others),
                                     "per_speaker" => per_spk)
        end
        isempty(per_class) && continue
        res["levels"][lvl] = per_class
        for (cls, v) in sort(collect(per_class), by = first)
            println(rpad(lvl, 20), rpad(cls, 18), " n=", lpad(v["anchor_n"], 6),
                    "  τ=", lpad(round(v["threshold"]; digits = 3), 8),
                    "  anchor=", round(v["anchor_flag_rate"]; digits = 4),
                    "  others med=", v["other_median_flag_rate"] === nothing ? "-" : round(v["other_median_flag_rate"]; digits = 4),
                    " max=", v["other_max_flag_rate"] === nothing ? "-" : round(v["other_max_flag_rate"]; digits = 4))
        end
    end
    open(io -> JSON3.pretty(io, res), out, "w")
    println("$n clips in $(round(time() - t0; digits = 1)) s -> $out")
end

if abspath(PROGRAM_FILE) == @__FILE__
    args = copy(ARGS)
    α = 0.01
    if (i = findfirst(==("--alpha"), args)) !== nothing
        α = parse(Float64, args[i+1]); deleteat!(args, i:i+1)
    end
    main(args[1], args[2]; α = α)
end
