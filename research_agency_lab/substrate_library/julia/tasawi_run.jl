# Tasāwī — the mastery question: are the held things held *equally*?
#
# Every test so far asks "is this instance right?". That is not what separates an ijazah-level reciter
# from a competent one. A master holds EVERY 4-count madd for the same 4 counts and EVERY ghunnah for
# the same 2, all the way through a sura. A weaker reciter may get each instance inside the permitted
# band while their lengths scatter — each instance passes, the recitation is still not masterful.
# Tasāwī (تساوي, "being equal") is that consistency requirement, and it is a *variance* statistic, so
# no per-instance threshold can express it.
#
# Measurement:
#   duration of a unit  = (last Viterbi frame − first + 1) × frame_s
#   haraka             = the clip's own median short-vowel (َ ُ ِ) duration = one count
#   counts             = duration / haraka       ← tempo-free, so a fast imam and a slow muallim are
#                                                  compared on the same scale (Frontier B1)
# For each reciter and each held class (madd run of 2/4/6, ghunnah run, ikhfa noon) we report
#   median counts  — is the length RIGHT (does a 4-count madd measure ~4)?
#   robust CV      — is it EQUAL (1.4826·MAD / |median|), the tasāwī score itself
# The CV threshold is conformal on the anchors, so "less consistent than a near-gold reciter is
# allowed to be" is a coverage statement, not a tuned constant.
#
#   julia --project=research_agency_lab/substrate_library/julia \
#         research_agency_lab/substrate_library/julia/tasawi_run.jl DUMP_DIR OUT.json [--alpha 0.01]

using QaariLab, JSON3, Statistics

const ANCHORS = Set(["everyayah:Husary_Muallim_128kbps", "everyayah:Husary_128kbps",
                     "everyayah:Husary_128kbps_Mujawwad", "qdc:12", "qdc:6"])
const SHORT_V = Set(['َ', 'ُ', 'ِ'])
const MADD = Set(['ا', 'ۥ', 'ۦ'])
const NASAL = Set(['ں', 'ن', 'م'])
speaker_of(id) = String(split(String(id), '/')[1])

"""Class of a unit for the tasāwī question, or `nothing` when its length carries no rule.

The class includes the *written* run length, because a 2-count madd and a 4-count madd are different
obligations and must not be pooled — pooling them is what made the legacy madd detectors meaningless."""
function held_class(sym::Char, len::Int)
    sym in MADD && len >= 2 && return "madd_$(len)"
    sym == 'ں' && return "ikhfa_noon"
    sym in NASAL && len >= 3 && return "ghunnah_$(sym)"
    return nothing
end

function main(dir, out; α = 0.01)
    C, c0, W, vocab, blank = load_dump_layout(dir)
    L = JSON3.read(read(joinpath(dir, "layout.json"), String))
    frame_s = 0.04
    # class -> speaker -> counts of every instance
    obs = Dict{String, Dict{String, Vector{Float64}}}()
    n = 0
    t0 = time()
    for line in eachline(joinpath(dir, "index.jsonl"))
        rec = JSON3.read(line)
        haskey(rec, :error) && continue
        ph = String(rec.ref_ph)
        all(c -> haskey(vocab, c), ph) || continue
        spk = speaker_of(rec.id)
        lp = read_dump_clip(dir, rec, C)[:, c0+1:c0+W]
        seq = [vocab[c] for c in ph]
        _, f, l = ctc_viterbi(lp, seq, blank)
        units = ph_units(ph)
        # CTC is peaky: Viterbi gives a symbol one frame and assigns the rest of its acoustic
        # duration to blank, so (l[b] − f[a]) is an alignment span, not a duration. The duration of a
        # unit is the interval from its own onset to the NEXT unit's onset, which absorbs the blanks
        # that belong to it. (With l−f the anchors read madd_2 = 3.0 and madd_4 = 7.0 counts with a
        # CV of exactly 0 — quantisation, not consistency.)
        nu = length(units)
        onset(i) = f[units[i][2]]
        dur_i(i) = (i < nu ? onset(i + 1) - onset(i) : l[units[i][3]] + 1 - onset(i)) * frame_s

        # one count = this clip's own median short vowel, so tempo cancels
        harakas = [dur_i(i) for i in 1:nu if units[i][1] in SHORT_V && units[i][3] == units[i][2]]
        length(harakas) >= 5 || continue
        h = median(harakas)
        h > 0 || continue

        for i in 1:nu
            (sym, a, b) = units[i]
            cls = held_class(sym, b - a + 1)
            cls === nothing && continue
            push!(get!(get!(obs, cls, Dict{String, Vector{Float64}}()), spk, Float64[]), dur_i(i) / h)
        end
        n += 1
        n % 500 == 0 && println(stderr, "$n clips, $(round(time() - t0; digits = 1)) s")
    end

    rcv(v) = length(v) < 4 ? NaN : (1.4826 * median(abs.(v .- median(v)))) / max(abs(median(v)), 1e-9)

    res = Dict{String, Any}("alpha" => α, "clips" => n, "classes" => Dict{String, Any}())
    println(rpad("class", 14), rpad("anchor med", 11), rpad("anchor CV", 10),
            rpad("τ(CV)", 9), rpad("others med CV", 14), "n_anchor")
    for cls in sort(collect(keys(obs)))
        d = obs[cls]
        anc = [rcv(v) for (s, v) in d if s in ANCHORS && length(v) >= 8]
        anc = filter(isfinite, anc)
        length(anc) >= 2 || continue
        # a HIGH cv is bad, so the conformal cut is the upper tail of the anchors' CVs
        τ, ok = conformal_threshold(anc, α)
        per_spk = Dict{String, Any}()
        for (s, v) in d
            length(v) >= 8 || continue
            per_spk[s] = Dict("n" => length(v), "median_counts" => median(v), "cv" => rcv(v),
                              "anchor" => s in ANCHORS, "exceeds_tau" => rcv(v) > τ)
        end
        others = [x["cv"] for (_, x) in per_spk if !x["anchor"] && isfinite(x["cv"])]
        anc_med = median(vcat([v for (s, v) in d if s in ANCHORS]...))
        res["classes"][cls] = Dict(
            "anchor_median_counts" => anc_med, "anchor_cv" => median(anc), "cv_threshold" => τ,
            "threshold_exact" => ok, "n_anchor_reciters" => length(anc),
            "other_median_cv" => isempty(others) ? nothing : median(others),
            "other_frac_exceeding" => isempty(others) ? nothing : mean(others .> τ),
            "per_speaker" => per_spk)
        println(rpad(cls, 14), rpad(round(anc_med; digits = 2), 11), rpad(round(median(anc); digits = 3), 10),
                rpad(round(τ; digits = 3), 9),
                rpad(isempty(others) ? "-" : round(median(others); digits = 3), 14), length(anc))
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
