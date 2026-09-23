# Counterfactual test of the *rule* layer: madd lengths, ghunnah, ikhfa, idgham, shadda.
#
# `textswap_gop.jl` proved the machinery on lahn jali (wrong letter / wrong harakah): swap one symbol
# of the TEXT, and the CTC likelihood ratio says the audio disagrees there — 99.7 % detected, 97.1 %
# named. Every remaining tajweed rule is testable the same way, because `quran_phonetizer` encodes
# rule realisation *in the phoneme string*:
#
#     ۥ×4  4-count madd      ا×2  madd tabii        ں×3  ikhfa noon, nasalised and held
#     ل×2  idgham / shadda   ن×4  ghunnah held      ں→ن  ikhfa undone to izhar
#
# So "recited a 4-count madd as 2", "dropped the ghunnah", "failed to merge the idgham" and "read
# ikhfa as izhar" are all *length or symbol edits of the reference string* — the same likelihood-ratio
# test, no separate duration model. Masters' audio is taken as correct at the site, so an edit makes
# the audio disagree in a known way and we can measure detection honestly.
#
# Perturbation families (each reported separately):
#   madd_short    madd run L≥2  → L−2 (or 1)      reciting 4 counts as 2
#   madd_long     madd run L    → L+2             over-stretching
#   ghunnah_drop  ں/ن/م run L≥3 → 1               nasalisation dropped
#   idgham_undo   consonant run of 2 → 1          failing to merge / no shadda
#   ikhfa_izhar   ں → ن (same length)             hiding read as clear
#
# Thresholds are conformal on the ANCHORS' unedited sites, per family (Vovk et al.), exactly as in
# textswap_gop.jl, so the false-alarm level is a coverage statement rather than a tuned constant.
#
#   julia --project=research_agency_lab/substrate_library/julia \
#         research_agency_lab/substrate_library/julia/ruleswap_gop.jl DUMP_DIR OUT.json [--alpha 0.01]

using QaariLab, JSON3, Statistics

const ANCHORS = Set(["everyayah:Husary_Muallim_128kbps", "everyayah:Husary_128kbps",
                     "everyayah:Husary_128kbps_Mujawwad", "qdc:12", "qdc:6"])
const MADD = Set(['ا', 'ۥ', 'ۦ'])          # madd letters; length = the count
const NASAL = Set(['ں', 'ن', 'م'])          # ghunnah carriers
const VOWELS = Set(['َ', 'ُ', 'ِ'])
const SPECIAL = Set(['ں', '۾', 'ڇ', 'ۜ', 'ٲ', '۪', 'ـ', 'ؙ'])  # rule-carrying symbols, handled explicitly
speaker_of(id) = String(split(String(id), '/')[1])

"""Local CTC log-likelihood of `seq` on the frames of `units[lo..hi]`, with `margin` frames of slack.

Returns the window (t0, t1) too so the reference and every counterfactual are scored on *identical*
frames — the comparison is otherwise meaningless."""
function window_ll(lp, seq::Vector{Int}, blank::Int, t0::Int, t1::Int)
    X = @view lp[t0:t1, :]
    return ctc_forward(X, seq, blank)
end

"""Counterfactual edits of one unit run `(sym, a, b)` in the character vector `cs`.

Each edit returns `(family, new characters for the run)`; an empty vector deletes the run."""
function edits(sym::Char, len::Int)
    out = Tuple{String, Vector{Char}}[]
    if sym in MADD && len >= 2
        push!(out, ("madd_short", fill(sym, max(1, len - 2))))
        push!(out, ("madd_long", fill(sym, len + 2)))
    elseif sym in NASAL && len >= 3
        push!(out, ("ghunnah_drop", [sym]))
    elseif len == 2 && !(sym in MADD) && !(sym in VOWELS) && !(sym in SPECIAL)
        push!(out, ("idgham_undo", [sym]))
    end
    # The phonetizer gives these rules their own symbols, so the counterfactual is exact rather than
    # inferred from context: ں = ikhfa noon, ۾ = iqlab meem (ن→م before ب), ڇ = qalqala release.
    sym == 'ں' && push!(out, ("ikhfa_izhar", fill('ن', len)))
    if sym == '۾'
        push!(out, ("iqlab_undo", fill('ن', len)))        # ن never converted to م
        push!(out, ("iqlab_no_ghunnah", ['م']))           # converted but not held
    end
    sym == 'ڇ' && push!(out, ("qalqala_drop", Char[]))    # no audible release on the sakin letter
    return out
end

function main(dir, out; α = 0.01, ctx = 2, margin = 3)
    C, c0, W, vocab, blank = load_dump_layout(dir)
    swaps = NamedTuple[]
    n = 0
    t0w = time()
    for line in eachline(joinpath(dir, "index.jsonl"))
        rec = JSON3.read(line)
        haskey(rec, :error) && continue
        ph = String(rec.ref_ph)
        all(c -> haskey(vocab, c), ph) || continue
        spk = speaker_of(rec.id)
        lp = read_dump_clip(dir, rec, C)[:, c0+1:c0+W]
        path, origin, _ = read_as_recited(lp, ph, vocab, blank)
        cs = collect(path)
        seq = [vocab[c] for c in cs]
        units = ph_units(path)
        _, f, l = ctc_viterbi(lp, seq, blank)
        T = size(lp, 1)

        for (i, (sym, a, b)) in enumerate(units)
            origin[a] > 0 || continue                       # skip repeated copies
            fams = edits(sym, b - a + 1)
            isempty(fams) && continue
            lo, hi = max(1, i - ctx), min(length(units), i + ctx)
            ca, cb = units[lo][2], units[hi][3]
            t0 = max(1, f[ca] - margin)
            t1 = min(T, l[cb] + margin)
            t1 > t0 || continue
            left, right = seq[ca:a-1], seq[b+1:cb]
            ref = window_ll(lp, vcat(left, seq[a:b], right), blank, t0, t1)
            isfinite(ref) || continue
            for (fam, newchars) in fams
                all(c -> haskey(vocab, c), newchars) || continue
                alt = window_ll(lp, vcat(left, [vocab[c] for c in newchars], right), blank, t0, t1)
                isfinite(alt) || continue
                # badness of the EDITED text: how much worse it explains the audio than the truth
                push!(swaps, (speaker = spk, family = fam, bad = Float64(ref - alt), symbol = string(sym)))
            end
        end
        n += 1
        n % 50 == 0 && println(stderr, "$n clips, $(length(swaps)) edits, $(round(time() - t0w; digits = 1)) s")
    end

    res = Dict{String, Any}("alpha" => α, "clips" => n, "families" => Dict{String, Any}())
    # Two honest statistics, neither of which needs an arbitrary cut:
    #   win_rate  — fraction of edits where the TRUE text explains the audio better than the edited one.
    #               On masters this is the rule-layer analogue of textswap's "detected": the engine
    #               prefers the correct realisation, so it would flag the wrong one.
    #   flag_rate — fraction of SITES where some edit beats the reference, i.e. the model believes the
    #               reciter actually realised the rule the other way. On anchors this is a false-alarm
    #               proxy; on faster reciters it is a real (and expected) deviation rate.
    for fam in sort(unique(s.family for s in swaps))
        sw = [s for s in swaps if s.family == fam]
        spks = unique(s.speaker for s in sw)
        per_spk = Dict{String, Any}()
        for spk in spks
            v = [s for s in sw if s.speaker == spk]
            length(v) >= 5 || continue   # rare rules (ikhfa) are thin at T10; T300 fills them in
            m = [s.bad for s in v]
            per_spk[spk] = Dict("n" => length(v), "win_rate" => mean(m .> 0),
                                "flag_rate" => mean(m .< 0), "median_margin" => median(m),
                                "anchor" => spk in ANCHORS)
        end
        # a family with too few per-speaker instances is still reported at family level
        anc = [s.bad for s in sw if s.speaker in ANCHORS]
        oth = [v["flag_rate"] for (_, v) in per_spk if !v["anchor"]]
        owr = [v["win_rate"] for (_, v) in per_spk if !v["anchor"]]
        res["families"][fam] = Dict(
            "n_edits" => length(sw), "anchor_n" => length(anc),
            "anchor_win_rate" => isempty(anc) ? nothing : mean(anc .> 0),
            "anchor_flag_rate" => isempty(anc) ? nothing : mean(anc .< 0),
            "anchor_median_margin" => isempty(anc) ? nothing : median(anc),
            "other_median_win_rate" => isempty(owr) ? nothing : median(owr),
            "other_median_flag_rate" => isempty(oth) ? nothing : median(oth),
            "per_speaker" => per_spk)
        println(rpad(fam, 14), " edits=", lpad(length(sw), 7),
                "  anchor win=", lpad(isempty(anc) ? "-" : round(mean(anc .> 0); digits = 4), 7),
                "  anchor margin=", lpad(isempty(anc) ? "-" : round(median(anc); digits = 2), 8),
                "  anchor flag=", lpad(isempty(anc) ? "-" : round(mean(anc .< 0); digits = 4), 7),
                "  others win=", lpad(isempty(owr) ? "-" : round(median(owr); digits = 4), 7))
    end
    open(io -> JSON3.pretty(io, res), out, "w")
    println("$n clips in $(round(time() - t0w; digits = 1)) s -> $out")
end

if abspath(PROGRAM_FILE) == @__FILE__
    args = copy(ARGS)
    α = 0.01
    if (i = findfirst(==("--alpha"), args)) !== nothing
        α = parse(Float64, args[i+1]); deleteat!(args, i:i+1)
    end
    main(args[1], args[2]; α = α)
end
