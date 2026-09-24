# Sukoon timing — the temporal signature of a master, and a properly grounded count unit.
#
# Tajweed does not give every sakin letter the same duration. A letter's *sifah* decides how long the
# articulation can be held:
#
#   rikhw    (رخاوة)  the breath keeps flowing (س ش ف ث ص ز ...)      → held LONGEST at sukoon
#   between  (بينية)  ل ن م ر ع — "لن عمر"                            → INTERMEDIATE
#   shadeed  (شدة)    the voice is fully blocked (ب ت ك ق د ج ط ء)    → SHORTEST, a stop is a stop
#
# A master realises that ordering audibly and consistently. Fast reciters compress it — they do not
# have time to let a rikhw letter flow — so the *separation* between the three classes is itself a
# mastery statistic, not merely a phonetic curiosity.
#
# It also fixes a real problem in tasawi_run.jl: the absolute count scale there did not match the
# 2:4:6 notation (measured 2.25 : 6.20 : 12.0). A count should not be guessed globally. Every reciter
# has their own tempo AND their own realisation of these classes, so this measures each speaker's own
# scale from their own correct instances — the vowelled letters give the count unit, and the three
# sukoon classes give three further reference scales — exactly how a teacher calibrates a student.
#
# Reported per reciter:
#   haraka_s              median duration of a vowelled letter = one count, in seconds (their tempo)
#   rikhw / between / shadeed   median sakin duration in that reciter's own counts
#   separation            rikhw − shadeed, in counts: how strongly the ordering is realised
#   ordered               whether rikhw > between > shadeed holds at all
#
# TIMING is `viterbi` (default, whole 40 ms frames) or `centroid` (sub-frame onsets from
# `centroid_onsets`). At 40 ms a sakin letter is one or two frames, so under Viterbi rikhw and
# bayniyya tie on quantised values; the centroid is what makes the three-way ordering measurable.
#
#   julia --project=research_agency_lab/substrate_library/julia \
#         research_agency_lab/substrate_library/julia/sukoon_timing.jl DUMP_DIR OUT.json [TIMING]

using QaariLab, JSON3, Statistics

const ANCHORS = Set(["everyayah:Husary_Muallim_128kbps", "everyayah:Husary_128kbps",
                     "everyayah:Husary_128kbps_Mujawwad", "qdc:12", "qdc:6"])
const SHORT_V = Set(['َ', 'ُ', 'ِ'])
const MADD = Set(['ا', 'ۥ', 'ۦ'])
const SPECIAL = Set(['ں', '۾', 'ڇ', 'ۜ', 'ٲ', '۪', 'ـ', 'ؙ'])
const CLASSES = ["rikhw", "between", "shadeed"]
speaker_of(id) = String(split(String(id), '/')[1])

is_consonant(c) = !(c in SHORT_V) && !(c in MADD) && !(c in SPECIAL)

function main(dir, out, timing = "viterbi")
    timing in ("viterbi", "centroid") || error("timing must be viterbi or centroid, not $timing")
    C, c0, W, vocab, blank = load_dump_layout(dir)
    refs = load_sifat_ref(dir)
    isempty(refs) && error("no sifat.jsonl in $dir — run experiments/learner_eval/sifat_ref.py first")
    frame_s = 0.04
    # speaker -> class -> durations in counts ; speaker -> haraka durations in seconds
    dur = Dict{String, Dict{String, Vector{Float64}}}()
    hara = Dict{String, Vector{Float64}}()
    n = 0
    t0 = time()
    # the sifat head's class id -> name, read from the dump's own layout
    lvl = only(l for l in JSON3.read(read(joinpath(dir, "layout.json"), String)).levels
               if l.level == "shidda_or_rakhawa")
    idname = Dict(i - 1 => String(v) for (i, v) in enumerate(lvl.vocab))   # 0 = [PAD]
    en = Dict("[شديد]" => "shadeed", "[رخو]" => "rikhw", "[بين الشدة والرخاوة]" => "between")

    for line in eachline(joinpath(dir, "index.jsonl"))
        rec = JSON3.read(line)
        haskey(rec, :error) && continue
        id = String(rec.id)
        haskey(refs, id) || continue
        ph = String(rec.ref_ph)
        all(c -> haskey(vocab, c), ph) || continue
        spk = speaker_of(id)
        lp = read_dump_clip(dir, rec, C)[:, c0+1:c0+W]
        seq = [vocab[c] for c in ph]
        _, fi, l = ctc_viterbi(lp, seq, blank)
        f = timing == "centroid" ?
            [isfinite(c) ? c : Float64(v) for (c, v) in zip(centroid_onsets(lp, seq, blank), fi)] :
            Float64.(fi)
        units = ph_units(ph)
        nu = length(units)
        onset(i) = f[units[i][2]]
        dur_i(i) = (i < nu ? onset(i + 1) - onset(i) : l[units[i][3]] + 1 - onset(i)) * frame_s

        # one count = a vowelled letter: consonant + its short vowel, onset to the next onset
        hs = Float64[]
        for i in 1:(nu - 1)
            is_consonant(units[i][1]) && units[i + 1][1] in SHORT_V || continue
            push!(hs, (onset(min(i + 2, nu)) - onset(i)) * frame_s)
        end
        length(hs) >= 5 || continue
        h = median(hs)
        h > 0 || continue
        append!(get!(hara, spk, Float64[]), hs)

        ids = get(refs[id], "shidda_or_rakhawa", Int[])
        for i in 1:nu
            (sym, a, b) = units[i]
            is_consonant(sym) || continue
            # sakin = not followed by a short vowel (the phoneme string writes every vowel explicitly)
            i < nu && units[i + 1][1] in SHORT_V && continue
            a <= length(ids) || continue
            cls = get(en, get(idname, ids[a], ""), "")
            cls == "" && continue
            push!(get!(get!(dur, spk, Dict{String, Vector{Float64}}()), cls, Float64[]), dur_i(i) / h)
        end
        n += 1
        n % 2000 == 0 && println(stderr, "$n clips, $(round(time() - t0; digits = 1)) s")
    end

    med(v) = isempty(v) ? NaN : median(v)
    rows = []
    for (spk, d) in dur
        all(haskey(d, c) && length(d[c]) >= 8 for c in CLASSES) || continue
        r, bt, sh = med(d["rikhw"]), med(d["between"]), med(d["shadeed"])
        push!(rows, (spk = spk, anchor = spk in ANCHORS, haraka_s = med(get(hara, spk, Float64[])),
                     rikhw = r, between = bt, shadeed = sh, separation = r - sh,
                     ordered = (r > bt > sh), n = sum(length(d[c]) for c in CLASSES)))
    end
    sort!(rows, by = x -> -x.separation)

    println(rpad("reciter", 46), rpad("haraka_s", 9), rpad("rikhw", 7), rpad("betwn", 7),
            rpad("shded", 7), rpad("sep", 7), "ord")
    for x in rows
        println(rpad((x.anchor ? "* " : "  ") * split(x.spk, ":")[end], 46),
                rpad(round(x.haraka_s; digits = 3), 9), rpad(round(x.rikhw; digits = 3), 7),
                rpad(round(x.between; digits = 3), 7), rpad(round(x.shadeed; digits = 3), 7),
                rpad(round(x.separation; digits = 3), 7), x.ordered ? "yes" : "NO")
    end
    anc = [x.separation for x in rows if x.anchor]
    oth = [x.separation for x in rows if !x.anchor]
    println("\nanchor separation median ", round(median(anc); digits = 3),
            " vs others ", round(median(oth); digits = 3),
            " | ordering holds: anchors ", count(x -> x.ordered && x.anchor, rows), "/", length(anc),
            ", others ", count(x -> x.ordered && !x.anchor, rows), "/", length(oth))
    open(io -> JSON3.pretty(io, Dict("clips" => n, "timing" => timing, "reciters" => [Dict(pairs(x)) for x in rows],
                                     "anchor_separation" => median(anc),
                                     "other_separation" => median(oth))), out, "w")
    println("$n clips in $(round(time() - t0; digits = 1)) s -> $out")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS...)
end
