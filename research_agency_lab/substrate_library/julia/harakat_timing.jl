# Itmām al-Ḥarakāt — vowel perfection: isochrony, Ikhtilās and Ishbā'.
#
# Treatise I §2.2 and Treatise III: the single short vowel is the universal unit (1U), and three laws
# follow that no part of the engine currently checks.
#
#   1. Isochrony across vowel quality   Zamān(Fatḥah) = Zamān(Kasrah) = Zamān(Ḍammah)
#   2. Independence from consonant weight  the vowel on a heavy letter (طَ) must last exactly as long
#      as the vowel on a light one (تَ) — the extra pharyngealisation effort belongs to the CONSONANT
#      contact phase, never to the vowel
#   3. Deviation in either direction is a named error:
#        Ikhtilās (اختلاس)  duration < 1U (the treatise puts it near ⅔U) — vowel stolen, laḥn khafī
#        Ishbā'  (إشباع)   duration > 1U toward 1.5–2U — the vowel becomes an unwritten madd
#
# All three are measurable from the onset-to-onset durations we already compute, normalised by each
# reciter's own median vowel so tempo cancels (a Ḥadr reciter and a Taḥqīq reciter are judged on the
# same scale — Treatise I §1.1).
#
#   julia --project=research_agency_lab/substrate_library/julia \
#         research_agency_lab/substrate_library/julia/harakat_timing.jl DUMP_DIR OUT.json

using QaariLab, JSON3, Statistics

const ANCHORS = Set(["everyayah:Husary_Muallim_128kbps", "everyayah:Husary_128kbps",
                     "everyayah:Husary_128kbps_Mujawwad", "qdc:12", "qdc:6"])
const VOWELS = Dict('َ' => "fatha", 'ُ' => "damma", 'ِ' => "kasra")
const MADD = Set(['ا', 'ۥ', 'ۦ'])
const SPECIAL = Set(['ں', '۾', 'ڇ', 'ۜ', 'ٲ', '۪', 'ـ', 'ؙ'])
# the treatise's own thresholds, in units of the reciter's own median vowel
const IKHTILAS = 0.67
const ISHBA = 1.5
speaker_of(id) = String(split(String(id), '/')[1])

function main(dir, out)
    C, c0, W, vocab, blank = load_dump_layout(dir)
    refs = load_sifat_ref(dir)
    lay = JSON3.read(read(joinpath(dir, "layout.json"), String))
    tlvl = only(l for l in lay.levels if l.level == "tafkheem_or_taqeeq")
    idname = Dict(i - 1 => String(v) for (i, v) in enumerate(tlvl.vocab))
    HEAVY, LIGHT = "[مفخم]", "[مرقق]"

    # speaker -> vowel -> durations (in that speaker's own units, filled after normalisation)
    raw = Dict{String, Dict{String, Vector{Float64}}}()
    weight = Dict{String, Dict{String, Vector{Float64}}}()   # speaker -> heavy/light -> durations
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
        nu = length(units)
        onset(i) = f[units[i][2]]
        dur_i(i) = (i < nu ? onset(i + 1) - onset(i) : l[units[i][3]] + 1 - onset(i)) * 0.04

        # a vowel unit that is NOT followed by a madd letter (that would be an elongation, not a 1U vowel)
        vi = [i for i in 1:nu if haskey(VOWELS, units[i][1]) &&
              !(i < nu && units[i + 1][1] in MADD)]
        length(vi) >= 5 || continue
        ds = [dur_i(i) for i in vi]
        h = median(ds)
        h > 0 || continue

        d = get!(raw, spk, Dict{String, Vector{Float64}}())
        for (k, i) in enumerate(vi)
            push!(get!(d, VOWELS[units[i][1]], Float64[]), ds[k] / h)
        end
        # the vowel's duration must not depend on the preceding consonant's weight
        ids = get(get(refs, String(rec.id), Dict{String, Vector{Int}}()), "tafkheem_or_taqeeq", Int[])
        if !isempty(ids)
            wd = get!(weight, spk, Dict{String, Vector{Float64}}())
            for (k, i) in enumerate(vi)
                i > 1 || continue
                ca = units[i - 1][2]
                ca <= length(ids) || continue
                cls = get(idname, ids[ca], "")
                cls == HEAVY && push!(get!(wd, "heavy", Float64[]), ds[k] / h)
                cls == LIGHT && push!(get!(wd, "light", Float64[]), ds[k] / h)
            end
        end
        n += 1
        n % 2000 == 0 && println(stderr, "$n clips, $(round(time() - t0; digits = 1)) s")
    end

    rows = []
    for (spk, d) in raw
        all(haskey(d, v) && length(d[v]) >= 20 for v in values(VOWELS)) || continue
        m = Dict(v => median(d[v]) for v in values(VOWELS))
        allv = vcat(values(d)...)
        wd = get(weight, spk, Dict{String, Vector{Float64}}())
        hv = haskey(wd, "heavy") && length(wd["heavy"]) >= 20 ? median(wd["heavy"]) : NaN
        lv = haskey(wd, "light") && length(wd["light"]) >= 20 ? median(wd["light"]) : NaN
        push!(rows, (spk = spk, anchor = spk in ANCHORS, n = length(allv),
                     fatha = m["fatha"], damma = m["damma"], kasra = m["kasra"],
                     # isochrony: 0 is perfect; the spread of the three medians
                     isochrony = maximum(values(m)) - minimum(values(m)),
                     ikhtilas_rate = mean(allv .< IKHTILAS), ishba_rate = mean(allv .> ISHBA),
                     heavy = hv, light = lv, weight_bias = hv - lv))
    end
    sort!(rows, by = x -> x.isochrony)

    println(rpad("reciter", 44), rpad("fatha", 7), rpad("damma", 7), rpad("kasra", 7),
            rpad("isochr", 8), rpad("ikhtil", 8), rpad("ishba", 8), "wt.bias")
    for x in rows
        println(rpad((x.anchor ? "* " : "  ") * split(x.spk, ":")[end], 44),
                rpad(round(x.fatha; digits = 3), 7), rpad(round(x.damma; digits = 3), 7),
                rpad(round(x.kasra; digits = 3), 7), rpad(round(x.isochrony; digits = 4), 8),
                rpad(round(x.ikhtilas_rate; digits = 4), 8), rpad(round(x.ishba_rate; digits = 4), 8),
                isnan(x.weight_bias) ? "-" : round(x.weight_bias; digits = 4))
    end
    anc(f) = median([f(x) for x in rows if x.anchor])
    oth(f) = median([f(x) for x in rows if !x.anchor])
    println("\nanchors vs others — isochrony ", round(anc(x -> x.isochrony); digits = 4), " / ",
            round(oth(x -> x.isochrony); digits = 4),
            " | ikhtilās ", round(anc(x -> x.ikhtilas_rate); digits = 4), " / ",
            round(oth(x -> x.ikhtilas_rate); digits = 4),
            " | ishbā' ", round(anc(x -> x.ishba_rate); digits = 4), " / ",
            round(oth(x -> x.ishba_rate); digits = 4))
    open(io -> JSON3.pretty(io, Dict("clips" => n, "ikhtilas_threshold" => IKHTILAS,
                                     "ishba_threshold" => ISHBA,
                                     "reciters" => [Dict(pairs(x)) for x in rows],
                                     "anchor_isochrony" => anc(x -> x.isochrony),
                                     "other_isochrony" => oth(x -> x.isochrony))), out, "w")
    println("$n clips in $(round(time() - t0; digits = 1)) s -> $out")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS[1], ARGS[2])
end
