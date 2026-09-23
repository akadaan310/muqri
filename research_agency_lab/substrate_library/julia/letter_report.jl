# The complete record of every letter recited — not just what went wrong.
#
# The app must be able to open any single letter and show its ENTIRE state: which of its
# characteristics were realised and which were not, its timing, and how confident the engine is in
# the letter's identity. A letter carries five to seven sifāt in the classical count (ر carries
# seven); muaalem gives a trained head for ten attributes, so every letter gets a full attribute
# vector with the *expected* value from the phonetizer beside the model's judgement — including the
# attributes that are correctly ABSENT, because "no qalqala here, and none was produced" is as much
# a part of the record as a violation.
#
# Two modes:
#   report    one clip -> a JSON array of per-letter records (the app's depth view)
#   coverage  the whole dump -> per reciter, what fraction of the phonetizer's expected letters and
#             attributes the engine actually confirms. This is the honest answer to "if someone
#             recites one word, do we really know everything about every letter?"
#
#   julia --project=research_agency_lab/substrate_library/julia \
#         research_agency_lab/substrate_library/julia/letter_report.jl DUMP_DIR OUT.json [--clip ID]

using QaariLab, JSON3, Statistics

const SHORT_V = Set(['َ', 'ُ', 'ِ'])
const MADD = Set(['ا', 'ۥ', 'ۦ'])
fin(x) = isfinite(x) ? round(x; digits = 3) : nothing

# The classical count: five OPPOSING pairs give every letter exactly one member each
#   hams/jahr · shidda|tawassuṭ|rakhāwa · isti'lā'/istifāl · iṭbāq/infitāḥ · idhlāq/iṣmāt
# and the non-opposing sifāt are counted only when present. So ر = 5 + takrīr + inḥirāf = seven.
# Two of the five pairs (isti'lā', idhlāq) have no acoustic head — they are inherent to the letter and
# come from app/data/letter_reference.json; tafkhīm is the *consequence* of isti'lā' in context, not a
# sixth pair, so it is reported as judged evidence but not counted again.
const BASE_PAIRS = ["hams_or_jahr", "shidda_or_rakhawa", "itbaq"]      # the three with heads
const EXTRA_HEADS = ["ghonna", "safeer", "qalqla", "tikraar", "tafashie", "istitala"]
# the value that means "this letter does NOT carry the sifah"
const UNMARKED = Set(["[لا غنة]", "[لا صفير]", "[لا قلقلة]", "[لا تكرار]", "[لا تفشي]", "[لا إستطالة]"])

function load_letter_ref(path)
    isfile(path) || return Dict{String, Any}()
    return Dict(String(k) => v for (k, v) in pairs(JSON3.read(read(path, String))))
end
const SPECIAL = Dict('ں' => "ikhfa_noon", '۾' => "iqlab_meem", 'ڇ' => "qalqala_release")
const ANCHORS = Set(["everyayah:Husary_Muallim_128kbps", "everyayah:Husary_128kbps",
                     "everyayah:Husary_128kbps_Mujawwad"])
speaker_of(id) = String(split(String(id), '/')[1])

"""Everything the engine knows about every unit of one clip."""
function analyse(dir, rec, C, c0, W, vocab, blank, blocks, expected, lref = Dict{String, Any}())
    ph = String(rec.ref_ph)
    full = read_dump_clip(dir, rec, C)
    lp = full[:, c0+1:c0+W]
    g = gop_sf(lp, ph, vocab, blank)
    units = ph_units(ph)
    nu = length(units)
    seq = [vocab[c] for c in ph]
    _, f, l = ctc_viterbi(lp, seq, blank)
    onset(i) = f[units[i][2]]
    dur_i(i) = (i < nu ? onset(i + 1) - onset(i) : l[units[i][3]] + 1 - onset(i)) * 0.04
    # The reciter's own count unit (Treatise I §2.2). A harakah is the time to articulate a VOWELLED
    # LETTER — consonant contact plus its vowel (the time to say كَ) — not the vowel alone. This must
    # match sukoon_timing.jl exactly or the two modules report different "counts" for the same audio.
    hs = Float64[]
    for i in 1:(nu - 1)
        (!(units[i][1] in SHORT_V) && !(units[i][1] in MADD) && units[i + 1][1] in SHORT_V) || continue
        push!(hs, (onset(min(i + 2, nu)) - onset(i)) * 0.04)
    end
    h = length(hs) >= 5 ? median(hs) : NaN

    frames = [u.frames for u in g]
    sif = sifat_llr(full, blocks, units, frames, expected)
    bylevel = Dict{Int, Vector{Any}}()
    for r in sif
        push!(get!(bylevel, r.unit, Any[]), r)
    end

    out = []
    for (i, (sym, a, b)) in enumerate(units)
        gi = g[i]
        kind = sym in SHORT_V ? "harakah" : sym in MADD ? "madd" :
               haskey(SPECIAL, sym) ? SPECIAL[sym] : "consonant"
        chars = Dict{String, Any}()
        for r in get(bylevel, i, Any[])
            chars[r.level] = Dict("expected" => r.expected, "model_best" => r.best,
                                  "llr" => fin(r.llr),
                                  # positive LLR = the model agrees the expected attribute was realised
                                  "realised" => r.llr > 0)
        end
        # inherent sifāt and makhraj: fixed properties of the letter, no model can judge them
        inherent = Dict{String, Any}()
        lr = get(lref, string(sym), nothing)
        if lr !== nothing
            inherent["makhraj"] = String(lr.makhraj)
            inherent["makhraj_zone"] = String(lr.makhraj_zone)
            inherent["istila"] = String(lr.istila)
            inherent["idhlaq"] = String(lr.idhlaq)
            lr.inhiraf && (inherent["inhiraf"] = true)
            lr.lin && (inherent["lin"] = true)
        end
        # classical count: 5 opposing pairs (3 judged + isti'lā' + idhlāq) plus the extras present
        n_marked = if kind == "consonant" && lr !== nothing
            5 + count(k -> haskey(chars, k) && !(chars[k]["expected"] in UNMARKED), EXTRA_HEADS) +
                (lr.inhiraf ? 1 : 0) + (lr.lin ? 1 : 0)
        else
            0
        end
        push!(out, Dict(
            "i" => i, "symbol" => string(sym), "kind" => kind, "run_length" => b - a + 1,
            "char_span" => [a, b], "frames" => [gi.frames[1], gi.frames[2]],
            "onset_s" => round((onset(i) - 1) * 0.04; digits = 3),
            "duration_s" => round(dur_i(i); digits = 3),
            "duration_counts" => isnan(h) || h <= 0 ? nothing : round(dur_i(i) / h; digits = 2),
            "identity" => Dict("gop" => fin(Float64(gi.gop)),
                               "best_competitor" => gi.best,
                               # -Inf when the unit has no confusion alternative (CtcGop NEG); JSON has no -Inf
                               "competitor_llr" => fin(Float64(gi.lr)),
                               # the reference symbol beat every competitor
                               "confirmed" => Float64(gi.lr) < 0),
            "characteristics" => chars, "inherent" => inherent,
            "n_judged" => length(chars), "n_marked" => n_marked,
            "n_characteristics" => length(chars) + length(inherent)))
    end
    return out, h
end

function main(dir, out; clip = nothing)
    C, c0, W, vocab, blank = load_dump_layout(dir)
    blocks = sifat_levels(dir)
    refs = load_sifat_ref(dir)
    isempty(refs) && error("no sifat.jsonl — run experiments/learner_eval/sifat_ref.py first")
    lref = load_letter_ref(joinpath(@__DIR__, "..", "..", "..", "app", "data", "letter_reference.json"))
    isempty(lref) && @warn "app/data/letter_reference.json missing — run datastore.letter_reference"

    if clip !== nothing
        for line in eachline(joinpath(dir, "index.jsonl"))
            rec = JSON3.read(line)
            String(get(rec, :id, "")) == clip || continue
            haskey(rec, :error) && error("clip $clip has an error record")
            recs, h = analyse(dir, rec, C, c0, W, vocab, blank, blocks, refs[clip], lref)
            nch = [r["n_characteristics"] for r in recs]
            println("clip $clip — $(length(recs)) units, haraka $(round(h; digits=3)) s")
            nm = [r["n_marked"] for r in recs if r["kind"] == "consonant"]
            println("characteristics per unit: min $(minimum(nch)) max $(maximum(nch)) mean $(round(mean(nch); digits=2))")
            println("classical sifāt per consonant: min $(minimum(nm)) max $(maximum(nm)) mean $(round(mean(nm); digits=2))")
            for r in recs[1:min(12, end)]
                ch = join(["$(k)=$(v["expected"])$(v["realised"] ? "✓" : "✗")"
                           for (k, v) in sort(collect(r["characteristics"]), by = first)], " ")
                inh = join(["$k=$v" for (k, v) in sort(collect(r["inherent"]), by = first)
                            if k != "makhraj"], " ")
                println("  ", lpad(r["i"], 3), " ", rpad(r["symbol"], 3), rpad(r["kind"], 16),
                        " ", rpad(string(r["duration_counts"]), 6), "U  ",
                        r["identity"]["confirmed"] ? "id✓" : "id✗",
                        "  sifāt=", r["n_marked"], "  ", ch, "  | ", inh)
            end
            open(io -> JSON3.pretty(io, Dict("clip" => clip, "haraka_s" => h, "units" => recs)), out, "w")
            println("-> $out")
            return
        end
        error("clip $clip not found")
    end

    # coverage mode: how complete is the record, per reciter
    agg = Dict{String, Dict{String, Vector{Float64}}}()
    n = 0
    t0 = time()
    for line in eachline(joinpath(dir, "index.jsonl"))
        rec = JSON3.read(line)
        haskey(rec, :error) && continue
        id = String(rec.id)
        haskey(refs, id) || continue
        ph = String(rec.ref_ph)
        all(c -> haskey(vocab, c), ph) || continue
        recs, h = analyse(dir, rec, C, c0, W, vocab, blank, blocks, refs[id], lref)
        isempty(recs) && continue
        spk = speaker_of(id)
        d = get!(agg, spk, Dict{String, Vector{Float64}}())
        push!(get!(d, "units", Float64[]), length(recs))
        push!(get!(d, "chars_per_unit", Float64[]), mean(r["n_characteristics"] for r in recs))
        push!(get!(d, "id_confirmed", Float64[]), mean(r["identity"]["confirmed"] for r in recs))
        push!(get!(d, "haraka_s", Float64[]), h)
        for k in ("harakah", "madd", "consonant")
            v = [r for r in recs if r["kind"] == k]
            isempty(v) || push!(get!(d, "id_$k", Float64[]), mean(x["identity"]["confirmed"] for x in v))
        end
        allc = [c for r in recs for c in values(r["characteristics"])]
        isempty(allc) || push!(get!(d, "attr_realised", Float64[]), mean(c["realised"] for c in allc))
        n += 1
        n % 2000 == 0 && println(stderr, "$n clips, $(round(time() - t0; digits = 1)) s")
    end

    rows = []
    for (spk, d) in agg
        # a short ayah can yield fewer than five consonant+vowel pairs, so its haraka is NaN; median
        # propagates a single NaN to the whole reciter, so drop non-finite values before aggregating
        function m(k)
            haskey(d, k) || return NaN
            v = filter(isfinite, d[k])
            return isempty(v) ? NaN : median(v)
        end
        push!(rows, (spk = spk, anchor = spk in ANCHORS, clips = length(d["units"]),
                     units = m("units"), chars_per_unit = m("chars_per_unit"),
                     haraka_s = m("haraka_s"), id_confirmed = m("id_confirmed"),
                     id_harakah = m("id_harakah"), id_madd = m("id_madd"),
                     id_consonant = m("id_consonant"), attr_realised = m("attr_realised")))
    end
    sort!(rows, by = x -> -x.id_confirmed)
    println(rpad("reciter", 44), rpad("hrk_s", 7), rpad("chars", 7), rpad("id_all", 8),
            rpad("id_vowel", 9), rpad("id_madd", 9), rpad("id_cons", 9), "attr✓")
    for x in rows
        println(rpad((x.anchor ? "* " : "  ") * split(x.spk, ":")[end], 44),
                rpad(round(x.haraka_s; digits = 2), 7), rpad(round(x.chars_per_unit; digits = 2), 7),
                rpad(round(x.id_confirmed; digits = 4), 8), rpad(round(x.id_harakah; digits = 4), 9),
                rpad(round(x.id_madd; digits = 4), 9), rpad(round(x.id_consonant; digits = 4), 9),
                round(x.attr_realised; digits = 4))
    end
    println("\nunits/clip median ", round(median([x.units for x in rows]); digits = 1),
            " | characteristics per unit ", round(median([x.chars_per_unit for x in rows]); digits = 2))
    jsonable(x) = Dict(k => (v isa AbstractFloat && !isfinite(v)) ? nothing : v for (k, v) in pairs(x))
    open(io -> JSON3.pretty(io, Dict("clips" => n, "reciters" => [jsonable(x) for x in rows])), out, "w")
    println("$n clips in $(round(time() - t0; digits = 1)) s -> $out")
end

if abspath(PROGRAM_FILE) == @__FILE__
    args = copy(ARGS)
    clip = nothing
    if (i = findfirst(==("--clip"), args)) !== nothing
        clip = args[i+1]; deleteat!(args, i:i+1)
    end
    main(args[1], args[2]; clip = clip)
end
