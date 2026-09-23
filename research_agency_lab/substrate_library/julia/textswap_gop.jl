# Counterfactual text-swap test of the muaalem segmentation-free GOP: does it flag the *right* mistake?
#
# Masters' audio is taken as (nearly) correct at the swapped unit. Swapping one letter or harakah of
# the *text* to a classical confusion (ض→د, ح→ه, fatha→kasra, …) makes the audio disagree with the text
# at exactly that unit, in a known way. A swap is
#   detected         if GOP of the swapped unit crosses the threshold, and
#   correctly named  if, in addition, its best competitor is the original symbol (the audio's letter).
# Thresholds are conformal (Frontier.conformal_threshold) at level α on the clean-text GOP of the
# near-gold ANCHORS only, per kind (consonant / vowel). Other reciters' clean-text flag rates are
# reported per speaker but are not false alarms by definition: faster reciters make real mistakes.
#
#   julia --project=research_agency_lab/substrate_library/julia research_agency_lab/substrate_library/julia/textswap_gop.jl DUMP_DIR OUT.json [--alpha 0.01]

using QaariLab, JSON3, Statistics

const ANCHORS = Set(["everyayah:Husary_Muallim_128kbps", "everyayah:Husary_128kbps",
                     "everyayah:Husary_128kbps_Mujawwad", "qdc:12", "qdc:6"])
const VOWELS = Set(['َ', 'ُ', 'ِ'])
# audio symbol → symbols the text is swapped to (same sets as experiments/lahn_gop/text_swap.py)
const SWAPS = Dict{Char, Vector{Char}}(
    'ض' => ['د', 'ظ'], 'د' => ['ض'], 'ص' => ['س'], 'س' => ['ص', 'ث'], 'ط' => ['ت'], 'ت' => ['ط'],
    'ح' => ['ه'], 'ه' => ['ح'], 'ع' => ['ء'], 'ق' => ['ك'], 'ك' => ['ق'], 'ث' => ['س'], 'ذ' => ['ز', 'د'],
    'ز' => ['ذ'], 'ظ' => ['ز'], 'غ' => ['خ'], 'خ' => ['غ'],
    'َ' => ['ِ', 'ُ'], 'ِ' => ['َ', 'ُ'], 'ُ' => ['َ', 'ِ'],
)

speaker_of(id) = String(split(String(id), '/')[1])   # "everyayah:<folder>" or "qdc:<reciter id>"

function main(dir, out; α = 0.01)
    C, c0, W, vocab, blank = load_dump_layout(dir)
    t0 = time()
    n = 0
    # 414 clips take ~64 s single-threaded, so a full-Quran or T300 dump is ~30× that. Every clip is
    # independent and gop_sf/read_as_recited are pure, so fan the clips over threads and merge the
    # per-thread buffers at the end (run julia with -t auto).
    clean = Dict{String, Vector{Tuple{String, Float64}}}()       # speaker → (kind, badness)
    swaps = NamedTuple[]
    ndev = Dict{String, Int}()                                  # speaker → repetitions + skips found
    # Sequential on purpose: a chunked Threads.@spawn version raced inside push! even with per-task
    # buffers, so something below gop_sf is not thread-safe. 414 clips take 64 s, T300 ~32 min
    # unattended, and the dump itself (now 50 Modal containers) is the real long pole.
    for line in eachline(joinpath(dir, "index.jsonl"))
        rec = JSON3.read(line)
        haskey(rec, :error) && continue
        ph = String(rec.ref_ph)
        all(c -> haskey(vocab, c), ph) || continue
        spk = speaker_of(rec.id)
        lp = read_dump_clip(dir, rec, C)[:, c0+1:c0+W]
        # score against the text as read (repetitions inserted); units inside a repeated copy are left out
        path, origin, devs = read_as_recited(lp, ph, vocab, blank)
        ndev[spk] = get(ndev, spk, 0) + length(devs)
        units = ph_units(path)
        for u in gop_sf(lp, path, vocab, blank)
            origin[u.span[1]] > 0 || continue
            kind = u.symbol in VOWELS ? "vowel" : "consonant"
            (u.symbol in keys(SWAPS)) && push!(get!(clean, spk, Tuple{String, Float64}[]), (kind, -Float64(u.gop)))
        end
        cs = collect(path)
        for (i, (y, a, b)) in enumerate(units)
            haskey(SWAPS, y) || continue
            origin[a] > 0 || continue
            for q in SWAPS[y]
                haskey(vocab, q) || continue
                # a swap that merges with a neighbouring run would change the unit structure: skip
                (i > 1 && units[i-1][1] == q) && continue
                (i < length(units) && units[i+1][1] == q) && continue
                sc = copy(cs); sc[a:b] .= q
                g = only(gop_sf(lp, String(sc), vocab, blank; only = Set([i])))
                push!(swaps, (speaker = spk, kind = y in VOWELS ? "vowel" : "consonant", audio = string(y),
                              text = string(q), bad = -Float64(g.gop), best = g.best))
            end
        end
        n += 1
        n % 50 == 0 && println(stderr, "$n clips, $(length(swaps)) swaps, $(round(time() - t0; digits = 1)) s")
    end
    res = Dict{String, Any}("alpha" => α, "clips" => n, "anchors" => collect(ANCHORS), "deviations" => ndev)
    for kind in ("consonant", "vowel")
        null = [b for (spk, v) in clean if spk in ANCHORS for (k, b) in v if k == kind]
        τ, ok = conformal_threshold(null, α)
        sw = [s for s in swaps if s.kind == kind]
        det = [s.bad > τ for s in sw]
        named = [s.bad > τ && s.best == s.audio for s in sw]
        per_pair = Dict{String, Any}()
        for key in unique((s.audio, s.text) for s in sw)
            idx = [j for (j, s) in enumerate(sw) if (s.audio, s.text) == key]
            per_pair["$(key[1])→$(key[2])"] = Dict("n" => length(idx), "recall" => mean(det[idx]),
                                                   "named" => mean(named[idx]))
        end
        per_speaker = Dict{String, Any}()
        for spk in sort(collect(keys(clean)))
            v = [b for (k, b) in clean[spk] if k == kind]
            idx = [j for (j, s) in enumerate(sw) if s.speaker == spk]
            per_speaker[spk] = Dict("anchor" => spk in ANCHORS, "clean_units" => length(v),
                                    "clean_flag_rate" => isempty(v) ? NaN : mean(v .> τ),
                                    "swaps" => length(idx), "recall" => isempty(idx) ? NaN : mean(det[idx]),
                                    "named" => isempty(idx) ? NaN : mean(named[idx]))
        end
        res[kind] = Dict("threshold" => τ, "threshold_exact" => ok, "anchor_null_n" => length(null),
                         "swaps" => length(sw), "recall" => mean(det), "named_recall" => mean(named),
                         "per_pair" => per_pair, "per_speaker" => per_speaker)
        println(rpad(kind, 10), " τ=", round(τ; digits = 3), " (anchor null n=", length(null), ")  swaps=", length(sw),
                "  recall=", round(mean(det); digits = 3), "  named=", round(mean(named); digits = 3))
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
