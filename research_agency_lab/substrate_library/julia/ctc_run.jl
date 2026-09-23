# Phoneme-level timestamps + segmentation-free GOP for every clip of a muaalem posterior dump.
#
#   julia --project=research_agency_lab/substrate_library/julia research_agency_lab/substrate_library/julia/ctc_run.jl DUMP_DIR OUT.jsonl [--export XCHK_DIR]
#
# DUMP_DIR is written by research_agency_lab/experiments/learner_eval/muaalem_dump.py (layout.json,
# index.jsonl, <id>.f32 row-major T×C float32). One output line per clip:
#   id, source, speaker, surah/ayah (masters) or edits (QuranMB), frame_s,
#   viterbi_score, logp (CTC forward of the reference), per-character first/last frame,
#   words: [[t0, t1], …] seconds (Uthmani word order, first reading), units: [[symbol, char0, char1, gop,
#   best, lr], …] on the reading path (negative char index = inside a repeated copy), deviations
#   (repetitions / skips against the text; see CtcGop.deviations)
# --export writes the first 5 clips' posteriors and results as plain TSV for octave/ctc_crosscheck.m.

using QaariLab, JSON3

function main(dir, out; export_dir = "")
    C, c0, W, vocab, blank = load_dump_layout(dir)
    n = nerr = 0
    exported = 0
    export_dir != "" && mkpath(export_dir)
    t0 = time()
    open(out, "w") do io
        for line in eachline(joinpath(dir, "index.jsonl"))
            rec = JSON3.read(line)
            haskey(rec, :error) && continue
            lp = read_dump_clip(dir, rec, C)[:, c0+1:c0+W]
            ph = String(rec.ref_ph)
            if any(!haskey(vocab, c) for c in ph)
                nerr += 1
                continue
            end
            frame_s = rec.duration_s / rec.frames
            # align to the text as it was actually read (repetitions inserted), then map back
            path, origin, devs = read_as_recited(lp, ph, vocab, blank)
            seq = [vocab[c] for c in path]
            score, pf, pl = ctc_viterbi(lp, seq, blank)
            logp = ctc_forward(lp, seq, blank)
            units = gop_sf(lp, path, vocab, blank)
            f = zeros(Int, length(ph)); l = zeros(Int, length(ph))
            for (p, o) in enumerate(origin)       # first reading of every reference character
                o > 0 && (f[o] = pf[p]; l[o] = pl[p])
            end
            words = unit_times(f, l, [(w[1], w[2]) for w in rec.word_ph], frame_s)
            row = Dict{String, Any}(
                "id" => rec.id, "source" => rec.source, "speaker" => get(rec, :speaker, ""),
                "frame_s" => frame_s, "frames" => rec.frames, "viterbi_score" => score, "logp" => logp,
                "logp_per_frame" => logp / rec.frames,
                "first" => f, "last" => l,
                "words" => [[round(a; digits = 3), round(b; digits = 3)] for (a, b) in words],
                # units: symbol, reference char span (negative = inside a repeated copy), gop, best, lr
                "units" => [[string(u.symbol), origin[u.span[1]], origin[u.span[2]], round(u.gop; digits = 4), u.best,
                             isfinite(u.lr) ? round(u.lr; digits = 4) : nothing] for u in units],
                "deviations" => [Dict("kind" => String(d.kind), "ref_from" => d.ref_from, "ref_to" => d.ref_to,
                                      "at" => d.at) for d in devs])
            for k in (:sura, :aya, :edits, :text)
                haskey(rec, k) && (row[String(k)] = rec[k])
            end
            JSON3.write(io, row); println(io)
            if export_dir != "" && exported < 5
                exported += 1
                base = joinpath(export_dir, "clip$(exported)")
                open(io2 -> foreach(r -> println(io2, join(r, '\t')), eachrow(lp)), base * "_lp.tsv", "w")
                write(base * "_seq.tsv", join(seq, '\t') * "\n")
                write(base * "_jl.tsv", string(logp, '\t', score, "\n", join(f, '\t'), "\n", join(l, '\t'), "\n"))
            end
            n += 1
            n % 50 == 0 && println(stderr, "$n clips, $(round(time() - t0; digits = 1)) s")
        end
    end
    println("$n clips aligned ($nerr with symbols outside the vocabulary) in $(round(time() - t0; digits = 1)) s -> $out")
    export_dir != "" && write(joinpath(export_dir, "blank.tsv"), "$blank\n")
end

if abspath(PROGRAM_FILE) == @__FILE__
    args = copy(ARGS)
    ex = ""
    if (i = findfirst(==("--export"), args)) !== nothing
        ex = args[i+1]; deleteat!(args, i:i+1)
    end
    main(args[1], args[2]; export_dir = ex)
end
