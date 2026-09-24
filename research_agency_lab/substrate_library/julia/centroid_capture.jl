# Capture the Julia reference for sub-frame onsets, so the numpy serving path can be pinned to it.
#
# For one dump clip, writes every label's continuous centroid onset (1-based frames, as
# `ctc_viterbi`'s `first`) and its occupancy mass. `tests/test_analysis_parity.py` compares
# `app/analysis.py::centroid_onsets` against this file.
#
#   julia --project=research_agency_lab/substrate_library/julia \
#         research_agency_lab/substrate_library/julia/centroid_capture.jl DUMP_DIR CLIP_ID OUT.json

using QaariLab, JSON3

function main(dir, clip, out)
    C, c0, W, vocab, blank = load_dump_layout(dir)
    rec = nothing
    for line in eachline(joinpath(dir, "index.jsonl"))
        r = JSON3.read(line)
        if get(r, :id, nothing) == clip && haskey(r, :file)
            rec = r
            break
        end
    end
    rec === nothing && error("clip $clip not in dump")
    lp = read_dump_clip(dir, rec, C)[:, c0+1:c0+W]
    seq = [vocab[c] for c in String(rec.ref_ph)]
    g = ctc_occupancy(lp, seq, blank)
    cen = centroid_onsets(lp, seq, blank)
    open(out, "w") do io
        JSON3.write(io, (clip = clip, onsets = cen, mass = vec(sum(g; dims = 1))))
    end
    println("wrote $(length(cen)) centroids for $clip")
end

main(ARGS...)
