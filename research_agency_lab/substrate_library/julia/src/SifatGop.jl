# Sifat GOP — likelihood-ratio scoring of the ten tajweed attribute heads.
#
# muaalem is a *multi-level* CTC model: besides the 43-column phoneme head it emits ten sifat heads
# (columns 43–74 of the dump) — ghunnah, hams/jahr, istitala, itbaq, qalqala, safeer,
# shidda/rakhawa, tafashie, tafkheem/tarqeeq, tikraar. Every dump we have ever computed already
# contains them; until now every analysis sliced only the phoneme block and threw them away.
#
# `quran_transcript`'s phonetizer states, per reference phoneme, which value of each sifah is
# *supposed* to be realised (exported per character by experiments/learner_eval/sifat_ref.py). So for
# a unit occupying frames t0:t1 we can ask the model directly: is the expected attribute more likely
# over this span than its best competitor?
#
#     s_c   = logΣ_t exp(logP_t(c)) − log(nframes)        (mean posterior of class c, in log space)
#     LLR   = s_expected − max_{c ≠ expected, c ≠ PAD} s_c
#
# LLR < τ means the model heard a *different* sifah than the text demands — a tajweed error located
# at that unit and named by the competitor that won. τ is conformal on the anchors, per (level,
# expected class), so "how often may a near-gold reciter be flagged" is a coverage statement rather
# than a tuned constant. This replaces the hand-written DSP detectors whose false-FAIL rates on known
# -good peers were measured at takreer 19 %, jahr 14 %, shiddah 10 %.

export SIFAT_LEVELS, sifat_levels, sifat_llr, load_sifat_ref

# A sifah is a property of a CONSONANT. During a short vowel the folds always vibrate, so scoring
# "expected hams" over a vowel's frames yields a systematic false violation; madd letters are
# prolongations of a vowel and carry no sifah of their own either. Measured over 801 T300 clips,
# 40.2 % of units are short vowels and 9.6 % madd — half of everything scored before this filter.
const SIFAT_SKIP = Set(['َ', 'ُ', 'ِ', 'ا', 'ۥ', 'ۦ'])

const SIFAT_LEVELS = ["ghonna", "hams_or_jahr", "istitala", "itbaq", "qalqla", "safeer",
                      "shidda_or_rakhawa", "tafashie", "tafkheem_or_taqeeq", "tikraar"]

"""
    sifat_levels(dir) -> Dict(level => (first, width, vocab))

Column block of each sifat head in a dump's `layout.json`; `first` is 0-based as written, `vocab` is
the head's token list (index 1 = class id 0 = `[PAD]`).
"""
function sifat_levels(dir)
    L = JSON3.read(read(joinpath(dir, "layout.json"), String))
    out = Dict{String, Tuple{Int, Int, Vector{String}}}()
    for l in L.levels
        lvl = String(l.level)
        lvl == "phonemes" && continue
        out[lvl] = (Int(l.first), Int(l.width), String.(collect(l.vocab)))
    end
    return out
end

"""
    load_sifat_ref(dir) -> Dict(clip id => Dict(level => Vector{Int}))

Per-character expected class ids written by `sifat_ref.py` (0 = unlabelled/PAD).
"""
function load_sifat_ref(dir)
    out = Dict{String, Dict{String, Vector{Int}}}()
    path = joinpath(dir, "sifat.jsonl")
    isfile(path) || return out
    for line in eachline(path)
        r = JSON3.read(line)
        out[String(r.id)] = Dict(String(k) => Int.(collect(v)) for (k, v) in pairs(r.levels))
    end
    return out
end

"mean posterior of each column of `X` in log space (log of the average probability)."
function _pooled(X::AbstractMatrix{Float32})
    n = size(X, 1)
    out = Vector{Float64}(undef, size(X, 2))
    @inbounds for j in 1:size(X, 2)
        m = -Inf32
        for t in 1:n
            X[t, j] > m && (m = X[t, j])
        end
        s = 0.0
        for t in 1:n
            s += exp(Float64(X[t, j] - m))
        end
        out[j] = Float64(m) + log(s) - log(n)
    end
    return out
end

"""
    sifat_llr(lp_full, blocks, units, frames, expected) -> Vector{NamedTuple}

One record per (unit, level) where the reference states an attribute: the log-likelihood ratio of the
expected class against its best competitor over that unit's frames, and the name of the competitor.

* `lp_full`  T×C log-posteriors of the whole dump row (all 75 columns).
* `blocks`   output of [`sifat_levels`](@ref).
* `units`    `(symbol, a, b)` character spans, as `ph_units` returns.
* `frames`   `(t0, t1)` Viterbi frame span per unit (the `frames` field of `gop_sf`).
* `expected` level → per-character expected class id.
"""
function sifat_llr(lp_full::AbstractMatrix{Float32}, blocks, units, frames,
                   expected::Dict{String, Vector{Int}})
    T = size(lp_full, 1)
    out = NamedTuple[]
    for (i, (sym, a, b)) in enumerate(units)
        sym in SIFAT_SKIP && continue          # vowels and madd letters carry no sifah
        t0, t1 = frames[i]
        (t0 >= 1 && t1 >= t0 && t1 <= T) || continue
        for lvl in SIFAT_LEVELS
            haskey(blocks, lvl) || continue
            ids = get(expected, lvl, Int[])
            a <= length(ids) || continue
            c = ids[a]
            c > 0 || continue                      # no expected value here (PAD)
            first, width, vocab = blocks[lvl]
            X = @view lp_full[t0:t1, first+1:first+width]
            s = _pooled(X)
            ecol = c + 1                           # class id 0 = [PAD] = column 1
            ecol <= width || continue
            best, bestcol = -Inf, 0
            for j in 2:width                       # skip [PAD]
                j == ecol && continue
                s[j] > best && (best = s[j]; bestcol = j)
            end
            bestcol == 0 && continue               # binary head with only the expected class
            push!(out, (unit = i, symbol = sym, span = (a, b), frames = (t0, t1), level = lvl,
                        expected = vocab[ecol], llr = s[ecol] - best, best = vocab[bestcol]))
        end
    end
    return out
end
