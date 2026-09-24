# CTC alignment and segmentation-free GOP on muaalem-v3.2 posteriors (included by QaariLab.jl).
#
# Research basis: deep_research/01_makharij.md §3.1b (Cao et al., arXiv:2507.16838; restricted
# substitutions, Parikh et al., arXiv:2506.02080); graph nodes algo:gop_segmentation_free,
# phen:lahn:jali. Mirrored in Octave by octave/ctc_forward.m, octave/ctc_viterbi.m and checked by
# ctc_crosscheck.jl / octave/ctc_crosscheck.m.
#
# muaalem's phoneme script writes one character per phoneme *slot*: a long vowel or a geminate is a
# run of the same character (ا ا ا ا, ص ص). A *unit* is one such run. For unit i with written symbol y
# and a restricted confusion set S_i (ض→د/ظ, ح→ه, fatha↔kasra↔damma, …, plus deletion):
#
#     LR_i(q) = log P_CTC(window with unit i := q | X_w) − log P_CTC(window | X_w)
#     P̂_i(y)  = 1 / (1 + Σ_q exp LR_i(q)),   GOP_i = log P̂_i(y)   (≤ 0; ≈ 0 when the text is supported)
#
# The window is the units i−ctx … i+ctx and the frames the Viterbi alignment gives them (± margin),
# so every hypothesis is scored on the same frames: segmentation-free inside the window.

export ctc_forward, ctc_viterbi, ctc_occupancy, centroid_onsets, ph_units, gop_sf, unit_times, CONFUSIONS, load_dump_layout, read_dump_clip

const NEG = -Inf32

@inline function lse(a::Float32, b::Float32)
    a == NEG && return b
    b == NEG && return a
    m = max(a, b)
    return m + log1p(exp(-abs(a - b)))
end

"""log P(seq | lp) over all CTC paths. `lp` is T×V log-posteriors, `seq` 1-based column indices."""
function ctc_forward(lp::AbstractMatrix{Float32}, seq::AbstractVector{Int}, blank::Int)
    T = size(lp, 1)
    L = length(seq)
    L == 0 && return sum(@view lp[:, blank])
    S = 2L + 1
    ext = Vector{Int}(undef, S)
    for s in 1:S
        ext[s] = isodd(s) ? blank : seq[s ÷ 2]
    end
    α = fill(NEG, S)
    α[1] = lp[1, blank]
    α[2] = lp[1, ext[2]]
    nα = similar(α)
    for t in 2:T
        for s in 1:S
            v = α[s]
            s > 1 && (v = lse(v, α[s-1]))
            if s > 2 && ext[s] != blank && ext[s] != ext[s-2]
                v = lse(v, α[s-2])
            end
            nα[s] = v == NEG ? NEG : v + lp[t, ext[s]]
        end
        α, nα = nα, α
    end
    return lse(α[S], α[S-1])
end

"""
    ctc_occupancy(lp, seq, blank) -> T×L matrix

Posterior probability that label k of `seq` occupies frame t (CTC forward-backward in log space). The
occupancy of label k is γ at extended state 2k. Mirrored by `app/analysis.py::ctc_occupancy`.
"""
function ctc_occupancy(lp::AbstractMatrix{Float32}, seq::AbstractVector{Int}, blank::Int)
    T, L = size(lp, 1), length(seq)
    L == 0 && return zeros(T, 0)
    S = 2L + 1
    ext = [isodd(s) ? blank : seq[s ÷ 2] for s in 1:S]
    skip = [s > 2 && ext[s] != blank && ext[s] != ext[s-2] for s in 1:S]
    α = fill(-Inf, T, S)
    β = fill(-Inf, T, S)
    α[1, 1] = lp[1, ext[1]]
    α[1, 2] = lp[1, ext[2]]
    for t in 2:T, s in 1:S
        v = α[t-1, s]
        s > 1 && (v = lse64(v, α[t-1, s-1]))
        skip[s] && (v = lse64(v, α[t-1, s-2]))
        α[t, s] = v == -Inf ? -Inf : v + lp[t, ext[s]]
    end
    β[T, S] = 0.0
    β[T, S-1] = 0.0
    for t in T-1:-1:1, s in 1:S
        v = β[t+1, s] + lp[t+1, ext[s]]
        s < S && (v = lse64(v, β[t+1, s+1] + lp[t+1, ext[s+1]]))
        s + 2 <= S && skip[s+2] && (v = lse64(v, β[t+1, s+2] + lp[t+1, ext[s+2]]))
        β[t, s] = v
    end
    logZ = lse64(α[T, S], α[T, S-1])
    return [exp(clamp(α[t, 2k] + β[t, 2k] - logZ, -60.0, 0.0)) for t in 1:T, k in 1:L]
end

"""
    centroid_onsets(lp, seq, blank) -> Vector{Float64}

Continuous onset of each label, in 1-based frames like `ctc_viterbi`'s `first`: the centre of mass
of its occupancy. Breaks the 40 ms quantisation of a Viterbi span. NaN for a label with no mass.
"""
function centroid_onsets(lp::AbstractMatrix{Float32}, seq::AbstractVector{Int}, blank::Int)
    g = ctc_occupancy(lp, seq, blank)
    t = collect(1.0:size(g, 1))
    return [(m = sum(@view g[:, k]); m > 1e-12 ? sum(t .* @view g[:, k]) / m : NaN) for k in 1:size(g, 2)]
end

@inline function lse64(a::Float64, b::Float64)
    a == -Inf && return b
    b == -Inf && return a
    m = max(a, b)
    return m + log1p(exp(-abs(a - b)))
end

"""Best CTC path. Returns (score, first frame, last frame) per label of `seq` (frames 1-based)."""
function ctc_viterbi(lp::AbstractMatrix{Float32}, seq::AbstractVector{Int}, blank::Int)
    T = size(lp, 1)
    L = length(seq)
    S = 2L + 1
    ext = [isodd(s) ? blank : seq[s ÷ 2] for s in 1:S]
    δ = fill(NEG, T, S)
    bp = zeros(Int8, T, S)  # 0 stay, 1 from s−1, 2 from s−2
    δ[1, 1] = lp[1, blank]
    S > 1 && (δ[1, 2] = lp[1, ext[2]])
    for t in 2:T, s in 1:S
        best, arg = δ[t-1, s], Int8(0)
        if s > 1 && δ[t-1, s-1] > best
            best, arg = δ[t-1, s-1], Int8(1)
        end
        if s > 2 && ext[s] != blank && ext[s] != ext[s-2] && δ[t-1, s-2] > best
            best, arg = δ[t-1, s-2], Int8(2)
        end
        δ[t, s] = best == NEG ? NEG : best + lp[t, ext[s]]
        bp[t, s] = arg
    end
    s = (S > 1 && δ[T, S-1] > δ[T, S]) ? S - 1 : S
    score = δ[T, s]
    first = zeros(Int, L)
    last = zeros(Int, L)
    for t in T:-1:1
        if iseven(s)
            k = s ÷ 2
            last[k] == 0 && (last[k] = t)
            first[k] = t
        end
        t > 1 && (s -= bp[t, s])
    end
    return score, first, last
end

"""Runs of identical characters in the phoneme string: (symbol, first char index, last char index)."""
function ph_units(ph::AbstractString)
    cs = collect(ph)
    units = Tuple{Char, Int, Int}[]
    i = 1
    while i <= length(cs)
        j = i
        while j < length(cs) && cs[j+1] == cs[i]
            j += 1
        end
        push!(units, (cs[i], i, j))
        i = j + 1
    end
    return units
end

# Restricted confusion sets on muaalem's phoneme characters (01 §3.5 priority pairs + app/lahn/gop.py).
const FATHA, DAMMA, KASRA = 'َ', 'ُ', 'ِ'
const CONFUSIONS = Dict{Char, Vector{Char}}(
    'ض' => ['د', 'ظ'], 'ظ' => ['ز', 'ذ', 'ض'], 'ذ' => ['ز', 'د'], 'ث' => ['س', 'ت'], 'ص' => ['س'],
    'س' => ['ص', 'ث'], 'ط' => ['ت'], 'ت' => ['ط'], 'ح' => ['ه'], 'ه' => ['ح'], 'ع' => ['ء'], 'ء' => ['ع'],
    'ق' => ['ك'], 'ك' => ['ق'], 'غ' => ['خ'], 'خ' => ['غ', 'ح'], 'ز' => ['ذ', 'ظ'], 'د' => ['ض', 'ذ'],
    FATHA => [KASRA, DAMMA], KASRA => [FATHA, DAMMA], DAMMA => [FATHA, KASRA],
    'ا' => ['ۦ', 'ۥ'], 'ۦ' => ['ا', 'ۥ'], 'ۥ' => ['ا', 'ۦ'],
)
const DELETABLE = Set(collect("بتثجحخدذرزسشصضطظعغفقكلمنهويء"))

"""Segmentation-free GOP for every unit of `ph` on log-posteriors `lp` (T×V, phoneme columns).

`vocab` maps a phoneme character to its 1-based column; `only` (a set of unit indices) restricts the
scoring to those units (the Viterbi alignment still covers the whole string). Returns one NamedTuple per unit:
symbol, char span, Viterbi frames, gop (log P̂(y)), the best competitor and its LR (\"∅\" = deletion)."""
function gop_sf(lp::AbstractMatrix{Float32}, ph::AbstractString, vocab::Dict{Char, Int}, blank::Int;
                ctx::Int = 2, margin::Int = 3, confusions = CONFUSIONS, deletable = DELETABLE,
                only = nothing)
    cs = collect(ph)
    seq = [vocab[c] for c in cs]
    units = ph_units(ph)
    _, f, l = ctc_viterbi(lp, seq, blank)
    T = size(lp, 1)
    out = NamedTuple[]
    for (i, (y, a, b)) in enumerate(units)
        only === nothing || i in only || continue
        lo, hi = max(1, i - ctx), min(length(units), i + ctx)
        ca, cb = units[lo][2], units[hi][3]
        t0 = max(1, f[ca] - margin)
        t1 = min(T, l[cb] + margin)
        X = @view lp[t0:t1, :]
        left = seq[ca:a-1]
        right = seq[b+1:cb]
        n = b - a + 1
        ref = ctc_forward(X, vcat(left, seq[a:b], right), blank)
        alts = Tuple{String, Float32}[]
        for q in get(confusions, y, Char[])
            haskey(vocab, q) || continue
            push!(alts, (string(q), ctc_forward(X, vcat(left, fill(vocab[q], n), right), blank) - ref))
        end
        # the ي / و of a madd leen (sakin after fatha) is a glide, not a consonant that can be dropped:
        # a certified reviewer rejected "dropped" on Husary's leen in ٱلْمَغْرِبَيْنِ
        leen = y in ('ي', 'و') && i > 1 && units[i-1][1] == FATHA &&
               (i == length(units) || !(units[i+1][1] in (FATHA, DAMMA, KASRA)))
        if y in deletable && !leen
            push!(alts, ("∅", ctc_forward(X, vcat(left, right), blank) - ref))
        end
        if isempty(alts)
            push!(out, (symbol = y, span = (a, b), frames = (f[a], l[b]), gop = 0.0f0, best = "", lr = NEG))
            continue
        end
        lrs = last.(alts)
        m = maximum(lrs)
        z = m > 0 ? exp(-m) + sum(exp.(lrs .- m)) : 1 + sum(exp.(lrs))
        gop = m > 0 ? -(m + log(z)) : -log(z)
        k = argmax(lrs)
        push!(out, (symbol = y, span = (a, b), frames = (f[a], l[b]), gop = Float32(gop),
                    best = alts[k][1], lr = lrs[k]))
    end
    return out
end

"""Word start/end times (s) from the Viterbi frames of the phoneme characters and each word's
[p0, p1) character span (``word_ph`` of the dump; −1 = word without phonemes)."""
function unit_times(first::Vector{Int}, last::Vector{Int}, word_ph, frame_s::Float64)
    out = Tuple{Float64, Float64}[]
    for (p0, p1) in word_ph
        if p0 < 0 || p1 <= p0
            push!(out, (NaN, NaN))
            continue
        end
        push!(out, ((first[p0+1] - 1) * frame_s, last[p1] * frame_s))
    end
    return out
end

# ------------------------------------------------ muaalem posterior dumps (muaalem_dump.py) --------
"""Column layout of a muaalem posterior dump: (columns, phoneme block offset, width, char → column, blank column)."""
function load_dump_layout(dir)
    L = JSON3.read(read(joinpath(dir, "layout.json"), String))
    ph = only(l for l in L.levels if l.level == "phonemes")
    vocab = Dict{Char, Int}()
    for (i, tok) in enumerate(ph.vocab)
        length(tok) == 1 && (vocab[only(tok)] = i)      # 1-based column within the phoneme block
    end
    return L.columns, ph.first, ph.width, vocab, L.blank + 1
end

"""One clip of a dump as a T×C Float32 matrix (the file is row-major float32)."""
function read_dump_clip(dir, rec, C)
    raw = Vector{Float32}(undef, rec.frames * C)
    read!(joinpath(dir, rec.file), raw)
    return permutedims(reshape(raw, C, rec.frames))     # row-major on disk → T×C
end

# ------------------------------------------------ structural deviations: repetition / skip -----------
# A master may re-read a phrase (iʿādah, allowed) and a learner may skip or restart; forcing such audio
# onto the written text turns the deviation into spurious letter errors. The free (greedy) decode is
# aligned to the reference (Needleman–Wunsch, unit costs); a long insertion block that re-reads an
# earlier stretch ref[s:e] is a *repetition*, a long deletion block a *skip*. The reading path — the
# reference with every repetition inserted where it was read — is what GOP is then run on.

export greedy_decode, nw_ops, deviations, reading_path, read_as_recited

"""Best-path CTC decode: argmax per frame, repeats collapsed, blanks removed (column indices)."""
function greedy_decode(lp::AbstractMatrix{Float32}, blank::Int)
    out = Int[]
    prev = 0
    for t in 1:size(lp, 1)
        k = argmax(@view lp[t, :])
        k != blank && k != prev && push!(out, k)
        prev = k
    end
    return out
end

"""Edit script ref → hyp as (op, ref index, hyp index), op ∈ :eq, :sub, :ins (hyp only), :del (ref only)."""
function nw_ops(ref::AbstractVector, hyp::AbstractVector)
    n, m = length(ref), length(hyp)
    D = zeros(Int, n + 1, m + 1)
    D[:, 1] = 0:n
    D[1, :] = 0:m
    for i in 1:n, j in 1:m
        D[i+1, j+1] = min(D[i, j] + (ref[i] == hyp[j] ? 0 : 1), D[i, j+1] + 1, D[i+1, j] + 1)
    end
    ops = Tuple{Symbol, Int, Int}[]
    i, j = n, m
    while i > 0 || j > 0
        if i > 0 && j > 0 && D[i+1, j+1] == D[i, j] + (ref[i] == hyp[j] ? 0 : 1)
            push!(ops, (ref[i] == hyp[j] ? :eq : :sub, i, j)); i -= 1; j -= 1
        elseif i > 0 && D[i+1, j+1] == D[i, j+1] + 1
            push!(ops, (:del, i, j)); i -= 1
        else
            push!(ops, (:ins, i, j)); j -= 1
        end
    end
    return reverse!(ops)
end

function _editdist(a, b)
    n, m = length(a), length(b)
    prev = collect(0:m)
    cur = similar(prev)
    for i in 1:n
        cur[1] = i
        for j in 1:m
            cur[j+1] = min(prev[j] + (a[i] == b[j] ? 0 : 1), prev[j+1] + 1, cur[j] + 1)
        end
        prev, cur = cur, prev
    end
    return prev[m+1]
end

"""Repetitions and skips of the reading `hyp` against `ref` (both column-index sequences).

Returns NamedTuples (kind = :repetition | :skip, ref_from, ref_to, at) in reading order: a repetition
re-reads ref[ref_from:ref_to] right after ref position `at`; a skip leaves ref[ref_from:ref_to] out.
Blocks shorter than `min_len` symbols are ordinary substitutions/insertions and are left to GOP."""
function deviations(ref::AbstractVector, hyp::AbstractVector; min_len::Int = 6, max_err::Float64 = 0.25)
    ops = nw_ops(ref, hyp)
    out = NamedTuple[]
    k = 1
    while k <= length(ops)
        op = ops[k][1]
        if op in (:ins, :del)
            l = k
            while l < length(ops) && ops[l+1][1] == op
                l += 1
            end
            if l - k + 1 >= min_len
                if op == :ins
                    block = [hyp[ops[x][3]] for x in k:l]
                    at = ops[k][2]                       # inserted after ref[at]
                    # the alignment may put the copy after the stretch it repeats (re-read ref[s:at]) or
                    # before it (ref[at+1:e] read twice); both are the same reading
                    best = (Inf, 0, 0)
                    for s in max(1, at - 3 * length(block)):at
                        e = _editdist(ref[s:at], block) / max(length(block), at - s + 1)
                        e < best[1] && (best = (e, s, at))
                    end
                    for e2 in at+1:min(length(ref), at + 3 * length(block))
                        e = _editdist(ref[at+1:e2], block) / max(length(block), e2 - at)
                        e < best[1] && (best = (e, at + 1, e2))
                    end
                    best[1] <= max_err &&
                        push!(out, (kind = :repetition, ref_from = best[2], ref_to = best[3], at = best[3]))
                else
                    push!(out, (kind = :skip, ref_from = ops[k][2], ref_to = ops[l][2], at = ops[k][2] - 1))
                end
            end
            k = l + 1
        else
            k += 1
        end
    end
    return out
end

"""The reference as actually read: every repetition's stretch inserted after its `at` position.
Returns (path, origin) where origin[p] is the reference index of path symbol p and a negative value
marks a symbol of a repeated copy (−(ref index)). Skips stay in the path: GOP flags them."""
function reading_path(ref::AbstractVector, devs)
    reps = sort([d for d in devs if d.kind == :repetition]; by = d -> d.at)
    path = eltype(ref)[]
    origin = Int[]
    r = 1
    for i in eachindex(ref)
        push!(path, ref[i]); push!(origin, i)
        while r <= length(reps) && reps[r].at == i
            for j in reps[r].ref_from:reps[r].ref_to
                push!(path, ref[j]); push!(origin, -j)
            end
            r += 1
        end
    end
    return path, origin
end

"""Deviations and reading path of one clip, as a phoneme string ready for `gop_sf`.

`ph` is the reference phoneme string. Returns (path string, origin per path character, deviations);
origin[p] > 0 is the reference character index, < 0 marks a character of a repeated copy."""
function read_as_recited(lp::AbstractMatrix{Float32}, ph::AbstractString, vocab::Dict{Char, Int}, blank::Int)
    inv = Dict(v => k for (k, v) in vocab)
    seq = [vocab[c] for c in ph]
    devs = deviations(seq, greedy_decode(lp, blank))
    path, origin = reading_path(seq, devs)
    return String([inv[c] for c in path]), origin, devs
end
