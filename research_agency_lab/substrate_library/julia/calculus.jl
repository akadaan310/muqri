# The calculus of characteristics: how the sifat combine, and what they produce together.
#
# The classical sifat are sets of letters, and tajweed is full of derived sets: qalqalah is the letters
# that are BOTH jahr (the sound is held) and shadid (the air is held) -- ب ج د ط ق -- with the one
# exception the tradition names, the hamza, the "stopping" letter that is jahr and shadid yet does not
# bounce. A certified reviewer also recognised a characteristic no rule names: on a doubled or sakin
# ب (and د) the sound keeps flowing and builds against the closure before the release -- the model's
# istitalah head fires there although istitalah "belongs" to ض. Latent characteristics like that are
# what a combination of other characteristics naturally produces.
#
# This script turns both ideas into computation over every consonant the 41 T300 reciters produced
# (research_agency_lab/experiments/calculus/extract.py):
#
#   1. CANON    letter x class table of what the phonetizer (the books) expects, per context
#   2. LATTICE  Formal Concept Analysis (Ganter's NextClosure) over letters x inherent classes: every
#               closed group of letters and the characteristics exactly they share -- the groups and
#               subgroups of the sifat, derived, not listed; and the qalqalah check
#   3. RESIDUAL where a head hears a class the text does not expect, per letter x context, per tier
#               (anchor / studio / imam / fast): consistent among masters => candidate phenomenon;
#               rising toward the fast tier => candidate mistake
#   4. RULES    which combinations of canonical classes and context raise an off-target class:
#               support, lift over its base rate, and mutual information
#   5. COUPLING head-by-head partial correlation of the "marked" probabilities after removing what
#               the letter and its context explain -- how characteristics act on each other
#
#   julia --project=research_agency_lab/substrate_library/julia \
#         research_agency_lab/substrate_library/julia/calculus.jl DATA_DIR OUT.json

using JSON3, Statistics, LinearAlgebra

const CONTEXTS = Dict(0 => "sakin", 1 => "fatha", 2 => "damma", 3 => "kasra", 4 => "madd", 5 => "end")

function load(dir)
    h = JSON3.read(read(joinpath(dir, "header.json"), String))
    n = h.n
    rd(f, T, cols) = (x = Vector{T}(undef, n * cols); read!(joinpath(dir, f), x); permutedims(reshape(x, cols, n)))
    nclass = sum(length(c) for c in h.head_classes)
    return (h = h, meta = rd("meta.i32", Int32, 6), probs = rd("probs.f32", Float32, nclass),
            expect = rd("expect.i32", Int32, length(h.heads)), speaker = vec(rd("speaker.i32", Int32, 1)))
end

# column range of each head inside the 22 class probabilities
function head_cols(h)
    cols, c = UnitRange{Int}[], 1
    for cl in h.head_classes
        push!(cols, c:c+length(cl)-1)
        c += length(cl)
    end
    return cols
end

context(meta, i) = meta[i, 2] >= 2 ? "shaddah" : (meta[i, 5] == 1 ? "stop" : CONTEXTS[meta[i, 3]])

# ---------------------------------------------------------------- 1. canon
"""For each letter and head, the distribution of the class the phonetizer expects (all contexts)."""
function canon(D)
    L, H = length(D.h.letters), length(D.h.heads)
    counts = [zeros(Int, length(D.h.head_classes[k])) for _ in 1:L, k in 1:H]
    for i in 1:size(D.meta, 1), k in 1:H
        e = D.expect[i, k]
        e >= 0 && (counts[D.meta[i, 1] + 1, k][e + 1] += 1)
    end
    return counts
end

# ---------------------------------------------------------------- 2. lattice
"""Formal context: letter possesses head-class a if the books expect it there in >= `share` of the
letter's occurrences (context-dependent classes -- a ر that is heavy here and light there -- carry
both)."""
function formal_context(D, counts; share = 0.05)
    attrs = [(k, c) for k in eachindex(D.h.heads) for c in eachindex(D.h.head_classes[k])]
    L = length(D.h.letters)
    I = falses(L, length(attrs))
    for l in 1:L, (j, (k, c)) in enumerate(attrs)
        tot = sum(counts[l, k])
        tot > 0 && (I[l, j] = counts[l, k][c] / tot >= share)
    end
    return I, attrs
end

extent(I, B) = [l for l in 1:size(I, 1) if all(I[l, j] for j in B)]
intent(I, A) = [j for j in 1:size(I, 2) if all(I[l, j] for l in A)]
closure(I, B) = intent(I, extent(I, B))

"""All formal concepts by Ganter's NextClosure (lectic order over attributes)."""
function concepts(I)
    m = size(I, 2)
    out = Tuple{Vector{Int}, Vector{Int}}[]
    B = closure(I, Int[])
    while true
        push!(out, (extent(I, B), B))
        length(B) == m && break
        found = false
        for j in m:-1:1
            j in B && continue
            C = closure(I, vcat(filter(<(j), B), j))
            if all(x -> x >= j || x in B, C) && all(x -> x in C, filter(<(j), B))
                B, found = sort(C), true
                break
            end
        end
        found || break
    end
    return out
end

# ---------------------------------------------------------------- 3. residual
function residuals(D; min_n = 40)
    cols = head_cols(D.h)
    tiers = D.h.tiers
    rows = Dict{Tuple{Int, String, Int, Int, Int}, Vector{Int}}()     # (letter, ctx, head, expected, heard) -> per-speaker hits
    tot = Dict{Tuple{Int, String, Int, Int}, Vector{Int}}()           # (letter, ctx, head, expected) -> per-speaker n
    S = length(D.h.speakers)
    for i in 1:size(D.meta, 1), k in eachindex(cols)
        e = D.expect[i, k]
        e < 0 && continue
        heard = argmax(@view D.probs[i, cols[k]]) - 1
        key = (Int(D.meta[i, 1]), context(D.meta, i), k, Int(e))
        v = get!(tot, key, zeros(Int, S))
        v[D.speaker[i] + 1] += 1
        heard == e && continue
        w = get!(rows, (key..., heard), zeros(Int, S))
        w[D.speaker[i] + 1] += 1
    end
    out = []
    for ((l, ctx, k, e, heard), hits) in rows
        n = tot[(l, ctx, k, e)]
        sum(n) >= min_n || continue
        rate(sel) = (a = sum(hits[sel]); b = sum(n[sel]); b > 0 ? a / b : NaN)
        bytier = Dict(t => rate([tiers[s] == t for s in 1:S]) for t in ("anchor", "studio", "imam", "fast"))
        per = [n[s] >= 5 ? hits[s] / n[s] : NaN for s in 1:S]
        good = filter(!isnan, per)
        masters = [per[s] for s in 1:S if tiers[s] in ("anchor", "studio") && !isnan(per[s])]
        push!(out, (letter = string(D.h.letters[l + 1]), context = ctx, head = D.h.heads[k],
                    expected = D.h.head_classes[k][e + 1], heard = D.h.head_classes[k][heard + 1],
                    n = sum(n), rate = sum(hits) / sum(n), by_tier = bytier,
                    masters_rate = isempty(masters) ? NaN : median(masters),
                    reciters_over_30pct = count(>(0.3), good), reciters_measured = length(good)))
    end
    sort!(out, by = r -> -r.rate * sqrt(r.n))
    return out
end

"""Phenomenon or mistake? A pattern held by the masters and flat or falling toward the fast tier
looks like a characteristic of recitation; one that rises toward the fast tier looks like a slip."""
function classify(r)
    m, f = r.by_tier["anchor"], r.by_tier["fast"]
    (isnan(m) || isnan(f)) && return "undetermined"
    r.masters_rate >= 0.3 && f <= 1.3 * m && return "phenomenon-like"
    f >= 1.5 * max(m, 0.02) && return "mistake-like"
    return "mixed"
end

# ---------------------------------------------------------------- 4. rules
"""Which combinations of the letter's canonical classes and its context raise a target event.
Features: the letter's inherent head-classes (from the formal context) + its context. Rules of up to
three features, with support, confidence, lift and the mutual information of the event with the
conjunction."""
function rules(D, I, attrs, target::Function; min_support = 150, maxlen = 3, top = 25)
    N = size(D.meta, 1)
    y = [target(i) for i in 1:N]
    base = mean(y)
    names = [string(D.h.heads[k], "=", D.h.head_classes[k][c]) for (k, c) in attrs]
    ctxs = sort(unique(context(D.meta, i) for i in 1:N))
    F = hcat([Bool[I[D.meta[i, 1] + 1, j] for i in 1:N] for j in eachindex(attrs)]...,
             [Bool[context(D.meta, i) == c for i in 1:N] for c in ctxs]...)
    fn = vcat(names, "ctx=" .* ctxs)
    keep = [j for j in 1:size(F, 2) if 0 < sum(F[:, j]) < N]
    H(p) = p <= 0 || p >= 1 ? 0.0 : -(p * log2(p) + (1 - p) * log2(1 - p))
    found = []
    function visit(sel, mask, start)
        length(sel) >= 1 && begin
            s = sum(mask)
            s >= min_support || return
            conf = mean(y[mask])
            lift = conf / base
            # I(Y; X) for the binary indicator X = the conjunction
            px = s / N
            mi = H(base) - (px * H(conf) + (1 - px) * H(sum(y[.!mask]) / max(1, N - s)))
            push!(found, (features = fn[sel], support = s, confidence = conf, lift = lift, mi_bits = mi))
        end
        length(sel) == maxlen && return
        for jj in start:length(keep)
            j = keep[jj]
            m2 = mask .& F[:, j]
            sum(m2) >= min_support || continue
            visit(vcat(sel, j), m2, jj + 1)
        end
    end
    visit(Int[], trues(N), 1)
    # a longer rule is only interesting if it lifts beyond each of its parts
    sort!(found, by = r -> -r.mi_bits)
    return (base_rate = base, rules = found[1:min(top, length(found))])
end

# ---------------------------------------------------------------- 5. coupling
"""Partial correlation between the heads' marked-class probabilities after removing each
(letter, context) cell's mean: what moves together beyond what the letter itself dictates."""
function coupling(D)
    cols = head_cols(D.h)
    X = hcat([D.probs[:, first(c)] for c in cols]...)        # the first class of each head = its marked class
    cell = [(D.meta[i, 1], context(D.meta, i)) for i in 1:size(X, 1)]
    R = similar(X, Float64)
    for c in unique(cell)
        idx = findall(==(c), cell)
        R[idx, :] .= X[idx, :] .- mean(X[idx, :], dims = 1)
    end
    C = cor(R)
    P = inv(C + 1e-6I)
    partial = [-P[a, b] / sqrt(P[a, a] * P[b, b]) for a in axes(P, 1), b in axes(P, 2)]
    return (heads = D.h.heads, marked = [D.h.head_classes[k][1] for k in eachindex(cols)],
            correlation = C, partial = partial)
end

function main(dir, out)
    D = load(dir)
    println("$(size(D.meta, 1)) consonant units, $(length(D.h.speakers)) reciters")
    counts = canon(D)
    I, attrs = formal_context(D, counts)
    aname((k, c)) = string(D.h.heads[k], "=", D.h.head_classes[k][c])
    C = concepts(I)
    println("formal context $(size(I)), $(length(C)) concepts")

    # the qalqalah theorem: the letters that are jahr AND shadid, against the qalqalah letters
    jid = findfirst(==(("hams_or_jahr", "[جهر]")), [(D.h.heads[k], D.h.head_classes[k][c]) for (k, c) in attrs])
    sid = findfirst(==(("shidda_or_rakhawa", "[شديد]")), [(D.h.heads[k], D.h.head_classes[k][c]) for (k, c) in attrs])
    qid = findfirst(==(("qalqla", "[مقلقل]")), [(D.h.heads[k], D.h.head_classes[k][c]) for (k, c) in attrs])
    L(ls) = join(string.(D.h.letters[ls]))
    js = extent(I, [jid, sid]); qs = extent(I, [qid])
    println("jahr ∩ shadid = {", L(js), "}   qalqalah = {", L(qs), "}   difference = {", L(setdiff(js, qs)), "}")

    res = residuals(D)
    println("\nOFF-TARGET CHARACTERISTICS (top 30 by rate x sqrt n)")
    for r in res[1:min(30, length(res))]
        println(rpad("$(r.letter) $(r.context)", 14), rpad("$(r.head): $(r.expected)->$(r.heard)", 58),
                "n=", rpad(r.n, 6), "rate=", rpad(round(r.rate; digits = 3), 7),
                "anchor=", rpad(round(r.by_tier["anchor"]; digits = 3), 7), "fast=", rpad(round(r.by_tier["fast"]; digits = 3), 7),
                classify(r))
    end

    cols = head_cols(D.h)
    ist = findfirst(==("istitala"), D.h.heads)
    taf = findfirst(==("tafashie"), D.h.heads)
    dad = findfirst(==('ض'), collect(D.h.letters)) - 1
    shin = findfirst(==('ش'), collect(D.h.letters)) - 1
    hold = rules(D, I, attrs, i -> D.meta[i, 1] != dad && argmax(@view D.probs[i, cols[ist]]) == 1)
    spread = rules(D, I, attrs, i -> D.meta[i, 1] != shin && argmax(@view D.probs[i, cols[taf]]) == 1)
    for (name, r) in (("istitalah heard off ض (the voiced hold)", hold), ("tafashshi heard off ش", spread))
        println("\nRULES: ", name, "   base rate ", round(r.base_rate; digits = 4))
        for x in r.rules[1:min(10, length(r.rules))]
            println("  ", rpad(join(x.features, " ∧ "), 70), " support=", rpad(x.support, 7),
                    " conf=", rpad(round(x.confidence; digits = 3), 6), " lift=", rpad(round(x.lift; digits = 2), 6),
                    " MI=", round(x.mi_bits; digits = 4))
        end
    end

    cp = coupling(D)
    println("\nCOUPLING (partial correlation beyond letter x context), strongest pairs")
    pairs = [(cp.partial[a, b], a, b) for a in axes(cp.partial, 1) for b in axes(cp.partial, 2) if a < b]
    for (v, a, b) in sort(pairs, by = x -> -abs(x[1]))[1:10]
        println("  ", rpad("$(cp.heads[a]) $(cp.marked[a])  ~  $(cp.heads[b]) $(cp.marked[b])", 70), round(v; digits = 3))
    end

    open(out, "w") do io
        JSON3.write(io, (n = size(D.meta, 1), attributes = aname.(attrs),
                         letters = string.(collect(D.h.letters)),
                         formal_context = [findall(I[l, :]) for l in axes(I, 1)],
                         concepts = [(extent = L(e), intent = aname.(attrs[b])) for (e, b) in C],
                         qalqalah_theorem = (jahr_and_shadid = L(js), qalqalah = L(qs), exception = L(setdiff(js, qs))),
                         residuals = [merge(r, (verdict = classify(r),)) for r in res],
                         rules = (voiced_hold = hold, tafashshi_off_shin = spread),
                         coupling = cp))
    end
    println("\n-> $out")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS...)
end
