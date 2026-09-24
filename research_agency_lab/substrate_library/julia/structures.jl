# Formal structures over the graded recitations: learning paths, dependency backbone, discovered and
# tested causal structure.
#
# Input: the per-verse records of the full-dataset engine pass (profiles/clips.jsonl, 11,996 T300
# verses by 41 reciters). Each verse gives, for every located rule, pass or fail; for every
# characteristic head, the share of its letters realised; the reciter's count unit (tempo); and the
# share of words fully correct.
#
# 1. KNOWLEDGE SPACE (Doignon & Falmagne). A skill is a rule. On verses where both a and b are
#    tested, a is a PREREQUISITE of b (a <= b) when b is almost never passed while a is failed
#    (violation rate < tol) and not the reverse. The surmise relation is closed transitively and
#    reduced to its Hasse diagram; its longest-path layers are the learning stages. Per reciter, the
#    KNOWLEDGE STATE is the set of skills held (pass rate >= 0.9) and the OUTER FRINGE is the skills not
#    held whose prerequisites all are -- what that reciter is ready to learn next.
# 2. CHOW-LIU TREE. Every measurement discretised into tertiles; pairwise mutual information; the
#    maximum spanning tree (Kruskal) is the best tree-structured approximation of their joint
#    distribution -- the dependency backbone.
# 3. PC ALGORITHM (Spirtes-Glymour). Skeleton from Fisher-z partial-correlation tests with
#    conditioning sets up to size 2, then v-structures (a -> c <- b) oriented: structure found
#    in the data, not assumed.
# 4. d-SEPARATION. The tradition's causal graph over the same variables implies conditional
#    independencies (Bayes-ball); each is tested with a partial correlation. A refuted independence
#    means the tradition's graph is missing an edge -- a phenomenon deduced, not merely noticed.
#
#   julia --project=research_agency_lab/substrate_library/julia \
#         research_agency_lab/substrate_library/julia/structures.jl CLIPS.jsonl OUT.json

using JSON3, Statistics, LinearAlgebra, Distributions

const FAIL = Set(["short", "long", "wrong"])

# ---------------------------------------------------------------- data
function load_verses(path)
    V = [JSON3.read(l) for l in eachline(path)]
    V = [v for v in V if !haskey(v, :error)]
    rules = sort(unique(String(r[1]) for v in V for r in v.rules if String(r[2]) in FAIL || String(r[2]) == "pass"))
    heads = sort(unique(String(h) for v in V for h in keys(v.heads)))
    return V, rules, heads
end

"""Per verse: pass fraction for each rule present (NaN when absent), realised share per head, tempo,
word accuracy."""
function verse_matrix(V, rules, heads)
    ri = Dict(r => i for (i, r) in enumerate(rules))
    X = fill(NaN, length(V), length(rules) + length(heads) + 2)
    for (n, v) in enumerate(V)
        acc = Dict{Int, Vector{Int}}()
        for r in v.rules
            st = String(r[2]); (st in FAIL || st == "pass") || continue
            i = ri[String(r[1])]
            a = get!(acc, i, [0, 0]); a[1] += st == "pass"; a[2] += 1
        end
        for (i, a) in acc
            X[n, i] = a[1] / a[2]
        end
        for (j, h) in enumerate(heads)
            haskey(v.heads, Symbol(h)) || continue
            t, k = v.heads[Symbol(h)]
            t > 0 && (X[n, length(rules) + j] = k / t)
        end
        v.haraka_s === nothing || (X[n, end-1] = v.haraka_s)
        v.word_accuracy === nothing || (X[n, end] = v.word_accuracy)
    end
    return X, vcat(rules, "sifah:" .* heads, "tempo_haraka_s", "word_accuracy")
end

# ---------------------------------------------------------------- 1. knowledge space
function surmise(V, rules; tol = 0.05, min_support = 150)
    R = length(rules); ri = Dict(r => i for (i, r) in enumerate(rules))
    both = zeros(Int, R, R); viol = zeros(Int, R, R)      # viol[a, b]: b passed while a failed
    for v in V
        st = Dict{Int, Bool}()
        for r in v.rules
            s = String(r[2]); (s in FAIL || s == "pass") || continue
            i = ri[String(r[1])]
            st[i] = get(st, i, true) && s == "pass"          # a verse passes a rule if every instance passes
        end
        ks = collect(keys(st))
        for a in ks, b in ks
            a == b && continue
            both[a, b] += 1
            viol[a, b] += (!st[a] && st[b])
        end
    end
    P = falses(R, R)                                        # P[a, b]: a is a prerequisite of b
    for a in 1:R, b in 1:R
        a == b && continue
        both[a, b] >= min_support || continue
        va, vb = viol[a, b] / both[a, b], viol[b, a] / both[a, b]
        P[a, b] = va < tol && vb >= tol                     # strictly one-directional
    end
    # transitive closure, then transitive reduction -> Hasse diagram
    C = copy(P)
    for k in 1:R, i in 1:R, j in 1:R
        C[i, j] |= C[i, k] & C[k, j]
    end
    H = copy(C)
    for i in 1:R, j in 1:R
        C[i, j] || continue
        any(C[i, k] && C[k, j] for k in 1:R if k != i && k != j) && (H[i, j] = false)
    end
    # learning stages: longest path from the sources
    stage = zeros(Int, R)
    for _ in 1:R, a in 1:R, b in 1:R
        H[a, b] && (stage[b] = max(stage[b], stage[a] + 1))
    end
    return H, stage, both, viol
end

function reciter_states(V, rules, H; held = 0.9, min_n = 5)
    ri = Dict(r => i for (i, r) in enumerate(rules))
    tally = Dict{String, Matrix{Int}}()
    for v in V
        m = get!(tally, String(v.speaker), zeros(Int, length(rules), 2))
        for r in v.rules
            s = String(r[2]); (s in FAIL || s == "pass") || continue
            i = ri[String(r[1])]; m[i, 1] += s == "pass"; m[i, 2] += 1
        end
    end
    out = Dict{String, Any}()
    for (spk, m) in tally
        tested = [i for i in eachindex(rules) if m[i, 2] >= min_n]
        state = Set(i for i in tested if m[i, 1] / m[i, 2] >= held)
        fringe = [i for i in tested if !(i in state) &&
                  all(j in state for j in eachindex(rules) if H[j, i] && j in tested)]
        out[spk] = (state = rules[sort(collect(state))], fringe = rules[fringe],
                    not_ready = rules[[i for i in tested if !(i in state) && !(i in fringe)]])
    end
    return out
end

# ---------------------------------------------------------------- 2. Chow-Liu
"""Three levels by tertile; a variable whose mass sits at its ceiling (a pass fraction that is 1.0 in
most verses) is split at the ceiling instead -- below it / at it -- or its tertiles coincide and every
mutual information collapses to zero."""
function tertiles(x)
    ok = .!isnan.(x); v = x[ok]; q = quantile(v, [1 / 3, 2 / 3])
    if q[1] == q[2]
        top = maximum(v)
        return [isnan(y) ? 0 : (y < top ? 1 : 2) for y in x]
    end
    return [isnan(y) ? 0 : (y <= q[1] ? 1 : y <= q[2] ? 2 : 3) for y in x]
end

function mutual_info(a, b)
    m = (a .> 0) .& (b .> 0); n = sum(m); n < 50 && return 0.0
    J = zeros(3, 3)
    for (x, y) in zip(a[m], b[m])
        J[x, y] += 1
    end
    J ./= n; pa = sum(J, dims = 2); pb = sum(J, dims = 1)
    return sum(J[i, j] > 0 ? J[i, j] * log2(J[i, j] / (pa[i] * pb[j])) : 0.0 for i in 1:3, j in 1:3)
end

function chow_liu(X, names; min_cover = 0.2)
    keep = [j for j in axes(X, 2) if mean(.!isnan.(X[:, j])) >= min_cover && length(unique(filter(!isnan, X[:, j]))) > 2]
    D = [tertiles(X[:, j]) for j in keep]
    E = [(mutual_info(D[a], D[b]), a, b) for a in eachindex(keep) for b in a+1:length(keep)]
    sort!(E, rev = true)
    parent = collect(eachindex(keep)); find(x) = parent[x] == x ? x : (parent[x] = find(parent[x]))
    tree = []
    for (w, a, b) in E
        ra, rb = find(a), find(b)
        ra == rb && continue
        parent[ra] = rb
        push!(tree, (a = names[keep[a]], b = names[keep[b]], mi_bits = round(w; digits = 4)))
    end
    return tree
end

# ---------------------------------------------------------------- 3. PC
"""Fisher-z test of X_i ⊥ X_j | X_S on rows complete for i, j and S."""
function ci_test(X, i, j, S; alpha = 0.001)
    cols = vcat(i, j, S)
    rows = [r for r in axes(X, 1) if all(!isnan, X[r, cols])]
    n = length(rows); n < length(cols) + 30 && return (true, NaN)
    Y = X[rows, cols]
    C = cor(Y)
    any(isnan, C) && return (true, NaN)
    P = pinv(C)
    r = clamp(-P[1, 2] / sqrt(P[1, 1] * P[2, 2]), -0.9999, 0.9999)
    z = 0.5 * log((1 + r) / (1 - r)) * sqrt(n - length(S) - 3)
    p = 2 * (1 - cdf(Normal(), abs(z)))
    return (p > alpha, r)
end

function pc(X, names; maxcond = 2, min_cover = 0.3)
    keep = [j for j in axes(X, 2) if mean(.!isnan.(X[:, j])) >= min_cover]
    K = length(keep)
    adj = trues(K, K); for i in 1:K; adj[i, i] = false; end
    sep = Dict{Tuple{Int, Int}, Vector{Int}}()
    for l in 0:maxcond, i in 1:K, j in i+1:K
        adj[i, j] || continue
        nb = [k for k in 1:K if adj[i, k] && k != j]
        length(nb) >= l || continue
        for S in (l == 0 ? [Int[]] : combos(nb, l))
            indep, _ = ci_test(X, keep[i], keep[j], keep[S])
            if indep
                adj[i, j] = adj[j, i] = false
                sep[(i, j)] = sep[(j, i)] = S
                break
            end
        end
    end
    # v-structures: i - k - j with i, j non-adjacent and k not in their separating set
    arrows = Set{Tuple{Int, Int}}()
    for k in 1:K, i in 1:K, j in i+1:K
        (adj[i, k] && adj[j, k] && !adj[i, j]) || continue
        k in get(sep, (i, j), Int[]) && continue
        push!(arrows, (i, k)); push!(arrows, (j, k))
    end
    edges = []
    for i in 1:K, j in i+1:K
        adj[i, j] || continue
        _, r = ci_test(X, keep[i], keep[j], Int[])
        dir = (i, j) in arrows && !((j, i) in arrows) ? "->" : ((j, i) in arrows && !((i, j) in arrows) ? "<-" : "--")
        push!(edges, (a = names[keep[i]], b = names[keep[j]], dir = dir, r = round(r; digits = 3)))
    end
    return edges
end

combos(v, l) = l == 1 ? [[x] for x in v] : [[v[a], v[b]] for a in eachindex(v) for b in a+1:length(v)]

# ---------------------------------------------------------------- 4. d-separation on the tradition's graph
"""Is x d-separated from y given Z in DAG `pa` (child -> parents)? Reachability (Bayes-ball)."""
function dsep(pa::Dict{String, Vector{String}}, x, y, Z::Set{String})
    ch = Dict{String, Vector{String}}()
    for (c, ps) in pa, p in ps
        push!(get!(ch, p, String[]), c)
    end
    anc = Set{String}(); stack = collect(Z)
    while !isempty(stack)
        n = pop!(stack); n in anc && continue; push!(anc, n)
        append!(stack, get(pa, n, String[]))
    end
    visited = Set{Tuple{String, Symbol}}(); q = [(x, :up)]
    while !isempty(q)
        n, d = pop!(q)
        (n, d) in visited && continue; push!(visited, (n, d))
        n == y && return false
        if d == :up && !(n in Z)
            foreach(p -> push!(q, (p, :up)), get(pa, n, String[]))
            foreach(c -> push!(q, (c, :down)), get(ch, n, String[]))
        elseif d == :down
            if !(n in Z)
                foreach(c -> push!(q, (c, :down)), get(ch, n, String[]))
            end
            n in anc && foreach(p -> push!(q, (p, :up)), get(pa, n, String[]))
        end
    end
    return true
end

"""The tradition's causal graph over the verse-level variables (child => parents). Tempo is the
pace the reciter chooses; the durational rules depend on it (compression), the articulation of each
characteristic on the reciter's articulation, and word accuracy on all of them."""
function tradition_dag(names)
    has(n) = n in names
    pa = Dict{String, Vector{String}}()
    dur = filter(n -> startswith(n, "madd_") || n in ("ghunnah", "ikhfa", "idgham_ghunnah", "iqlab", "ikhfa_shafawi", "idgham_shafawi"), names)
    sif = filter(n -> startswith(n, "sifah:"), names)
    for d in dur
        pa[d] = has("tempo_haraka_s") ? ["tempo_haraka_s"] : String[]
    end
    for s in sif
        pa[s] = String[]                     # articulation: exogenous at the verse level
    end
    pa["word_accuracy"] = vcat(dur, sif)
    pa["tempo_haraka_s"] = String[]
    return pa
end

"""Each variable minus its reciter's own mean: what varies verse to verse inside one voice."""
function within(X, groups)
    Y = copy(X)
    for g in unique(groups)
        m = groups .== g
        for j in axes(X, 2)
            v = X[m, j]; ok = .!isnan.(v)
            any(ok) && (Y[m, j] = v .- mean(v[ok]))
        end
    end
    return Y
end

"""Two-way within: remove both the reciter's and the verse's mean (alternating projections), so
neither the voice nor the verse's text can carry a dependence."""
function within2(X, g1, g2; iters = 15)
    Y = copy(X)
    for _ in 1:iters
        Y = within(Y, g1)
        Y = within(Y, g2)
    end
    return Y
end

function test_implications(X, names, pa; max_tests = 400, Xw = nothing, Xw2 = nothing)
    idx = Dict(n => i for (i, n) in enumerate(names))
    vars = [n for n in keys(pa) if haskey(idx, n) && mean(.!isnan.(X[:, idx[n]])) >= 0.3]
    out = []
    for a in eachindex(vars), b in a+1:length(vars)
        x, y = vars[a], vars[b]
        # the implied independence given the parents of both (a valid adjustment for non-adjacent pairs)
        Z = Set(filter(v -> haskey(idx, v) && v in vars, union(get(pa, x, String[]), get(pa, y, String[]))))
        (x in Z || y in Z) && continue
        dsep(pa, x, y, Z) || continue
        indep, r = ci_test(X, idx[x], idx[y], [idx[z] for z in Z])
        # a refuted independence: does the dependence survive inside one voice?
        wi, wr = Xw === nothing || indep ? (true, NaN) : ci_test(Xw, idx[x], idx[y], [idx[z] for z in Z])
        vi, vr = Xw2 === nothing || indep ? (true, NaN) : ci_test(Xw2, idx[x], idx[y], [idx[z] for z in Z])
        cause = indep ? nothing :
            wi ? "the reciter (voice / recording / style: a trait)" :
            vi ? "the verse's text (its letters, for every reciter)" :
                 "the reciter's performance in that verse (a state)"
        push!(out, (x = x, y = y, given = collect(Z), implied = "independent", holds = indep,
                    partial_r = isnan(r) ? nothing : round(r; digits = 3),
                    within_reciter_r = isnan(wr) ? nothing : round(wr; digits = 3),
                    within_reciter_and_verse_r = isnan(vr) ? nothing : round(vr; digits = 3),
                    common_cause = cause))
        length(out) >= max_tests && break
    end
    return out
end

# ---------------------------------------------------------------- main
function main(path, out)
    V, rules, heads = load_verses(path)
    X, names = verse_matrix(V, rules, heads)
    println("$(length(V)) verses, $(length(rules)) rules, $(length(heads)) heads -> $(size(X, 2)) variables")

    H, stage, both, viol = surmise(V, rules)
    println("\nKNOWLEDGE SPACE: prerequisite (Hasse) edges, a -> b = 'b is almost never held without a'")
    for a in eachindex(rules), b in eachindex(rules)
        H[a, b] && println("   ", rpad(rules[a], 24), " -> ", rpad(rules[b], 24),
                           " violation ", round(viol[a, b] / both[a, b]; digits = 3), "  vs reverse ",
                           round(viol[b, a] / both[a, b]; digits = 3), "  n=", both[a, b])
    end
    order = sortperm(stage)
    println("LEARNING STAGES (longest-path layer):")
    for s in sort(unique(stage))
        println("   stage $s: ", join(rules[stage .== s], ", "))
    end
    states = reciter_states(V, rules, H)
    println("\nPER RECITER: ready to learn next (outer fringe)")
    for spk in sort(collect(keys(states)))[1:min(8, end)]
        println("   ", rpad(spk, 36), join(states[spk].fringe, ", "))
    end

    tree = chow_liu(X, names)
    println("\nCHOW-LIU BACKBONE (strongest dependencies of the maximum-MI spanning tree)")
    for e in sort(tree, by = e -> -e.mi_bits)[1:min(15, end)]
        println("   ", rpad(e.a, 26), " — ", rpad(e.b, 26), e.mi_bits, " bits")
    end

    edges = pc(X, names)
    println("\nPC SKELETON ($(length(edges)) edges; oriented where a v-structure was found)")
    for e in sort(edges, by = e -> -abs(e.r))[1:min(20, end)]
        println("   ", rpad(e.a, 26), " ", e.dir, " ", rpad(e.b, 26), " r=", e.r)
    end

    pa = tradition_dag(names)
    spkv = [String(v.speaker) for v in V]; verse = ["$(v.surah):$(v.ayah)" for v in V]
    imp = test_implications(X, names, pa; Xw = within(X, spkv), Xw2 = within2(X, spkv, verse))
    bad = filter(x -> !x.holds, imp)
    println("\nd-SEPARATION: $(length(imp)) independencies implied by the tradition's graph; $(length(bad)) refuted by the data")
    for x in sort(bad, by = x -> -abs(something(x.partial_r, 0)))[1:min(12, end)]
        println("   ", rpad(x.x, 24), " ⊥ ", rpad(x.y, 24), " | ", rpad(join(x.given, ","), 16), " r=", rpad(x.partial_r, 7),
                " | reciter removed ", rpad(something(x.within_reciter_r, "—"), 7),
                " | + verse removed ", rpad(something(x.within_reciter_and_verse_r, "—"), 7), " -> ", x.common_cause)
    end

    # the latent precision state: the leading factor of the characteristics' two-way residuals (the
    # dependence the tradition's graph could not explain), scored per verse; per reciter its mean is a
    # level and its spread across verses a STEADINESS
    Xw2 = within2(X, spkv, verse)
    hcols = [j for (j, n) in enumerate(names) if startswith(n, "sifah:") && n != "sifah:tikraar"]
    rows = [r for r in axes(Xw2, 1) if all(!isnan, Xw2[r, hcols])]
    Z = Xw2[rows, hcols]; Z ./= max.(1e-9, std(Z, dims = 1))
    F = eigen(Symmetric(cov(Z)))
    w = F.vectors[:, end]; w .*= sign(sum(w))                    # higher = more fully realised
    explained = F.values[end] / sum(F.values)
    score = Z * w
    lvl = Dict{String, Vector{Float64}}()
    for (k, r) in enumerate(rows)
        push!(get!(lvl, spkv[r], Float64[]), score[k])
    end
    tier = Dict(String(v.speaker) => "" for v in V)
    println("\nPRECISION STATE: leading factor of the residual characteristics explains ",
            round(100 * explained; digits = 1), " % of their joint variance; loadings:")
    for (j, c) in enumerate(hcols)
        println("   ", rpad(names[c], 30), round(w[j]; digits = 3))
    end
    steadiness = sort([(spk = k, verses = length(v), spread = std(v)) for (k, v) in lvl if length(v) >= 50],
                      by = x -> x.spread)
    println("STEADINESS (spread of the verse-level precision state; lower = steadier)")
    for x in vcat(steadiness[1:min(6, end)], steadiness[max(1, end - 5):end])
        println("   ", rpad(x.spk, 40), round(x.spread; digits = 3), "   over ", x.verses, " verses")
    end

    open(io -> JSON3.write(io, (variables = names, rules = rules,
        precision_state = (loadings = Dict(names[c] => w[j] for (j, c) in enumerate(hcols)),
                           explained = explained, steadiness = steadiness),
        prerequisites = [(a = rules[a], b = rules[b], violation = round(viol[a, b] / both[a, b]; digits = 4),
                          reverse = round(viol[b, a] / both[a, b]; digits = 4), n = both[a, b])
                         for a in eachindex(rules), b in eachindex(rules) if H[a, b]],
        stages = Dict(rules[i] => stage[i] for i in eachindex(rules)),
        reciter_states = states, chow_liu = tree, pc = edges, implications = imp)), out, "w")
    println("\n-> $out")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS...)
end
