# Identity tests for CtcGop.jl: CTC forward = brute-force path sum, Viterbi = brute-force best path,
# GOP ≈ 0 on text the posteriors support and ≪ 0 on a substituted letter.
#   julia --project=research_agency_lab/substrate_library/julia research_agency_lab/substrate_library/julia/test/test_ctcgop.jl
using Test, Random
using QaariLab

function collapse(path, blank)
    out = Int[]
    prev = 0
    for p in path
        p != blank && p != prev && push!(out, p)
        prev = p
    end
    return out
end

function brute(lp, seq, blank)
    T, V = size(lp)
    tot, best = -Inf, -Inf
    for idx in CartesianIndices(ntuple(_ -> V, T))
        path = collect(Tuple(idx))
        collapse(path, blank) == seq || continue
        s = sum(lp[t, path[t]] for t in 1:T)
        tot = tot == -Inf ? s : max(tot, s) + log1p(exp(-abs(tot - s)))
        best = max(best, s)
    end
    return tot, best
end

function randlp(rng, T, V)
    x = randn(rng, T, V)
    return Float32.(x .- log.(sum(exp.(x); dims = 2)))
end

@testset "CtcGop" begin
    rng = MersenneTwister(7)
    blank = 1
    for (T, seq) in [(4, [2, 3]), (5, [2, 2]), (5, [3, 2, 3]), (6, [2, 3, 3]), (3, [2])]
        lp = randlp(rng, T, 3)
        tot, best = brute(lp, seq, blank)
        @test isapprox(ctc_forward(lp, seq, blank), tot; atol = 1e-4)
        score, f, l = ctc_viterbi(lp, seq, blank)
        @test isapprox(score, best; atol = 1e-4)
        @test all(f .<= l) && issorted(f) && issorted(l)
        @test score <= ctc_forward(lp, seq, blank) + 1e-5
    end

    # A peaked posterior that spells the reference: GOP ≈ 0 everywhere; spell a substitution: GOP ≪ 0 there.
    chars = collect("بَسطَ")
    vocab = Dict('ب' => 2, '\u064e' => 3, 'س' => 4, 'ط' => 5, 'ص' => 6, 'ت' => 7, '\u0650' => 8, '\u064f' => 9)
    function spell(cs)
        T = 3 * length(cs) + 1
        lp = fill(log(0.01f0 / 8), T, 9)
        lp[:, blank] .= log(0.01f0 / 8)
        t = 1
        for c in cs
            lp[t, blank] = log(0.99f0); t += 1
            lp[t, vocab[c]] = log(0.99f0); lp[t+1, vocab[c]] = log(0.99f0); t += 2
        end
        lp[t, blank] = log(0.99f0)
        return lp ./ 1  # already near-normalised; GOP only uses differences
    end
    ok = gop_sf(spell(chars), String(chars), vocab, blank)
    @test all(u.gop > -0.05 for u in ok)
    bad = gop_sf(spell(['ب', '\u064e', 'ص', 'ط', '\u064e']), String(chars), vocab, blank)
    s = only(u for u in bad if u.symbol == 'س')
    @test s.gop < -3 && s.best == "ص"
    @test all(u.gop > -0.05 for u in bad if u.symbol != 'س')

    # structural deviations: a re-read phrase and a skipped stretch are found and typed
    rng2 = MersenneTwister(3)
    ref = rand(rng2, 2:20, 40)
    rep = vcat(ref[1:25], ref[14:25], ref[26:40])
    d = deviations(ref, rep)
    @test length(d) == 1 && d[1].kind == :repetition && d[1].ref_to - d[1].ref_from + 1 == 12
    path, origin = reading_path(ref, d)
    @test path == rep && count(<(0), origin) == 12
    skip = vcat(ref[1:9], ref[19:40])
    d = deviations(ref, skip)
    @test length(d) == 1 && d[1].kind == :skip && (d[1].ref_from, d[1].ref_to) == (10, 18)
    @test isempty(deviations(ref, ref)) && isempty(deviations(ref, vcat(ref[1:5], [21], ref[7:40])))
    lp = fill(log(0.001f0), 3 * length(ref) + 1, 21); lp[:, 1] .= log(0.98f0)
    for (k, c) in enumerate(ref)
        lp[3k-1, c] = log(0.98f0); lp[3k-1, 1] = log(0.001f0)
    end
    @test greedy_decode(lp, 1) == ref   # blanks separate repeated labels, so nothing collapses
end
