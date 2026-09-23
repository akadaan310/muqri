# Write fixtures + Julia results for the Octave cross-check of the frontier layer.
#   julia --project=research_agency_lab/substrate_library/julia \
#         research_agency_lab/substrate_library/julia/frontier_crosscheck.jl DIR
# then:  octave-cli --path research_agency_lab/substrate_library/octave --eval "fr_crosscheck('DIR')"
# Optional: DIR/clouds_*.csv (real per-key clouds exported by frontier.jl) are also recomputed.
using LinearAlgebra, Random
include(joinpath(@__DIR__, "src", "QaariLab.jl"))
using .QaariLab
const Q = QaariLab

dir = ARGS[1]; mkpath(dir)
wcsv(name, A) = open(joinpath(dir, name * ".csv"), "w") do io
    A = A isa AbstractVector ? reshape(A, :, 1) : A
    for i in 1:size(A, 1)
        println(io, join((repr(Float64(x)) for x in A[i, :]), ","))
    end
end
rng = MersenneTwister(20260923)
randspd(d) = (A = randn(rng, d, d); Matrix(Symmetric(A * A' + 0.1I)))

X = vcat(randn(rng, 400, 3) * [2.0 0 0; 0.5 1 0; 0 0.3 0.3] .+ [5.0 -1 2], 30 .+ 5randn(rng, 20, 3))
wcsv("X", X); μ, Σ = Q.ogk(X); wcsv("ogk_mu", μ); wcsv("ogk_S", Σ)

k, d = 5, 4
Cs = [randspd(d) for _ in 1:k]; ms = [randn(rng, d) for _ in 1:k]; w = rand(rng, k)
for i in 1:k; wcsv("C$i", Cs[i]); end
wcsv("M", reduce(hcat, ms)); wcsv("w", w)
m̄, C̄, _ = Q.bw_barycenter(ms, Cs, w); wcsv("bary_m", m̄); wcsv("bary_C", C̄)
loc, sh, tot = Q.w2_gauss(ms[1], Cs[1], ms[2], Cs[2])
kar = Q.karcher_mean(Cs, w); wcsv("karcher", kar)
sc = rand(rng, 11)
wcsv("scores", sc)
Mg = [abs(i - j) / 4 for i in 1:5, j in 1:5]; a = normalize(rand(rng, 5), 1); b = normalize(rand(rng, 5), 1)
wcsv("ground", Mg); wcsv("hist_a", a); wcsv("hist_b", b)
wcsv("scalars", [loc, sh, tot, Q.airm_dist(Cs[1], Cs[2]), Q.conformal_threshold(sc, 0.2)[1],
                 Q.conformal_threshold(sc, 0.05)[1], Q.sinkhorn_divergence(a, b, Mg)])

# EVT / GPD tail (A6): a heavy-tailed sample, its fitted shape/scale and two POT cuts
gpd_q(u, ξ, σ) = (σ / ξ) * ((1 - u)^(-ξ) - 1)
tail = [gpd_q(rand(rng), 0.28, 1.7) for _ in 1:3_000]
wcsv("tail", tail)
ξt, σt = Q.gpd_fit(tail)
qa, _, _, ua, nua = Q.pot_threshold(tail, 1e-2)
qb, _, _, _, _ = Q.pot_threshold(tail, 1e-3)
wcsv("evt", [ξt, σt, qa, qb, ua, Float64(nua)])
println("fixtures written to ", dir)
