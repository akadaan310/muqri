# The learner-population prior, measured: for every skill, the mean and spread of learners' pass
# log-odds, fitted on real learner recordings instead of assumed.
#
# The serving prior (app/learner.py) is the 41-professional cohort widened by an assumed floor
# (LEARNER_FLOOR_SD = 1.0 log-odds) with the professionals' MEAN kept. Here, for each skill s, each
# learner i contributes (k_i passes of n_i graded instances) and
#
#     k_i ~ Binomial(n_i, sigmoid(theta_i)),   theta_i ~ Normal(mu_s, tau_s^2),
#
# fitted by maximum marginal likelihood (theta integrated out with Gauss-Hermite quadrature). k and n
# are divided by the cohort's overdispersion phi first: instances inside one verse fail together, so
# n instances carry n / phi instances' worth of evidence -- the same discount the posterior applies.
#
#   ~/julia-1.11.5/bin/julia --project=. learner_prior.jl IN.json OUT.json
#
# IN:  {"phi": 1.3, "min_learners": 20, "learners": {"<id>": {"<skill>": [k, n], ...}, ...}}
# OUT: {"phi": .., "skills": {"<skill>": {"mu", "tau", "learners", "instances", "loglik"}}}

using JSON3, Optim, ForwardDiff, LinearAlgebra

"Gauss-Hermite nodes and weights (weight exp(-x^2)) by Golub-Welsch."
function gauss_hermite(m::Int)
    J = SymTridiagonal(zeros(m), [sqrt(i / 2) for i in 1:m-1])
    F = eigen(J)
    return F.values, sqrt(pi) .* F.vectors[1, :] .^ 2
end

const GH_X, GH_W = gauss_hermite(40)

log_sig(x) = -log1p(exp(-x))          # log sigmoid(x), stable for the x in play (|x| < 40)
log1m_sig(x) = -log1p(exp(x))

"Marginal log-likelihood of (mu, log tau) for one skill's learners."
function marginal_loglik(p, k::Vector{Float64}, n::Vector{Float64})
    mu, tau = p[1], exp(p[2])
    ll = zero(eltype(p))
    for i in eachindex(k)
        # log sum_j w_j / sqrt(pi) * L_i(mu + sqrt(2) tau x_j), in log space
        terms = [log(GH_W[j] / sqrt(pi)) + k[i] * log_sig(mu + sqrt(2) * tau * GH_X[j]) +
                 (n[i] - k[i]) * log1m_sig(mu + sqrt(2) * tau * GH_X[j]) for j in eachindex(GH_X)]
        mx = maximum(terms)
        ll += mx + log(sum(exp.(terms .- mx)))
    end
    return ll
end

function fit_skill(k, n)
    p0 = [log((sum(k) + 0.5) / (sum(n) - sum(k) + 0.5)), log(1.0)]
    obj = p -> -marginal_loglik(p, k, n)
    r = optimize(obj, p0, LBFGS(); autodiff = Optim.ADTypes.AutoForwardDiff())
    p = Optim.minimizer(r)
    return p[1], exp(p[2]), -Optim.minimum(r)
end

function main(inp, outp)
    d = JSON3.read(read(inp, String))
    phi = Float64(d.phi)
    minl = Int(get(d, :min_learners, 20))
    by = Dict{String, Tuple{Vector{Float64}, Vector{Float64}}}()
    for (_, t) in pairs(d.learners), (s, kn) in pairs(t)
        kn[2] > 0 || continue
        ks, ns = get!(by, String(s), (Float64[], Float64[]))
        push!(ks, kn[1] / phi)
        push!(ns, kn[2] / phi)
    end
    out = Dict{String, Any}()
    for s in sort(collect(keys(by)))
        k, n = by[s]
        length(k) >= minl || continue
        mu, tau, ll = fit_skill(k, n)
        out[s] = Dict("mu" => round(mu; digits = 5), "tau" => round(tau; digits = 5),
                      "learners" => length(k), "instances" => round(Int, sum(n) * phi),
                      "loglik" => round(ll; digits = 3))
    end
    open(outp, "w") do io
        JSON3.write(io, Dict("phi" => phi, "min_learners" => minl, "skills" => out))
    end
    println("fitted $(length(out)) skills -> $outp")
end

main(ARGS[1], ARGS[2])
