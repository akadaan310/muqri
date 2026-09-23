# Model discovery on the reference reciters (included by QaariLab.jl).
#
# 1. Tempo dynamics — does a reciter's harakah drift through a surah, and how fast does it settle?
#       dT/dτ = κ (T∞ − T) + φ,     T(0) = T₀,      τ = recited time (s)
#    solved with OrdinaryDiffEq (Tsit5) and fitted per surah by L-BFGS on a Huber loss, gradients by
#    forward-mode AD through the solver. Compared with the constant-tempo model by AIC. The same model
#    on Taraweeh imams quantifies fatigue (φ > 0: slowing; κ: settling rate).
#
# 2. Duration law — how a held sound's duration relates to its count and the tempo:
#       D = Θ(n, T, final) ξ,   Θ = [1, T, n·T, n, T², n·T², final, final·T]
#    by sequentially-thresholded least squares (STLSQ, the SINDy regression), threshold chosen by BIC.
#    An intercept or a T term that survives means "counts" are not proportional to duration: that
#    constant is the transition overhead that inflates a naive duration/harakah ratio.

using OrdinaryDiffEq
using ForwardDiff

export tempo_dynamics, duration_law

# ------------------------------------------------------------------------ tempo ODE -------------
function _tempo_series(rows, reciter)
    by = Dict{Int,Vector{Tuple{Int,Float64,Float64}}}()
    for r in rows
        r["reciter"] == reciter || continue
        T, d = Float64(something(r["haraka_ms"], NaN)), Float64(something(r["duration_s"], NaN))
        (isfinite(T) && isfinite(d)) || continue
        push!(get!(by, Int(r["surah"]), Tuple{Int,Float64,Float64}[]), (Int(r["ayah"]), T, d))
    end
    out = Dict{Int,Tuple{Vector{Float64},Vector{Float64}}}()
    for (s, v) in by
        sort!(v)
        τ = cumsum([0.0; [x[3] for x in v[1:end-1]]])
        out[s] = (τ, [x[2] for x in v])
    end
    return out
end

function _solve_tempo(p, τ)
    T0, Tinf, logκ, φ = p
    f(T, q, t) = exp(q[3]) * (q[2] - T) + q[4]
    prob = ODEProblem(f, T0, (0.0, max(τ[end], 1e-3)), p)
    sol = solve(prob, Tsit5(); saveat = τ, abstol = 1e-6, reltol = 1e-6)
    return sol.u
end

huber(r, δ) = abs(r) <= δ ? 0.5 * r^2 : δ * (abs(r) - 0.5 * δ)

"""
    tempo_dynamics(rows, reciter; min_ayahs = 20)

Fit the relaxation ODE per surah with ≥ `min_ayahs` ayahs. Returns per-surah parameters, the ΔAIC
against a constant tempo, and pooled medians (time constant 1/κ in seconds, drift φ in ms/s).
"""
function tempo_dynamics(rows, reciter::String; min_ayahs::Int = 20, δ::Float64 = 15.0)
    series = _tempo_series(rows, reciter)
    fits = Dict{String,Any}()
    for (s, (τ, T)) in series
        length(T) >= min_ayahs || continue
        med = median(T)
        loss(p) = sum(huber(a - b, δ) for (a, b) in zip(_solve_tempo(p, τ), T))
        p0 = [T[1], med, log(1 / max(τ[end] / 4, 1.0)), 0.0]
        res = try
            Optim.optimize(loss, p0, Optim.LBFGS(), Optim.Options(iterations = 300); autodiff = :forward)
        catch err
            @warn "tempo fit failed" s err
            continue
        end
        p = Optim.minimizer(res)
        n = length(T)
        const_loss = sum(huber(t - med, δ) for t in T)
        # Huber loss as a pseudo-likelihood: AIC = 2k + 2·loss/δ² (Gaussian-equivalent scale)
        aic_ode = 2 * 4 + 2 * Optim.minimum(res) / δ^2
        aic_const = 2 * 1 + 2 * const_loss / δ^2
        fits[string(s)] = Dict("n_ayahs" => n, "recited_s" => τ[end], "T0_ms" => p[1], "Tinf_ms" => p[2],
                               "tau_s" => 1 / exp(p[3]), "drift_ms_per_s" => p[4],
                               "delta_aic_vs_constant" => aic_const - aic_ode)
    end
    vals(k) = [Float64(f[k]) for f in values(fits)]
    pooled = isempty(fits) ? Dict{String,Any}() : Dict(
        "surahs" => length(fits),
        "median_tau_s" => median(vals("tau_s")), "median_drift_ms_per_s" => median(vals("drift_ms_per_s")),
        "median_T0_minus_Tinf_ms" => median(vals("T0_ms") .- vals("Tinf_ms")),
        "share_ode_preferred" => mean(vals("delta_aic_vs_constant") .> 2))
    return Dict("reciter" => reciter, "model" => "dT/dτ = κ(T∞ − T) + φ", "pooled" => pooled, "per_surah" => fits)
end

# ------------------------------------------------------------------ duration law (STLSQ) ---------
const LIBRARY = ["1", "T", "n·T", "n", "T²", "n·T²", "final", "final·T"]

function _theta(n, T, fin)
    Ts = T / 100  # keep columns on comparable scales (T in 100-ms units)
    hcat(ones(length(n)), Ts, n .* Ts, n, Ts .^ 2, n .* Ts .^ 2, fin, fin .* Ts)
end

function stlsq(Θ, y, λ; iters = 10, ridge = 1e-6)
    ξ = (Θ' * Θ + ridge * I) \ (Θ' * y)
    for _ in 1:iters
        small = abs.(ξ) .< λ
        ξ[small] .= 0
        big = .!small
        any(big) || break
        ξ[big] = (Θ[:, big]' * Θ[:, big] + ridge * I) \ (Θ[:, big]' * y)
    end
    return ξ
end

"""
    duration_law(obs, reciter; metric = "core_ms")

Sparse regression of the held-sound duration (ms) on the library above, over every duration-rule
observation of `reciter` with a textbook count `n` (rule-type midpoint) and local harakah `T`.
The threshold λ is swept and chosen by BIC; the result lists the surviving terms in ms.
"""
function duration_law(obs, reciter::String, expected::Dict{String,Float64}; metric::String = "core_ms")
    n = Float64[]; T = Float64[]; y = Float64[]; fin = Float64[]
    for o in obs
        o.reciter == reciter || continue
        haskey(expected, o.rule_type) || continue
        (haskey(o.metrics, metric) && haskey(o.metrics, "haraka_ms")) || continue
        o.metrics[metric] > 0 || continue
        get(o.metrics, "align_min_ms", Inf) >= 40 || continue
        push!(n, expected[o.rule_type]); push!(T, o.metrics["haraka_ms"]); push!(y, o.metrics[metric])
        push!(fin, o.rule_type == "madd_arid_lissukun" ? 1.0 : 0.0)
    end
    length(y) < 30 && return Dict("reciter" => reciter, "n" => length(y), "note" => "too few observations")
    Θ = _theta(n, T, fin)
    best = nothing
    for λ in (0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0)
        ξ = stlsq(Θ, y, λ)
        rss = sum(abs2, y - Θ * ξ)
        k = count(!iszero, ξ)
        bic = length(y) * log(rss / length(y)) + k * log(length(y))
        (best === nothing || bic < best[3]) && (best = (λ, ξ, bic, rss))
    end
    λ, ξ, bic, rss = best
    r2 = 1 - rss / sum(abs2, y .- mean(y))
    terms = Dict(LIBRARY[i] => ξ[i] for i in eachindex(ξ) if !iszero(ξ[i]))
    # Effective ms per count at the median tempo, and the constant overhead absorbed by a CTC span
    Tm = median(T) / 100
    per_count = get(terms, "n·T", 0.0) * Tm + get(terms, "n", 0.0) + get(terms, "n·T²", 0.0) * Tm^2
    overhead = get(terms, "1", 0.0) + get(terms, "T", 0.0) * Tm + get(terms, "T²", 0.0) * Tm^2
    return Dict("reciter" => reciter, "metric" => metric, "n" => length(y), "lambda" => λ, "bic" => bic,
                "r2" => r2, "terms_ms" => terms, "T_unit" => "100 ms",
                "ms_per_count_at_median_T" => per_count, "overhead_ms_at_median_T" => overhead,
                "median_haraka_ms" => median(T),
                "implied_counts_scale" => per_count > 0 ? median(T) / per_count : NaN)
end
