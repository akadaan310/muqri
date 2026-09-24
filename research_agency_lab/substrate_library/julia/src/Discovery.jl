# Model discovery on the reference reciters (included by QaariLab.jl).
#
# 1. Tempo dynamics — does a reciter's harakah drift through a surah, and how fast does it settle?
#    Constant vs linear drift (dT/dτ = φ) vs relaxation ODE (dT/dτ = (T∞ − T)/τc), by AIC per surah;
#    the ODE is solved with OrdinaryDiffEq (Tsit5) and fitted by L-BFGS on a Huber loss with gradients
#    from forward-mode AD through the solver. On Taraweeh imams the same comparison quantifies fatigue
#    (φ > 0: slowing through the surah).
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

sigmoid(u) = 1 / (1 + exp(-u))

"Time constant τc (s) from an unconstrained parameter, bounded to [5 s, 5 × recited time]."
tau_c(u, span) = 5.0 + (5 * max(span, 5.0) - 5.0) * sigmoid(u)

function _solve_tempo(p, τ)
    T0, Tinf, u = p
    span = max(τ[end], 1e-3)
    f(T, q, t) = (q[2] - T) / tau_c(q[3], span)
    prob = ODEProblem(f, T0, (0.0, span), p)
    sol = solve(prob, Tsit5(); saveat = τ, abstol = 1e-6, reltol = 1e-6)
    return sol.u
end

huber(r, δ) = abs(r) <= δ ? 0.5 * r^2 : δ * (abs(r) - 0.5 * δ)

"Gaussian-equivalent AIC from a Huber loss (2·loss plays the role of the RSS)."
aic(loss, n, k) = n * log(max(2 * loss / n, 1e-9)) + 2k

"""
    tempo_dynamics(rows, reciter; min_ayahs = 20)

Per surah with ≥ `min_ayahs` ayahs, three models of the harakah T over recited time τ compete by AIC:

* constant        T = c                                   (k = 1)
* linear drift    dT/dτ = φ                               (k = 2, closed-form robust fit)
* relaxation ODE  dT/dτ = (T∞ − T)/τc, T(0) = T₀           (k = 3; τc bounded to [5 s, 5·span])

The ODE is solved with Tsit5 and fitted by L-BFGS with gradients from forward-mode AD through the
solver. The first formulation (κ(T∞ − T) + φ) was not identifiable — T∞ and φ/κ trade off and the
fit ran to κ → ∞ — so drift and relaxation are now separate hypotheses.
"""
function tempo_dynamics(rows, reciter::String; min_ayahs::Int = 20, δ::Float64 = 15.0)
    series = _tempo_series(rows, reciter)
    fits = Dict{String,Any}()
    for (s, (τ, T)) in series
        n = length(T)
        n >= min_ayahs || continue
        span = max(τ[end], 1e-3)
        med = median(T)
        l_const = sum(huber(t - med, δ) for t in T)
        # linear drift: robust (Huber) line by iteratively reweighted least squares
        X = hcat(ones(n), τ)
        β = X \ T
        for _ in 1:20
            r = T - X * β
            w = [abs(x) <= δ ? 1.0 : δ / abs(x) for x in r]
            β = (X' * (w .* X)) \ (X' * (w .* T))
        end
        l_drift = sum(huber(a - b, δ) for (a, b) in zip(X * β, T))
        loss(p) = sum(huber(a - b, δ) for (a, b) in zip(_solve_tempo(p, τ), T))
        k0 = max(1, n ÷ 5)
        p0 = [median(T[1:k0]), median(T[end-k0+1:end]), 0.0]
        res = try
            # Gradient by forward-mode AD through the Tsit5 solve (dual numbers propagate through the ODE).
            g!(G, p) = ForwardDiff.gradient!(G, loss, p)
            Optim.optimize(loss, g!, p0, Optim.LBFGS(), Optim.Options(iterations = 300))
        catch err
            @warn "tempo fit failed" s err
            continue
        end
        p = Optim.minimizer(res)
        a = Dict("constant" => aic(l_const, n, 1), "drift" => aic(l_drift, n, 2),
                 "relaxation" => aic(Optim.minimum(res), n, 3))
        best = argmin(a)
        fits[string(s)] = Dict("n_ayahs" => n, "recited_s" => span, "median_T_ms" => med,
                               "T0_ms" => p[1], "Tinf_ms" => p[2], "tau_s" => tau_c(p[3], span),
                               "drift_ms_per_min" => 60 * β[2], "aic" => a, "preferred" => best,
                               "delta_aic_best_vs_constant" => a["constant"] - a[best])
    end
    vals(k) = [Float64(f[k]) for f in values(fits)]
    prefs = [f["preferred"] for f in values(fits)]
    relax = [f for f in values(fits) if f["preferred"] == "relaxation"]
    pooled = isempty(fits) ? Dict{String,Any}() : Dict(
        "surahs" => length(fits),
        "share_preferred" => Dict(m => mean(prefs .== m) for m in ("constant", "drift", "relaxation")),
        "median_drift_ms_per_min" => median(vals("drift_ms_per_min")),
        "median_tau_s_where_relaxation" => isempty(relax) ? NaN : median([Float64(f["tau_s"]) for f in relax]),
        "median_T0_minus_Tinf_ms_where_relaxation" =>
            isempty(relax) ? NaN : median([Float64(f["T0_ms"] - f["Tinf_ms"]) for f in relax]))
    return Dict("reciter" => reciter, "models" => ["T = c", "dT/dτ = φ", "dT/dτ = (T∞ − T)/τc"],
                "pooled" => pooled, "per_surah" => fits)
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
