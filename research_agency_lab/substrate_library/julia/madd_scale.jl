# Fit the absolute count scale, so a duration verdict can finally be stated in counts.
#
# Every durational verdict has shipped flagged "unvalidated" because the measured scale did not match
# the notation: anchors read madd_2 = 2.25, madd_4 = 6.20, madd_6 = 12.0 own-counts against a nominal
# 2 : 4 : 6. The ratios are wrong, not just the offset, so no single multiplier fixes it — which is
# exactly why it needed fitting rather than guessing.
#
# The rule-instance join supplies the data: the parser states what each located madd requires and the
# engine measures what that instance actually got, over thousands of instances on the anchors. We fit
#
#     measured ≈ a + b · nominal
#
# by a robust (median-based, Theil–Sen) regression, because a handful of mis-segmented instances must
# not drag the line. Theil–Sen takes the median of all pairwise slopes and has a 29 % breakdown point,
# against 0 % for least squares.
#
# The inverse, nominal ≈ (measured − a) / b, is what turns a measured duration into "you gave 2.1
# counts where 4 are required".
#
#   julia --project=research_agency_lab/substrate_library/julia \
#         research_agency_lab/substrate_library/julia/madd_scale.jl PAIRS.jsonl OUT.json

using JSON3, Statistics, Random

"Theil–Sen slope and intercept: the median of pairwise slopes, robust to 29 % contamination."
function theil_sen(x::Vector{Float64}, y::Vector{Float64}; max_pairs::Int = 200_000)
    n = length(x)
    n >= 2 || return (NaN, NaN)
    slopes = Float64[]
    rng = MersenneTwister(20260923)
    total = n * (n - 1) ÷ 2
    if total <= max_pairs
        for i in 1:n-1, j in i+1:n
            x[j] != x[i] && push!(slopes, (y[j] - y[i]) / (x[j] - x[i]))
        end
    else                                    # subsample pairs when the corpus is large
        for _ in 1:max_pairs
            i, j = rand(rng, 1:n), rand(rng, 1:n)
            x[j] != x[i] && push!(slopes, (y[j] - y[i]) / (x[j] - x[i]))
        end
    end
    isempty(slopes) && return (NaN, NaN)
    b = median(slopes)
    a = median(y .- b .* x)
    return (a, b)
end

function main(inp, out)
    rows = [JSON3.read(l) for l in eachline(inp)]
    x = Float64[r.nominal for r in rows]
    y = Float64[r.measured for r in rows]
    keep = isfinite.(x) .& isfinite.(y) .& (y .> 0) .& (y .< 60)
    x, y = x[keep], y[keep]
    println("instances: ", length(x), " (", count(.!keep), " dropped as non-finite or absurd)")

    a, b = theil_sen(x, y)
    println("\nGLOBAL FIT   measured ≈ ", round(a; digits = 3), " + ", round(b; digits = 3), " · nominal")
    println("             inverse  nominal ≈ (measured − ", round(a; digits = 3), ") / ", round(b; digits = 3))

    # per nominal level, so the fit can be checked where it matters rather than only on average
    println("\n", rpad("nominal", 9), rpad("n", 8), rpad("median", 9), rpad("predicted", 11),
            rpad("residual", 10), "implied counts")
    levels = Dict{Float64, Vector{Float64}}()
    for (xi, yi) in zip(x, y)
        push!(get!(levels, xi, Float64[]), yi)
    end
    per = Dict{String, Any}()
    for nom in sort(collect(keys(levels)))
        v = levels[nom]
        length(v) >= 20 || continue
        med = median(v)
        pred = a + b * nom
        implied = (med - a) / b                 # what the fit says this group was actually given
        per[string(nom)] = Dict("n" => length(v), "median_measured" => med, "predicted" => pred,
                                "implied_counts" => implied,
                                "cv" => 1.4826 * median(abs.(v .- med)) / max(abs(med), 1e-9))
        println(rpad(nom, 9), rpad(length(v), 8), rpad(round(med; digits = 2), 9),
                rpad(round(pred; digits = 2), 11), rpad(round(med - pred; digits = 2), 10),
                round(implied; digits = 2))
    end

    # how well does the fit recover the notation? the implied counts should land on the nominal ones
    noms = sort(collect(keys(levels)))
    good = [nm for nm in noms if haskey(per, string(nm))]
    err = [abs(per[string(nm)]["implied_counts"] - nm) for nm in good]
    println("\nrecovery: max |implied − nominal| = ", round(maximum(err); digits = 3),
            " counts over ", length(good), " levels")
    open(io -> JSON3.pretty(io, Dict("intercept" => a, "slope" => b, "n" => length(x),
                                     "per_nominal" => per,
                                     "max_recovery_error" => maximum(err))), out, "w")
    println("-> ", out)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS[1], ARGS[2])
end
