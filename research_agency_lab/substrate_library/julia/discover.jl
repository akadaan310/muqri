# Model discovery: tempo-dynamics ODE and the sparse duration law for each reciter.
#
#   julia --project=research_agency_lab/substrate_library/julia \
#         research_agency_lab/substrate_library/julia/discover.jl OUT.json RUNS.jsonl...
using JSON3
include(joinpath(@__DIR__, "src", "QaariLab.jl"))
using .QaariLab

tax = JSON3.read(read(joinpath(@__DIR__, "data", "taxonomy.json"), String))
expected = Dict(String(k) => Float64(v) for (k, v) in pairs(tax.expected_counts))
out, runs = ARGS[1], ARGS[2:end]
obs, rows = QaariLab.load_rows(collect(runs); modes = ("studio",))
reciters = sort(unique(o.reciter for o in obs))
result = Dict{String,Any}()
for r in reciters
    law_core = QaariLab.duration_law(obs, r, expected; metric = "core_ms")
    law_span = QaariLab.duration_law(obs, r, expected; metric = "duration_ms")
    tempo = QaariLab.tempo_dynamics(rows, r)
    result[r] = Dict("duration_law_core" => law_core, "duration_law_span" => law_span, "tempo_dynamics" => tempo)
    println(r)
    for (nm, law) in (("core", law_core), ("span", law_span))
        haskey(law, "terms_ms") || continue
        println("  D_", nm, " = ", join(["$(round(v, digits = 1))·$k" for (k, v) in law["terms_ms"]], " + "),
                "   R²=", round(law["r2"], digits = 3), "  ms/count=", round(law["ms_per_count_at_median_T"], digits = 1),
                "  overhead=", round(law["overhead_ms_at_median_T"], digits = 1), " ms")
    end
    p = tempo["pooled"]
    isempty(p) || println("  tempo over ", p["surahs"], " surahs: preferred ", p["share_preferred"],
                          "; drift ", round(p["median_drift_ms_per_min"], digits = 2), " ms/min; τc (relaxation) ",
                          round(p["median_tau_s_where_relaxation"], digits = 1), " s")
end
QaariLab.write_json(out, result)
