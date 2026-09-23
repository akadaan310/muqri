# Build app/data/calibration.json from benchmark runs (Husary = anchor, ijaazah peers, imams).
#
#   julia --project=research_agency_lab/substrate_library/julia \
#         research_agency_lab/substrate_library/julia/calibrate.jl OUT.json RUNS.jsonl...
#
# Reciter sets come from data/taxonomy.json (exported from benchmarks/roster.py); the anchor is
# Husary_128kbps. Everything else (metric choice, bands, LOO, weights) is estimated from the data.
using JSON3
include(joinpath(@__DIR__, "src", "QaariLab.jl"))
using .QaariLab

const HERE = @__DIR__
spec = JSON3.read(read(joinpath(HERE, "data", "metric_spec.json"), String))
tax = JSON3.read(read(joinpath(HERE, "data", "taxonomy.json"), String))
out, runs = ARGS[1], ARGS[2:end]
anchor = "Husary_128kbps"
peers = sort([String(k) for (k, v) in pairs(tax.roster) if v.set == "studio" && String(k) != anchor])
imams = sort([String(k) for (k, v) in pairs(tax.roster) if v.set == "taraweeh"])
cal = QaariLab.calibrate(collect(runs); anchor = anchor, peers = peers, imams = imams, spec = spec, taxonomy = tax)
QaariLab.write_json(out, cal)
v = cal["validation"]
println("count_scale = ", round(cal["count_scale"], digits = 3), " (ruler: ", cal["count_scale_ruler"], ")")
println("bands for ", length(cal["rules"]), " rule keys; reliability ", cal["reliability"])
for r in vcat([anchor], peers, imams)
    raw = get(v["raw_textbook"], r, Dict("perfection" => NaN))["perfection"]
    cal_ = get(v["calibrated_in_sample"], r, Dict("perfection" => NaN))["perfection"]
    loo = haskey(v["leave_one_peer_out"], r) ? v["leave_one_peer_out"][r]["perfection"] : NaN
    println(rpad(r, 40), " raw ", lpad(round(raw, digits = 1), 6), "  calibrated ", lpad(round(cal_, digits = 1), 6),
            "  LOO ", lpad(round(loo, digits = 1), 6))
end
println("weights: ", v["weight_search"])
