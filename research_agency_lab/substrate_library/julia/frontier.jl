# Distributional (frontier) calibration of every rule family + the B1 beat-normalisation test.
#
#   julia --project=research_agency_lab/substrate_library/julia \
#         research_agency_lab/substrate_library/julia/frontier.jl OUT.json RUNS.jsonl... [--export DIR]
#
# --export DIR writes each key's per-reciter clouds as CSV so octave fr_real_crosscheck.m can recompute
# the barycenter and distances independently.
using JSON3, Statistics
include(joinpath(@__DIR__, "src", "QaariLab.jl"))
using .QaariLab
const Q = QaariLab

args = collect(ARGS)
exp_dir = nothing
if (i = findfirst(==("--export"), args)) !== nothing
    exp_dir = args[i+1]; deleteat!(args, i:i+1); mkpath(exp_dir)
end
out, runs = args[1], args[2:end]
spec = JSON3.read(read(joinpath(@__DIR__, "data", "metric_spec.json"), String))
tax = JSON3.read(read(joinpath(@__DIR__, "data", "taxonomy.json"), String))
anchor = "Husary_128kbps"
peers = sort([String(k) for (k, v) in pairs(tax.roster) if v.set == "studio" && String(k) != anchor])
obs, rows = Q.load_rows(collect(runs); modes = ("studio",))
present = sort(unique(o.reciter for o in obs))
others = [r for r in present if r != anchor && !(r in peers)]
confs = [o.metrics["align_conf"] for o in obs if o.reciter == anchor && haskey(o.metrics, "align_conf")]
conf_min = isempty(confs) ? 0.2 : max(0.2, Q.quantile(confs, 0.02))

model = Q.frontier_calibrate(obs, anchor, peers, others, spec; conf_min)
summary = Q.frontier_summary(model)
beat = Q.beat_normalization([o for o in obs if Q.reliable(o, conf_min, Float64(spec.collapsed_unit_ms), true)])

if exp_dir !== nothing
    for (key, km) in model
        feats = km["features"]
        ok = [o for o in obs if o.key == key && Q.reliable(o, conf_min, Float64(spec.collapsed_unit_ms),
                                                               o.rule_type in String.(spec.duration_rules))]
        kk = replace(key, ':' => '_')
        open(joinpath(exp_dir, "expected__$(kk).csv"), "w") do io
            for (r, v) in sort(collect(km["reciters"]); by = first)
                println(io, r, ",", v["is_reference"] ? 1 : 0, ",", repr(v["loc2"]), ",", repr(v["airm"]), ",", repr(v["total2"]))
            end
        end
        for r in keys(km["reciters"])
            X = Q.cloud(ok, r, feats)
            open(joinpath(exp_dir, "cloud__$(replace(key, ':' => '_'))__$(r).csv"), "w") do io
                for i in 1:size(X, 1); println(io, join(repr.(X[i, :]), ",")); end
            end
        end
    end
end

Q.write_json(out, Dict("anchor" => anchor, "peers_present" => [p for p in peers if p in present],
                       "others" => others, "conf_min" => conf_min, "keys" => model, "summary" => summary,
                       "beat_normalization" => Dict(k => v for (k, v) in beat if k != "series")))
println("rule keys modelled: ", length(model), "   reciters: ", join(present, ", "))
println(rpad("reciter", 40), " PASS WARN FAIL  inst.index")
for r in sort(collect(keys(summary)))
    s = summary[r]
    println(rpad(r, 40), lpad(s["keys_pass"], 5), lpad(s["keys_warn"], 5), lpad(s["keys_fail"], 5),
            "  ", round(s["instance_index"], digits = 1))
end
println("\nB1 beat normalisation (robust CV raw / global counts / local counts):")
for (k, v) in sort(collect(beat["per_key"]); by = first)
    v["n"] > 0 && println(rpad(k, 22), " n=", lpad(v["n"], 4), "  ", join(round.([v["rcv_raw"], v["rcv_global"], v["rcv_local"]], digits = 3), " / "),
                          "   median counts global ", round(v["median_global"], digits = 2), " local ", round(v["median_local"], digits = 2))
end
println("Fisher separation raw/global/local: ", round.([beat["fisher_raw"], beat["fisher_global"], beat["fisher_local"]], digits = 3))
