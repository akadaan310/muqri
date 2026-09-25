# Independent numerical layer for the causal-experiment harness (research_agency_lab/experiments/causal).
#
#   julia --project=research_agency_lab/substrate_library/julia xcheck.jl spans index.json out.json
#       Raw float32 spans (16 kHz) -> duration_s, voiced_fraction (own normalised autocorrelation,
#       30 ms frames, 10 ms hop, 70-400 Hz lag range, peak > 0.45), nasal_ratio_db (own radix-2 FFT,
#       Hann, 10 log10 E[150-400] / E[750-1100]), rms_db. Written without the Python code.
#
#   julia --project=... xcheck.jl table records.json out.json
#       Re-derives every delta and its direction (transformed - baseline) for physics and Muqri metrics
#       straight from the stored raw values, so the Python classification can be checked against it.

using JSON3
using Statistics

const SR = 16000

function fft!(a::Vector{ComplexF64})
    n = length(a)
    j = 1
    for i in 1:n-1                                    # bit reversal
        if i < j
            a[i], a[j] = a[j], a[i]
        end
        m = n >> 1
        while m >= 1 && j > m
            j -= m
            m >>= 1
        end
        j += m
    end
    len = 2
    while len <= n
        w = exp(-2im * pi / len)
        for s in 1:len:n
            wk = 1.0 + 0im
            for k in 0:(len >> 1)-1
                u = a[s+k]
                v = a[s+k+(len >> 1)] * wk
                a[s+k] = u + v
                a[s+k+(len >> 1)] = u - v
                wk *= w
            end
        end
        len <<= 1
    end
    a
end

function band_energy(x::Vector{Float64}, lo, hi)
    n = max(512, nextpow(2, length(x)))
    h = [0.5 - 0.5 * cos(2pi * (i - 1) / (length(x) - 1)) for i in 1:length(x)]
    a = zeros(ComplexF64, n)
    a[1:length(x)] .= x .* h
    fft!(a)
    e = 0.0
    for k in 0:(n >> 1)
        f = k * SR / n
        if lo <= f < hi
            e += abs2(a[k+1])
        end
    end
    e
end

function voiced_fraction(x::Vector{Float64})
    fl, hop = round(Int, 0.03SR), round(Int, 0.01SR)
    lmin, lmax = floor(Int, SR / 400), ceil(Int, SR / 70)
    length(x) < fl && return nothing
    v = 0; t = 0
    for s in 1:hop:(length(x) - fl + 1)
        f = x[s:s+fl-1] .- mean(x[s:s+fl-1])
        e0 = sum(abs2, f)
        t += 1
        e0 <= 1e-10 && continue
        best = maximum(sum(f[1:fl-l] .* f[1+l:fl]) / e0 for l in lmin:min(lmax, fl - 1))
        v += best > 0.45
    end
    t == 0 ? nothing : v / t
end

function spans(index_path, out_path)
    idx = JSON3.read(read(index_path, String))
    out = Dict{String,Any}()
    for it in idx
        x = Float64.(reinterpret(Float32, read(String(it.file))))
        vf = voiced_fraction(x)
        lo, an = band_energy(x, 150, 400), band_energy(x, 750, 1100)
        out[String(it.id)] = Dict("duration_s" => round(length(x) / SR; digits=4),
            "voiced_fraction" => vf === nothing ? nothing : round(vf; digits=4),
            "nasal_ratio_db" => round(10 * log10((lo + 1e-12) / (an + 1e-12)); digits=2),
            "rms_db" => round(20 * log10(sqrt(mean(abs2, x)) + 1e-12); digits=2))
    end
    write(out_path, JSON3.write(out))
end

num(v) = v isa Number && !(v isa Bool) && isfinite(v) ? Float64(v) : nothing

function table(records_path, out_path)
    recs = JSON3.read(read(records_path, String))
    out = Any[]
    for r in recs
        for (layer, a, b) in (("physics", r.physics_baseline, r.physics_transformed),
                              ("muqri", r.muqri_baseline, r.muqri_transformed))
            for k in union(keys(a), keys(b))
                x, y = num(get(a, k, nothing)), num(get(b, k, nothing))
                d = (x === nothing || y === nothing) ? nothing : round(y - x; digits=4)
                push!(out, Dict("experiment_id" => r.experiment_id, "layer" => layer, "metric" => String(k),
                                "delta" => d, "direction" => d === nothing ? nothing : Int(sign(d))))
            end
        end
    end
    write(out_path, JSON3.write(out))
end

if ARGS[1] == "spans"
    spans(ARGS[2], ARGS[3])
elseif ARGS[1] == "table"
    table(ARGS[2], ARGS[3])
else
    error("mode: spans | table")
end
