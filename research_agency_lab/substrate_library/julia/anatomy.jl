# The anatomy of a recited letter: collision, hold, separation, echo.
#
# The tradition describes a letter's sound in two acts. A letter with sukun is a COLLISION (tasadum):
# the articulators meet and the air and sound are stopped or narrowed. A voweled letter is a
# SEPARATION: the articulators part, and the release is heard like paper torn apart under tension. A
# letter with shaddah is both in one -- the collision of its sakin first half, then the separation of
# its voweled second -- and a master makes both audible, so the doubling is clear even at a distance.
# Qalqalah is the separation a sakin qalqalah letter produces although no vowel follows; it is
# strongest when the reciter stops on it, strongest of all on a stopped letter with shaddah. And a
# certified reviewer heard a fourth thing on the doubled / sakin ب and د: the sound keeps flowing
# into the closure and builds against it before the release -- a voiced HOLD.
#
# Each act is measured from the waveform around a consonant whose onset and end come from the
# engine's alignment (40 ms frames); the measurement itself runs on 5 ms envelopes, so the acts are
# resolved well below the model's frame:
#
#   collision_db   loudness of the preceding vowel minus the quietest point inside the letter:
#                  how completely the sound is stopped
#   approach       steepest fall of the envelope entering the letter, dB per 10 ms: how hard it meets
#   hold_db        low-band (voice-bar) level inside the closure relative to the vowel before it: how
#                  much sound continues through the closure (near 0 = the voice never stopped)
#   burst_db       high-band (pre-emphasised) peak at the release above the closure's high-band floor:
#                  the strength of the separation
#   echo_db        loudness in the 120 ms after the release relative to the vowel before the letter:
#                  for a sakin / stopped qalqalah letter, the bounce
#
# Pure time-domain filters (pre-emphasis for the high band, a moving average for the low band), so
# the same numbers can be reproduced in Octave (octave/anatomy.m) and in numpy without an FFT.
#
#   julia --project=research_agency_lab/substrate_library/julia \
#         research_agency_lab/substrate_library/julia/anatomy.jl UNITS_DIR OUT.json

using JSON3, Statistics

const SR = 16000
const HOP = 80                      # 5 ms

"""RMS envelope in dB over non-overlapping 5 ms windows."""
function env_db(x::AbstractVector{<:Real})
    n = length(x) ÷ HOP
    return [10 * log10(sum(abs2, @view x[(k-1)*HOP+1:k*HOP]) / HOP + 1e-12) for k in 1:n]
end

"""High band: first-order pre-emphasis, the standard proxy for the >2 kHz energy of a burst."""
preemph(x) = [i == 1 ? Float64(x[1]) : x[i] - 0.97 * x[i-1] for i in eachindex(x)]

"""Low band: a 20-sample moving average (first null at 800 Hz) keeps the voice bar, drops frication."""
function lowpass(x; w = 20)
    c = cumsum(vcat(0.0, Float64.(x)))
    return [(c[min(end, i + w)] - c[i]) / w for i in 1:length(x)]
end

win(e, t0, t1) = e[clamp(floor(Int, t0 * SR / HOP) + 1, 1, length(e)):clamp(floor(Int, t1 * SR / HOP), 1, length(e))]

"""The three envelopes of one recording: broadband, high band, low band (5 ms, dB)."""
envelopes(x) = (env_db(x), env_db(preemph(x)), env_db(lowpass(x)))

"""The acts of one consonant spanning [t0, t1) seconds, from the recording's `envelopes`."""
function acts(env, t0, t1)
    e, h, l = env
    pre = win(e, max(0, t0 - 0.06), t0)                 # the vowel before
    pre_l = win(l, max(0, t0 - 0.06), t0)
    inside = win(e, t0, t1)
    (isempty(pre) || isempty(inside) || t0 < 0.06) && return nothing
    ref, ref_l = median(pre), median(pre_l)
    k = argmin(inside)
    closure = win(h, t0, t1)
    lin = win(l, t0, t1)
    # approach: steepest 10 ms fall across the entry into the letter
    entry = win(e, max(0, t0 - 0.03), t0 + 0.03)
    approach = length(entry) > 2 ? maximum(entry[i] - entry[i+2] for i in 1:length(entry)-2) : NaN
    burst_win = win(h, max(t0, t1 - 0.03), t1 + 0.03)
    after = win(e, t1, t1 + 0.12)
    return (collision_db = ref - inside[k], approach = approach,
            hold_db = median(lin) - ref_l,
            burst_db = isempty(burst_win) ? NaN : maximum(burst_win) - minimum(closure),
            echo_db = isempty(after) ? NaN : maximum(after) - ref)
end

"""UNITS_DIR holds wave_<k>.f32 (16 kHz mono) and units.json: [{wave, t0, t1, letter, context, speaker}]."""
function main(dir, out)
    units = JSON3.read(read(joinpath(dir, "units.json"), String))
    cache = Dict{Int, Any}()
    rows = []
    for u in units
        env = get!(cache, u.wave) do
            p = joinpath(dir, "wave_$(u.wave).f32")
            v = Vector{Float32}(undef, filesize(p) ÷ 4); read!(p, v); envelopes(v)
        end
        a = acts(env, u.t0, u.t1)
        a === nothing && continue
        push!(rows, merge((letter = u.letter, context = u.context, speaker = u.speaker), a))
    end
    # per letter x context: medians of each act, and per reciter so masters can be set beside imams
    groups = Dict{Tuple{String, String}, Vector{Any}}()
    for r in rows
        push!(get!(groups, (r.letter, r.context), Any[]), r)
    end
    summary = []
    for ((l, c), rs) in sort(collect(groups), by = kv -> kv[1])
        length(rs) >= 8 || continue
        m(f) = round(median(filter(!isnan, [getfield(r, f) for r in rs])); digits = 2)
        push!(summary, (letter = l, context = c, n = length(rs), collision_db = m(:collision_db),
                        approach = m(:approach), hold_db = m(:hold_db), burst_db = m(:burst_db),
                        echo_db = m(:echo_db)))
    end
    open(io -> JSON3.write(io, (units = rows, summary = summary)), out, "w")
    println(rpad("letter ctx", 16), rpad("n", 6), rpad("collision", 11), rpad("approach", 10),
            rpad("hold", 8), rpad("burst", 8), "echo")
    for s in summary
        println(rpad("$(s.letter) $(s.context)", 16), rpad(s.n, 6), rpad(s.collision_db, 11),
                rpad(s.approach, 10), rpad(s.hold_db, 8), rpad(s.burst_db, 8), s.echo_db)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS...)
end
