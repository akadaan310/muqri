# The causal layer: which mechanisms and contexts PRODUCE which measurable acts, tested on the data.
#
# The set layer (calculus.jl, reciter_sets.jl) says what goes with what. A causal edge says what
# produces what, and must survive holding the confounders fixed. The structure follows the
# tradition's anatomy of sound:
#
#   mechanisms    air held / flowing (jahr / hams), sound held / flowing (shiddah / rakhawah),
#                 qalqalah class (jahr ∧ shadid, minus the hamza)
#   interventions voweled (separation), sakin (collision), shaddah (both), the stop (waqf), tier
#   observables   collision_db, hold_db, burst_db, echo_db (anatomy.jl, waveform); the heads' classes
#
# Every edge carries its PROVENANCE -- "tradition" (a certified reciter's teaching or the books) --
# and its TEST: the effect estimated by comparing like with like. For a binary intervention T on an
# outcome Y, the strata are (letter, reciter): the same letter by the same voice with and without T,
# so letter identity and voice cannot masquerade as the effect. The estimate is the stratum-weighted
# mean difference (weights min(n1, n0)), with a bootstrap over strata for the 95 % interval. An edge
# is "supported" when the interval excludes 0 in the direction the tradition states, "refuted" when
# it excludes 0 the other way, "not shown" otherwise.
#
#   julia --project=research_agency_lab/substrate_library/julia \
#         research_agency_lab/substrate_library/julia/causal.jl ANATOMY_JSON CALC_DIR OUT.json

using JSON3, Statistics, Random

const QALQALAH = Set(['ق', 'ط', 'ب', 'ج', 'د'])
const VOICED_STOPS = Set(['ب', 'د', 'ج', 'ط', 'ق'])      # jahr ∧ shadid, the hamza aside
const VOICELESS_STOPS = Set(['ت', 'ك'])                  # hams ∧ shadid
const MASTERS = Set(["Husary_128kbps", "Husary_Muallim_128kbps", "Husary_128kbps_Mujawwad",
                     "Minshawy_Murattal_128kbps", "Minshawy_Mujawwad_192kbps"])

"""Stratum-weighted mean difference E[Y | T=1] - E[Y | T=0] within strata, and a bootstrap CI."""
function stratified_effect(rows, treat::Function, y::Symbol, stratum::Function; B = 2000, seed = 7)
    groups = Dict{Any, Tuple{Vector{Float64}, Vector{Float64}}}()
    for r in rows
        t = treat(r)
        t === nothing && continue
        v = Float64(getproperty(r, y))
        isnan(v) && continue
        g = get!(groups, stratum(r), (Float64[], Float64[]))
        push!(t ? g[1] : g[2], v)
    end
    strata = [(mean(a) - mean(b), min(length(a), length(b))) for (a, b) in values(groups) if !isempty(a) && !isempty(b)]
    isempty(strata) && return nothing
    est(ss) = sum(d * w for (d, w) in ss) / sum(w for (_, w) in ss)
    point = est(strata)
    rng = MersenneTwister(seed)
    boots = [est(strata[rand(rng, 1:length(strata), length(strata))]) for _ in 1:B]
    lo, hi = quantile(boots, [0.025, 0.975])
    return (effect = point, ci = (lo, hi), strata = length(strata),
            n = sum(w for (_, w) in strata))
end

verdict(e, sign) = e === nothing ? "untested" :
    (sign * e.ci[1] > 0 || sign * e.ci[2] > 0) && (e.ci[1] > 0 || e.ci[2] < 0) ?
        (sign * e.effect > 0 ? "supported" : "refuted") : "not shown"

function main(anatomy, calcdir, out)
    A = JSON3.read(read(anatomy, String)).units
    rows = [(letter = only(r.letter), context = String(r.context), speaker = String(r.speaker),
             collision_db = Float64(something(r.collision_db, NaN)), hold_db = Float64(something(r.hold_db, NaN)),
             burst_db = Float64(something(r.burst_db, NaN)), echo_db = Float64(something(r.echo_db, NaN)))
            for r in A]
    by_letter_voice(r) = (r.letter, r.speaker)
    by_voice(r) = r.speaker
    ctx(c) = r -> r.context == c
    edges = []
    function edge(cause, effect_on, claim, sign, e; mechanism = "", provenance = "tradition")
        v = verdict(e, sign)
        push!(edges, (cause = cause, effect = effect_on, claim = claim, provenance = provenance,
                      mechanism = mechanism, expected_sign = sign, verdict = v,
                      estimate = e === nothing ? nothing : round(e.effect; digits = 3),
                      ci95 = e === nothing ? nothing : round.(collect(e.ci); digits = 3),
                      strata = e === nothing ? 0 : e.strata, n = e === nothing ? 0 : e.n))
        println(rpad("$cause → $effect_on", 40), rpad(v, 11),
                e === nothing ? "" : "Δ=$(round(e.effect; digits = 2))  [$(round(e.ci[1]; digits = 2)), $(round(e.ci[2]; digits = 2))]  strata=$(e.strata)")
    end

    stops = filter(r -> r.letter in union(VOICED_STOPS, VOICELESS_STOPS), rows)
    # shaddah = collision + separation, against the same letter voweled, same voice
    two(a, b) = r -> r.context == a ? true : (r.context == b ? false : nothing)
    edge("shaddah", "collision_db", "a doubled letter collides (its sakin half)", +1,
         stratified_effect(stops, two("shaddah", "voweled"), :collision_db, by_letter_voice); mechanism = "closure")
    edge("shaddah", "burst_db", "a doubled letter separates (its voweled half), harder", +1,
         stratified_effect(stops, two("shaddah", "voweled"), :burst_db, by_letter_voice); mechanism = "release")
    edge("vowel", "burst_db", "a voweled letter is a separation; a sakin one is not", +1,
         stratified_effect(stops, two("voweled", "sakin"), :burst_db, by_letter_voice); mechanism = "release")
    # qalqalah: a sakin qalqalah letter releases without a vowel; the stop strengthens it
    q = filter(r -> r.letter in QALQALAH, rows)
    edge("stop", "echo_db", "qalqalah is stronger at the stop than mid-word", +1,
         stratified_effect(q, two("stop", "sakin"), :echo_db, by_letter_voice); mechanism = "qalqalah")
    edge("shaddah_stop", "echo_db", "strongest on a stopped letter with shaddah", +1,
         stratified_effect(q, two("shaddah_stop", "stop"), :echo_db, by_voice); mechanism = "qalqalah")
    # the qalqalah class itself: jahr ∧ shadid produces the echo at sukun; hams ∧ shadid does not
    sak = filter(r -> r.context == "sakin" && r.letter in union(QALQALAH, VOICELESS_STOPS), rows)
    edge("qalqalah_class", "echo_db", "jahr ∧ shadid letters bounce at sukun; ت ك do not", +1,
         stratified_effect(sak, r -> r.letter in QALQALAH, :echo_db, by_voice); mechanism = "qalqalah")
    # the voiced hold: jahr keeps the voice through the closure
    edge("jahr", "hold_db", "voicing continues through a voiced stop's closure", +1,
         stratified_effect(filter(r -> r.context in ("sakin", "shaddah"), stops),
                           r -> r.letter in VOICED_STOPS, :hold_db, r -> (r.speaker, r.context)); mechanism = "voice")
    bd = filter(r -> r.context in ("sakin", "shaddah") && r.letter in VOICED_STOPS, rows)
    edge("letter ∈ {ب, د}", "hold_db", "among the voiced stops, ب and د hold the voice most (the voiced hold)", +1,
         stratified_effect(bd, r -> r.letter in ('ب', 'د'), :hold_db, r -> (r.speaker, r.context));
         mechanism = "voice", provenance = "certified reviewer")
    # the hamza: jahr and shadid, yet the stopping letter -- no bounce
    hz = filter(r -> r.context == "sakin" && (r.letter == 'ء' || r.letter in QALQALAH), rows)
    edge("hamza (vs qalqalah letters)", "echo_db", "the stopping hamza does not bounce", -1,
         stratified_effect(hz, r -> r.letter == 'ء', :echo_db, by_voice); mechanism = "qalqalah")
    # mastery: masters make the shaddah clearer (same letter; tier is a person, so this is adjusted
    # association, not an intervention)
    sh = filter(r -> r.context == "shaddah", stops)
    edge("master (vs fast imam)", "burst_db on shaddah", "a master makes the doubling audible", +1,
         stratified_effect(sh, r -> r.speaker in MASTERS, :burst_db, r -> r.letter); mechanism = "mastery",
         provenance = "certified reviewer")
    edge("master (vs fast imam)", "collision_db on shaddah", "a master makes the collision complete", +1,
         stratified_effect(sh, r -> r.speaker in MASTERS, :collision_db, r -> r.letter); mechanism = "mastery",
         provenance = "certified reviewer")
    hams = filter(r -> r.context == "sakin" && r.letter in VOICELESS_STOPS, rows)
    edge("master (vs fast imam)", "echo_db on sakin ت ك", "masters do not bounce a hams stop", -1,
         stratified_effect(hams, r -> r.speaker in MASTERS, :echo_db, r -> r.letter); mechanism = "mastery",
         provenance = "discovered (anatomy)")

    open(io -> JSON3.write(io, (edges = edges, n_units = length(rows),
                                method = "stratum-weighted mean difference within (letter, reciter), bootstrap CI over strata")),
         out, "w")
    println("\n-> $out")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS...)
end
