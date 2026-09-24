# The ra' calculus: heaviness of every ra' in master recitations -- what the classical rules say, what
# the engine's reference expects, and what the acoustic model hears -- by context.
#
# Input: research_agency_lab/experiments/quran/ra_instances.jsonl (ra_extract.py): every ra' of al-Qamar
# (55 ayahs x 10 masters, nearly all ending in ra') and of T300 (the 23 ra'-final verses x 41 reciters,
# plus 1,000 other recitations), with the four sounds before it, the sound after, doubling, and whether
# it closes the ayah.
#
# The classical rule, from the context alone:
#   voweled ra'   -- its own vowel: fatha / damma (or the madd alif / waw after it) heavy; kasra (madd ya) light
#   sakin or at a stop -- the vowel before it: fatha / damma heavy, kasra light (but heavy before a heavy
#                  letter in the same word); a sakin letter between looks one vowel further back, except a
#                  sakin ya (khayr, qadir) -> light and a sakin heavy letter after kasrah (misr) -> either
#   after a madd  -- alif / waw heavy, ya light
#   doubled       -- like any voweled ra' (رَّ heavy, رِّ light); only a doubled ra' at a stop looks back
#   a stop on tanwin fath (صَبْرًا -> ṣabrā) keeps the ra' voweled; يَسْرِ (its ya' dropped) allows either
# Then, per context class: the reference's expectation, the model's hearing, and which disagrees with the
# classical rule -- a reference error (masters and rule agree, reference not) or a perception blind spot
# (rule and reference agree, the model hears otherwise).
#
#   ~/julia-1.11.5/bin/julia --project=research_agency_lab/substrate_library/julia \
#       research_agency_lab/substrate_library/julia/ra.jl

using JSON3, Statistics

const ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
const DATA = joinpath(ROOT, "research_agency_lab/experiments/quran/ra_instances.jsonl")
const OUT = joinpath(ROOT, "research_agency_lab/experiments/quran/ra_calculus.json")
const HEAVY_LETTERS = Set(collect("خصضغطقظ"))
const FATHA, DAMMA, KASRA = 'َ', 'ُ', 'ِ'
const HEAVY_VOWELS = Set([FATHA, DAMMA, 'ا', 'ۥ'])
const LIGHT_VOWELS = Set([KASRA, 'ۦ'])

isvowel(c) = c in HEAVY_VOWELS || c in LIGHT_VOWELS

"The classical verdict and the context class it came from."
function classical(prev::Vector{Char}, nxt::Char, doubled::Bool, final::Bool, word_final::Bool, bare::String="")
    endswith(bare, "يسر") && final && return ("either", "stop:yasr (dropped ya')")
    prev = [c for c in prev if c != 'ڇ']            # the qalqalah echo is not a vowel: skip to the letter
    # a heavy letter after a kasra-ra' only counts inside the same word (تُصَعِّرْ خَدَّكَ stays light)
    nxt_in_word = word_final ? ' ' : nxt
    own = nxt
    if isvowel(own)                        # a stop on tanwin fath keeps its vowel in the stream
        return (own in HEAVY_VOWELS ? "heavy" : "light", (doubled ? "doubled:" : "voweled:") * string(own))
    end
    # sakin, at a stop, or doubled: look back
    isempty(prev) && return ("?", "no context")
    p1 = prev[end]
    tag = (final ? (doubled ? "stop doubled" : "stop") : "sakin")
    if p1 in ('ا', 'ۥ')
        return ("heavy", tag * ":after madd " * string(p1))
    elseif p1 == 'ۦ'
        return ("light", tag * ":after madd ya")
    elseif p1 == FATHA || p1 == DAMMA
        return ("heavy", tag * ":after " * (p1 == FATHA ? "fatha" : "damma"))
    elseif p1 == KASRA
        return (!final && nxt_in_word in HEAVY_LETTERS ? "heavy" : "light", tag * ":after kasra" * (!final && nxt_in_word in HEAVY_LETTERS ? " before a heavy letter" : ""))
    else                                   # a sakin letter between
        length(prev) < 2 && return ("?", tag * ":after sakin, no vowel")
        p2 = prev[end - 1]
        if p1 == 'ي'
            return ("light", tag * ":after sakin ya")
        elseif p1 in HEAVY_LETTERS && p2 == KASRA
            return ("either", tag * ":after a sakin heavy letter after kasra")
        elseif p2 == FATHA || p2 == DAMMA || p2 in ('ا', 'ۥ')
            return ("heavy", tag * ":after sakin letter after " * (p2 == DAMMA ? "damma" : "fatha"))
        elseif p2 == KASRA || p2 == 'ۦ'
            return ("light", tag * ":after sakin letter after kasra")
        end
        return ("?", tag * ":after sakin, before it " * string(p2))
    end
end

cls(s) = occursin("مفخم", s) && !occursin("أدنى", s) ? "heavy" : (occursin("مرقق", s) ? "light" : (occursin("أدنى", s) ? "heavy" : "other"))

function main()
    rows = []
    for line in eachline(DATA)
        r = JSON3.read(line)
        haskey(r, :none) && continue
        prev = Char[only(collect(String(p))) for p in r.prev if length(collect(String(p))) == 1]
        nxt = only(collect(String(r.next)))
        # the ra' closes its word when the word's bare letters end in ra'
        bare = [c for c in String(r.word) if 'ء' <= c <= 'ي']
        word_final = !isempty(bare) && bare[end] == 'ر'
        rule, ctx = classical(prev, nxt, r.run_length >= 2, r.ayah_final, word_final, String(bare))
        push!(rows, (ctx = ctx, rule = rule, ref = cls(String(r.expected)), heard = cls(String(r.observed)),
                     margin = Float64(r.margin), word = String(r.word), final = r.ayah_final, source = String(r.source),
                     speaker = String(r.speaker), surah = r.surah, ayah = r.ayah))
    end
    println(length(rows), " ra' instances (", count(r -> r.final, rows), " closing an ayah)")
    groups = Dict{String, Vector{Any}}()
    for r in rows
        push!(get!(groups, r.ctx, Any[]), r)
    end
    out = []
    println(rpad("context", 52), rpad("n", 6), rpad("rule", 8), rpad("reference", 18), rpad("model hears heavy", 20), "finding")
    for (ctx, rs) in sort(collect(groups); by = g -> -length(g[2]))
        n = length(rs)
        rule = rs[1].rule
        ref_heavy = count(r -> r.ref == "heavy", rs) / n
        heard_heavy = count(r -> r.heard == "heavy", rs) / n
        rule_heavy = rule == "heavy" ? 1.0 : (rule == "light" ? 0.0 : NaN)
        finding = if isnan(rule_heavy)
            "either / undetermined"
        elseif abs(ref_heavy - rule_heavy) > 0.5 && abs(heard_heavy - rule_heavy) < 0.5
            "REFERENCE ERROR: masters follow the rule"
        elseif abs(ref_heavy - rule_heavy) > 0.5
            "reference disagrees with the rule"
        elseif abs(heard_heavy - rule_heavy) > 0.5
            "PERCEPTION blind spot"
        elseif abs(heard_heavy - rule_heavy) > 0.2
            "model unsure here"
        else
            "agree"
        end
        words = Dict{String, Int}()
        for r in rs
            words[r.word] = get(words, r.word, 0) + 1
        end
        ex = [w for (w, _) in sort(collect(words); by = x -> -x[2])[1:min(4, length(words))]]
        n >= 10 && println(rpad(ctx, 52), rpad(n, 6), rpad(rule, 8), rpad(round(ref_heavy; digits = 2), 18),
                           rpad(round(heard_heavy; digits = 2), 20), finding, "   ", join(ex, " "))
        push!(out, Dict("context" => ctx, "n" => n, "rule" => rule, "reference_heavy" => ref_heavy,
                        "heard_heavy" => heard_heavy, "finding" => finding, "examples" => ex,
                        "margin_median" => median([r.margin for r in rs])))
    end
    open(replace(OUT, "ra_calculus.json" => "ra_contexts.jsonl"), "w") do io
        for r in rows
            println(io, JSON3.write(r.ctx))
        end
    end
    open(OUT, "w") do io
        JSON3.pretty(io, Dict("instances" => length(rows), "contexts" => out))
    end
    println("-> ", OUT)
end

main()
