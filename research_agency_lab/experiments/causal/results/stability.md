# Causal smoke test: stability report

## zay_fatha/determinism
- TRANSFORM: determinism {} (Engine.analyze twice)
- PHYSICS: — requested direction +0, predicted None, measured None -> n/a (control)
- UNRELATED: head:ghonna:margin = stable; head:hams_or_jahr:margin = stable; head:istitala:margin = stable; head:itbaq:margin = stable; head:qalqla:margin = stable; head:safeer:margin = stable; head:shidda_or_rakhawa:margin = stable; head:tafashie:margin = stable; head:tafkheem_or_taqeeq:margin = stable; head:tikraar:margin = stable; identity:margin = stable; letter:duration_s = stable; rule:madd_tabii:counts = stable; vowel:duration_s = stable; vowel:identity:margin = stable
- COLLATERAL beyond the no-op floor but within same-letter variation: none; beyond same-letter variation: none
- ELSEWHERE IN THE AYAH: 0 verdict flips 

## zay_fatha/noop_splice
- TRANSFORM: noop_splice {} (perturb.splice(original samples))
- PHYSICS: — requested direction +0, predicted None, measured None -> n/a (control)
- UNRELATED: head:ghonna:margin = stable; head:hams_or_jahr:margin = stable; head:istitala:margin = stable; head:itbaq:margin = stable; head:qalqla:margin = stable; head:safeer:margin = stable; head:shidda_or_rakhawa:margin = stable; head:tafashie:margin = stable; head:tafkheem_or_taqeeq:margin = stable; head:tikraar:margin = stable; identity:margin = stable; letter:duration_s = stable; rule:madd_tabii:counts = stable; vowel:duration_s = stable; vowel:identity:margin = stable
- COLLATERAL beyond the no-op floor but within same-letter variation: none; beyond same-letter variation: none
- ELSEWHERE IN THE AYAH: 0 verdict flips 

## zay_fatha/noop_world
- TRANSFORM: noop_world {} (world_lab.world -> synth (no edit))
- PHYSICS: — requested direction +0, predicted None, measured None -> n/a (control)
- UNRELATED: head:ghonna:margin = stable; head:hams_or_jahr:margin = stable; head:istitala:margin = stable; head:itbaq:margin = stable; head:qalqla:margin = stable; head:safeer:margin = stable; head:shidda_or_rakhawa:margin = stable; head:tafashie:margin = stable; head:tafkheem_or_taqeeq:margin = stable; head:tikraar:margin = stable; identity:margin = stable; letter:duration_s = stable; rule:madd_tabii:counts = stable; vowel:duration_s = stable; vowel:identity:margin = stable
- COLLATERAL beyond the no-op floor but within same-letter variation: none; beyond same-letter variation: none
- ELSEWHERE IN THE AYAH: 0 verdict flips 

## zay_fatha/noop_pv
- TRANSFORM: noop_pv {} (perturb.stretch (librosa phase vocoder))
- PHYSICS: — requested direction +0, predicted None, measured None -> n/a (control)
- UNRELATED: head:ghonna:margin = stable; head:hams_or_jahr:margin = stable; head:istitala:margin = stable; head:itbaq:margin = stable; head:qalqla:margin = stable; head:safeer:margin = stable; head:shidda_or_rakhawa:margin = stable; head:tafashie:margin = stable; head:tafkheem_or_taqeeq:margin = stable; head:tikraar:margin = stable; identity:margin = stable; letter:duration_s = stable; rule:madd_tabii:counts = stable; vowel:duration_s = stable; vowel:identity:margin = stable
- COLLATERAL beyond the no-op floor but within same-letter variation: none; beyond same-letter variation: none
- ELSEWHERE IN THE AYAH: 0 verdict flips 

## zay_fatha/boundary-0.02|voicing
- TRANSFORM: voicing {"level": 0.3} (world_lab.devoice)
- PHYSICS: — requested direction +0, predicted None, measured None -> n/a (control)
- UNRELATED: head:ghonna:margin = stable; head:hams_or_jahr:margin = stable; head:istitala:margin = stable; head:itbaq:margin = stable; head:qalqla:margin = stable; head:safeer:margin = stable; head:shidda_or_rakhawa:margin = stable; head:tafashie:margin = stable; head:tafkheem_or_taqeeq:margin = stable; head:tikraar:margin = stable; identity:margin = stable; letter:duration_s = stable; rule:madd_tabii:counts = stable; vowel:duration_s = stable; vowel:identity:margin = stable
- COLLATERAL beyond the no-op floor but within same-letter variation: none; beyond same-letter variation: makhraj:ذ, makhraj:س
- ELSEWHERE IN THE AYAH: 0 verdict flips 

## zay_fatha/boundary+0.02|voicing
- TRANSFORM: voicing {"level": 0.3} (world_lab.devoice)
- PHYSICS: — requested direction +0, predicted None, measured None -> n/a (control)
- UNRELATED: head:ghonna:margin = stable; head:hams_or_jahr:margin = stable; head:istitala:margin = stable; head:itbaq:margin = stable; head:qalqla:margin = stable; head:safeer:margin = stable; head:shidda_or_rakhawa:margin = stable; head:tafashie:margin = stable; head:tafkheem_or_taqeeq:margin = stable; head:tikraar:margin = stable; identity:margin = stable; letter:duration_s = stable; rule:madd_tabii:counts = stable; vowel:duration_s = stable; vowel:identity:margin = stable
- COLLATERAL beyond the no-op floor but within same-letter variation: makhraj:س; beyond same-letter variation: none
- ELSEWHERE IN THE AYAH: 0 verdict flips 

## zay_fatha/same-letter|ز 55:20 55:20:L13
- TRANSFORM: swap {"donor": "ز 55:20 55:20:L13"} (perturb.splice(donor span))
- PHYSICS: — requested direction +0, predicted None, measured None -> n/a (control)
- UNRELATED: head:ghonna:margin = unexpected change; head:hams_or_jahr:margin = stable; head:istitala:margin = unexpected change; head:itbaq:margin = stable; head:qalqla:margin = stable; head:safeer:margin = stable; head:shidda_or_rakhawa:margin = stable; head:tafashie:margin = unexpected change; head:tafkheem_or_taqeeq:margin = stable; head:tikraar:margin = unexpected change; identity:margin = unexpected change; letter:duration_s = stable; rule:madd_tabii:counts = stable; vowel:duration_s = stable; vowel:identity:margin = unexpected change
- COLLATERAL beyond the no-op floor but within same-letter variation: head:ghonna:margin, head:istitala:margin, head:tafashie:margin, head:tikraar:margin, identity:margin, makhraj:ذ, makhraj:س, vowel:identity:margin; beyond same-letter variation: none
- ELSEWHERE IN THE AYAH: 0 verdict flips 

## zay_fatha/neighbour|س 10:44 10:44:L48
- TRANSFORM: swap {"donor": "س 10:44 10:44:L48"} (perturb.splice(donor span))
- PHYSICS: — requested direction +0, predicted None, measured None -> n/a (control)
- TARGET identity:margin: EXPECTED decrease; OBSERVED delta -0.356 (noise 1.0) -> stable
- UNRELATED: head:ghonna:margin = stable; head:hams_or_jahr:margin = unexpected change; head:istitala:margin = stable; head:itbaq:margin = stable; head:qalqla:margin = stable; head:safeer:margin = stable; head:shidda_or_rakhawa:margin = unexpected change; head:tafashie:margin = unexpected change; head:tafkheem_or_taqeeq:margin = unexpected change; head:tikraar:margin = unexpected change; letter:duration_s = stable; rule:madd_tabii:counts = stable; vowel:duration_s = stable; vowel:identity:margin = unexpected change
- COLLATERAL beyond the no-op floor but within same-letter variation: vowel:identity:margin; beyond same-letter variation: head:hams_or_jahr:margin, head:shidda_or_rakhawa:margin, head:tafashie:margin, head:tafkheem_or_taqeeq:margin, head:tikraar:margin, makhraj:س
- ELSEWHERE IN THE AYAH: 0 verdict flips 

## zay_fatha/voicing|{"level": 0.6}
- TRANSFORM: voicing {"level": 0.6} (world_lab.devoice)
- PHYSICS: voiced_fraction requested direction -1, predicted None, measured 0.0 -> NOT_VERIFIED
- TARGET head:hams_or_jahr:margin: EXPECTED decrease; OBSERVED delta -0.248 (noise 1.0) -> stable
- UNRELATED: head:ghonna:margin = stable; head:istitala:margin = stable; head:itbaq:margin = stable; head:qalqla:margin = stable; head:safeer:margin = stable; head:shidda_or_rakhawa:margin = stable; head:tafashie:margin = stable; head:tafkheem_or_taqeeq:margin = stable; head:tikraar:margin = stable; identity:margin = stable; letter:duration_s = stable; rule:madd_tabii:counts = stable; vowel:duration_s = stable; vowel:identity:margin = stable
- COLLATERAL beyond the no-op floor but within same-letter variation: none; beyond same-letter variation: makhraj:ذ, makhraj:س
- ELSEWHERE IN THE AYAH: 0 verdict flips 

## zay_fatha/voicing|{"level": 0.3}
- TRANSFORM: voicing {"level": 0.3} (world_lab.devoice)
- PHYSICS: voiced_fraction requested direction -1, predicted None, measured 0.0 -> NOT_VERIFIED
- TARGET head:hams_or_jahr:margin: EXPECTED decrease; OBSERVED delta -0.679 (noise 1.0) -> stable
- UNRELATED: head:ghonna:margin = stable; head:istitala:margin = stable; head:itbaq:margin = stable; head:qalqla:margin = stable; head:safeer:margin = stable; head:shidda_or_rakhawa:margin = stable; head:tafashie:margin = unexpected change; head:tafkheem_or_taqeeq:margin = stable; head:tikraar:margin = stable; identity:margin = stable; letter:duration_s = stable; rule:madd_tabii:counts = stable; vowel:duration_s = stable; vowel:identity:margin = stable
- COLLATERAL beyond the no-op floor but within same-letter variation: head:tafashie:margin, makhraj:س; beyond same-letter variation: makhraj:ذ
- ELSEWHERE IN THE AYAH: 0 verdict flips 

## zay_fatha/voicing|{"level": 0.0}
- TRANSFORM: voicing {"level": 0.0} (world_lab.devoice)
- PHYSICS: voiced_fraction requested direction -1, predicted None, measured -1.0 -> VERIFIED
- TARGET head:hams_or_jahr:margin: EXPECTED decrease; OBSERVED delta -0.744 (noise 1.0) -> stable
- UNRELATED: head:ghonna:margin = stable; head:istitala:margin = stable; head:itbaq:margin = stable; head:qalqla:margin = stable; head:safeer:margin = stable; head:shidda_or_rakhawa:margin = stable; head:tafashie:margin = stable; head:tafkheem_or_taqeeq:margin = stable; head:tikraar:margin = stable; identity:margin = stable; letter:duration_s = stable; rule:madd_tabii:counts = stable; vowel:duration_s = stable; vowel:identity:margin = stable
- COLLATERAL beyond the no-op floor but within same-letter variation: makhraj:ذ; beyond same-letter variation: none
- ELSEWHERE IN THE AYAH: 0 verdict flips 

## zay_fatha/duration|{"k": 0.5}
- TRANSFORM: duration {"k": 0.5} (perturb.stretch (librosa phase vocoder))
- PHYSICS: span_duration_s requested direction -1, predicted -0.04, measured -0.04 -> VERIFIED
- TARGET letter:duration_s: EXPECTED decrease; OBSERVED delta -0.04 (noise 0.041) -> stable
- UNRELATED: head:ghonna:margin = stable; head:hams_or_jahr:margin = stable; head:istitala:margin = stable; head:itbaq:margin = stable; head:qalqla:margin = stable; head:safeer:margin = stable; head:shidda_or_rakhawa:margin = stable; head:tafashie:margin = unexpected change; head:tafkheem_or_taqeeq:margin = stable; head:tikraar:margin = unexpected change; identity:margin = unexpected change; rule:madd_tabii:counts = stable; vowel:duration_s = stable; vowel:identity:margin = unexpected change
- COLLATERAL beyond the no-op floor but within same-letter variation: head:tikraar:margin, identity:margin, makhraj:ذ; beyond same-letter variation: head:tafashie:margin, makhraj:س, vowel:identity:margin
- ELSEWHERE IN THE AYAH: 0 verdict flips 

## zay_fatha/duration|{"k": 1.5}
- TRANSFORM: duration {"k": 1.5} (perturb.stretch (librosa phase vocoder))
- PHYSICS: span_duration_s requested direction +1, predicted 0.04, measured 0.04 -> VERIFIED
- TARGET letter:duration_s: EXPECTED increase; OBSERVED delta -0.04 (noise 0.041) -> stable
- UNRELATED: head:ghonna:margin = stable; head:hams_or_jahr:margin = unexpected change; head:istitala:margin = stable; head:itbaq:margin = stable; head:qalqla:margin = unexpected change; head:safeer:margin = stable; head:shidda_or_rakhawa:margin = stable; head:tafashie:margin = unexpected change; head:tafkheem_or_taqeeq:margin = unexpected change; head:tikraar:margin = unexpected change; identity:margin = stable; rule:madd_tabii:counts = stable; vowel:duration_s = stable; vowel:identity:margin = unexpected change
- COLLATERAL beyond the no-op floor but within same-letter variation: vowel:identity:margin; beyond same-letter variation: head:hams_or_jahr:margin, head:qalqla:margin, head:tafashie:margin, head:tafkheem_or_taqeeq:margin, head:tikraar:margin, makhraj:س
- ELSEWHERE IN THE AYAH: 0 verdict flips 

## zay_fatha/duration|{"k": 2.0}
- TRANSFORM: duration {"k": 2.0} (perturb.stretch (librosa phase vocoder))
- PHYSICS: span_duration_s requested direction +1, predicted 0.08, measured 0.08 -> VERIFIED
- TARGET letter:duration_s: EXPECTED increase; OBSERVED delta -0.0 (noise 0.041) -> stable
- UNRELATED: head:ghonna:margin = stable; head:hams_or_jahr:margin = unexpected change; head:istitala:margin = unexpected change; head:itbaq:margin = stable; head:qalqla:margin = stable; head:safeer:margin = stable; head:shidda_or_rakhawa:margin = unexpected change; head:tafashie:margin = unexpected change; head:tafkheem_or_taqeeq:margin = unexpected change; head:tikraar:margin = unexpected change; identity:margin = unexpected change; rule:madd_tabii:counts = stable; vowel:duration_s = stable; vowel:identity:margin = unexpected change
- COLLATERAL beyond the no-op floor but within same-letter variation: identity:margin, makhraj:ذ, vowel:identity:margin; beyond same-letter variation: head:hams_or_jahr:margin, head:istitala:margin, head:shidda_or_rakhawa:margin, head:tafashie:margin, head:tafkheem_or_taqeeq:margin, head:tikraar:margin, makhraj:س
- ELSEWHERE IN THE AYAH: 0 verdict flips 

## sad_fatha/determinism
- TRANSFORM: determinism {} (Engine.analyze twice)
- PHYSICS: — requested direction +0, predicted None, measured None -> n/a (control)
- UNRELATED: head:ghonna:margin = stable; head:hams_or_jahr:margin = stable; head:istitala:margin = stable; head:itbaq:margin = stable; head:qalqla:margin = stable; head:safeer:margin = stable; head:shidda_or_rakhawa:margin = stable; head:tafashie:margin = stable; head:tafkheem_or_taqeeq:margin = stable; head:tikraar:margin = stable; identity:margin = stable; letter:duration_s = stable; rule:ikhfa:counts = stable; vowel:duration_s = stable; vowel:identity:margin = stable
- COLLATERAL beyond the no-op floor but within same-letter variation: none; beyond same-letter variation: none
- ELSEWHERE IN THE AYAH: 0 verdict flips 

## sad_fatha/noop_splice
- TRANSFORM: noop_splice {} (perturb.splice(original samples))
- PHYSICS: — requested direction +0, predicted None, measured None -> n/a (control)
- UNRELATED: head:ghonna:margin = stable; head:hams_or_jahr:margin = stable; head:istitala:margin = stable; head:itbaq:margin = stable; head:qalqla:margin = stable; head:safeer:margin = stable; head:shidda_or_rakhawa:margin = stable; head:tafashie:margin = stable; head:tafkheem_or_taqeeq:margin = stable; head:tikraar:margin = stable; identity:margin = stable; letter:duration_s = stable; rule:ikhfa:counts = stable; vowel:duration_s = stable; vowel:identity:margin = stable
- COLLATERAL beyond the no-op floor but within same-letter variation: none; beyond same-letter variation: none
- ELSEWHERE IN THE AYAH: 0 verdict flips 

## sad_fatha/noop_world
- TRANSFORM: noop_world {} (world_lab.world -> synth (no edit))
- PHYSICS: — requested direction +0, predicted None, measured None -> n/a (control)
- UNRELATED: head:ghonna:margin = stable; head:hams_or_jahr:margin = stable; head:istitala:margin = stable; head:itbaq:margin = stable; head:qalqla:margin = stable; head:safeer:margin = stable; head:shidda_or_rakhawa:margin = stable; head:tafashie:margin = stable; head:tafkheem_or_taqeeq:margin = stable; head:tikraar:margin = stable; identity:margin = stable; letter:duration_s = stable; rule:ikhfa:counts = stable; vowel:duration_s = stable; vowel:identity:margin = stable
- COLLATERAL beyond the no-op floor but within same-letter variation: makhraj:ز, makhraj:س; beyond same-letter variation: none
- ELSEWHERE IN THE AYAH: 1 verdict flips ['70:5:L12 tafkheem_or_taqeeq']

## sad_fatha/noop_pv
- TRANSFORM: noop_pv {} (perturb.stretch (librosa phase vocoder))
- PHYSICS: — requested direction +0, predicted None, measured None -> n/a (control)
- UNRELATED: head:ghonna:margin = stable; head:hams_or_jahr:margin = stable; head:istitala:margin = stable; head:itbaq:margin = stable; head:qalqla:margin = stable; head:safeer:margin = stable; head:shidda_or_rakhawa:margin = stable; head:tafashie:margin = stable; head:tafkheem_or_taqeeq:margin = stable; head:tikraar:margin = stable; identity:margin = stable; letter:duration_s = stable; rule:ikhfa:counts = stable; vowel:duration_s = stable; vowel:identity:margin = stable
- COLLATERAL beyond the no-op floor but within same-letter variation: none; beyond same-letter variation: none
- ELSEWHERE IN THE AYAH: 0 verdict flips 

## sad_fatha/boundary-0.02|formant
- TRANSFORM: formant {"a": 0.8} (world_lab.warp)
- PHYSICS: — requested direction +0, predicted None, measured None -> n/a (control)
- UNRELATED: head:ghonna:margin = unexpected change; head:hams_or_jahr:margin = unexpected change; head:istitala:margin = stable; head:itbaq:margin = unexpected change; head:qalqla:margin = unexpected change; head:safeer:margin = unexpected change; head:shidda_or_rakhawa:margin = unexpected change; head:tafashie:margin = stable; head:tafkheem_or_taqeeq:margin = unexpected change; head:tikraar:margin = unexpected change; identity:margin = unexpected change; letter:duration_s = stable; rule:ikhfa:counts = stable; vowel:duration_s = stable; vowel:identity:margin = unexpected change
- COLLATERAL beyond the no-op floor but within same-letter variation: makhraj:ز, vowel:identity:margin; beyond same-letter variation: head:ghonna:margin, head:hams_or_jahr:margin, head:itbaq:margin, head:qalqla:margin, head:safeer:margin, head:shidda_or_rakhawa:margin, head:tafkheem_or_taqeeq:margin, head:tikraar:margin, identity:margin
- ELSEWHERE IN THE AYAH: 1 verdict flips ['70:5:L12 tafkheem_or_taqeeq']

## sad_fatha/boundary+0.02|formant
- TRANSFORM: formant {"a": 0.8} (world_lab.warp)
- PHYSICS: — requested direction +0, predicted None, measured None -> n/a (control)
- UNRELATED: head:ghonna:margin = unexpected change; head:hams_or_jahr:margin = unexpected change; head:istitala:margin = unexpected change; head:itbaq:margin = stable; head:qalqla:margin = unexpected change; head:safeer:margin = unexpected change; head:shidda_or_rakhawa:margin = unexpected change; head:tafashie:margin = unexpected change; head:tafkheem_or_taqeeq:margin = unexpected change; head:tikraar:margin = unexpected change; identity:margin = unexpected change; letter:duration_s = stable; rule:ikhfa:counts = stable; vowel:duration_s = stable; vowel:identity:margin = unexpected change
- COLLATERAL beyond the no-op floor but within same-letter variation: identity:margin, makhraj:ز; beyond same-letter variation: head:ghonna:margin, head:hams_or_jahr:margin, head:istitala:margin, head:qalqla:margin, head:safeer:margin, head:shidda_or_rakhawa:margin, head:tafashie:margin, head:tafkheem_or_taqeeq:margin, head:tikraar:margin, makhraj:س, vowel:identity:margin
- ELSEWHERE IN THE AYAH: 0 verdict flips 

## sad_fatha/same-letter|صَ 74:17 74:17:L12
- TRANSFORM: swap {"donor": "صَ 74:17 74:17:L12"} (perturb.splice(donor span))
- PHYSICS: — requested direction +0, predicted None, measured None -> n/a (control)
- UNRELATED: head:ghonna:margin = unexpected change; head:hams_or_jahr:margin = unexpected change; head:istitala:margin = unexpected change; head:itbaq:margin = stable; head:qalqla:margin = stable; head:safeer:margin = stable; head:shidda_or_rakhawa:margin = unexpected change; head:tafashie:margin = stable; head:tafkheem_or_taqeeq:margin = stable; head:tikraar:margin = stable; identity:margin = unexpected change; letter:duration_s = stable; rule:ikhfa:counts = stable; vowel:duration_s = stable; vowel:identity:margin = unexpected change
- COLLATERAL beyond the no-op floor but within same-letter variation: head:ghonna:margin, head:hams_or_jahr:margin, head:istitala:margin, head:shidda_or_rakhawa:margin, identity:margin, makhraj:ز, makhraj:س, vowel:identity:margin; beyond same-letter variation: none
- ELSEWHERE IN THE AYAH: 1 verdict flips ['70:5:L8 qalqla']

## sad_fatha/neighbour|سَ 10:44 10:44:L48
- TRANSFORM: swap {"donor": "سَ 10:44 10:44:L48"} (perturb.splice(donor span))
- PHYSICS: — requested direction +0, predicted None, measured None -> n/a (control)
- TARGET identity:margin: EXPECTED decrease; OBSERVED delta -13.062 (noise 1.0) -> expected change
- UNRELATED: head:ghonna:margin = unexpected change; head:hams_or_jahr:margin = unexpected change; head:istitala:margin = unexpected change; head:itbaq:margin = unexpected change; head:qalqla:margin = unexpected change; head:safeer:margin = unexpected change; head:shidda_or_rakhawa:margin = unexpected change; head:tafashie:margin = unexpected change; head:tafkheem_or_taqeeq:margin = unexpected change; head:tikraar:margin = unexpected change; letter:duration_s = stable; rule:ikhfa:counts = stable; vowel:duration_s = stable; vowel:identity:margin = unexpected change
- COLLATERAL beyond the no-op floor but within same-letter variation: none; beyond same-letter variation: head:ghonna:margin, head:hams_or_jahr:margin, head:istitala:margin, head:itbaq:margin, head:qalqla:margin, head:safeer:margin, head:shidda_or_rakhawa:margin, head:tafashie:margin, head:tafkheem_or_taqeeq:margin, head:tikraar:margin, makhraj:ز, makhraj:س, vowel:identity:margin
- ELSEWHERE IN THE AYAH: 1 verdict flips ['70:5:L12 tafkheem_or_taqeeq']

## sad_fatha/formant|{"a": 0.9}
- TRANSFORM: formant {"a": 0.9} (world_lab.warp)
- PHYSICS: f2_hz requested direction +1, predicted 120.0, measured 137.9 -> VERIFIED
- TARGET head:tafkheem_or_taqeeq:margin: EXPECTED decrease; OBSERVED delta 1.023 (noise 1.0) -> unexpected change
- UNRELATED: head:ghonna:margin = stable; head:hams_or_jahr:margin = stable; head:istitala:margin = stable; head:itbaq:margin = stable; head:qalqla:margin = stable; head:safeer:margin = stable; head:shidda_or_rakhawa:margin = stable; head:tafashie:margin = stable; head:tikraar:margin = stable; identity:margin = stable; letter:duration_s = stable; rule:ikhfa:counts = stable; vowel:duration_s = stable; vowel:identity:margin = unexpected change
- COLLATERAL beyond the no-op floor but within same-letter variation: vowel:identity:margin; beyond same-letter variation: none
- ELSEWHERE IN THE AYAH: 0 verdict flips 

## sad_fatha/formant|{"a": 0.8}
- TRANSFORM: formant {"a": 0.8} (world_lab.warp)
- PHYSICS: f2_hz requested direction +1, predicted 269.9, measured 296.9 -> VERIFIED
- TARGET head:tafkheem_or_taqeeq:margin: EXPECTED decrease; OBSERVED delta -2.142 (noise 1.0) -> expected change
- UNRELATED: head:ghonna:margin = unexpected change; head:hams_or_jahr:margin = unexpected change; head:istitala:margin = unexpected change; head:itbaq:margin = unexpected change; head:qalqla:margin = stable; head:safeer:margin = unexpected change; head:shidda_or_rakhawa:margin = unexpected change; head:tafashie:margin = unexpected change; head:tikraar:margin = unexpected change; identity:margin = unexpected change; letter:duration_s = stable; rule:ikhfa:counts = stable; vowel:duration_s = stable; vowel:identity:margin = unexpected change
- COLLATERAL beyond the no-op floor but within same-letter variation: vowel:identity:margin; beyond same-letter variation: head:ghonna:margin, head:hams_or_jahr:margin, head:istitala:margin, head:itbaq:margin, head:safeer:margin, head:shidda_or_rakhawa:margin, head:tafashie:margin, head:tikraar:margin, identity:margin, makhraj:ز, makhraj:س
- ELSEWHERE IN THE AYAH: 1 verdict flips ['70:5:L12 tafkheem_or_taqeeq']

## sad_fatha/formant|{"a": 0.7}
- TRANSFORM: formant {"a": 0.7} (world_lab.warp)
- PHYSICS: f2_hz requested direction +1, predicted 462.7, measured 470.4 -> VERIFIED
- TARGET head:tafkheem_or_taqeeq:margin: EXPECTED decrease; OBSERVED delta -6.423 (noise 1.0) -> expected change
- UNRELATED: head:ghonna:margin = unexpected change; head:hams_or_jahr:margin = unexpected change; head:istitala:margin = unexpected change; head:itbaq:margin = unexpected change; head:qalqla:margin = unexpected change; head:safeer:margin = unexpected change; head:shidda_or_rakhawa:margin = unexpected change; head:tafashie:margin = unexpected change; head:tikraar:margin = unexpected change; identity:margin = unexpected change; letter:duration_s = stable; rule:ikhfa:counts = stable; vowel:duration_s = stable; vowel:identity:margin = unexpected change
- COLLATERAL beyond the no-op floor but within same-letter variation: none; beyond same-letter variation: head:ghonna:margin, head:hams_or_jahr:margin, head:istitala:margin, head:itbaq:margin, head:qalqla:margin, head:safeer:margin, head:shidda_or_rakhawa:margin, head:tafashie:margin, head:tikraar:margin, identity:margin, makhraj:ز, makhraj:س, vowel:identity:margin
- ELSEWHERE IN THE AYAH: 1 verdict flips ['70:5:L12 tafkheem_or_taqeeq']

## sad_fatha/f0|{"semitones": -3}
- TRANSFORM: f0 {"semitones": -3} (WORLD f0 x 2^(s/12) (world_lab.world/synth))
- PHYSICS: f0_hz requested direction -1, predicted -27.5, measured -27.59 -> VERIFIED
- UNRELATED: head:ghonna:margin = stable; head:hams_or_jahr:margin = stable; head:istitala:margin = stable; head:itbaq:margin = stable; head:qalqla:margin = stable; head:safeer:margin = stable; head:shidda_or_rakhawa:margin = stable; head:tafashie:margin = stable; head:tafkheem_or_taqeeq:margin = stable; head:tikraar:margin = stable; identity:margin = stable; letter:duration_s = stable; rule:ikhfa:counts = stable; vowel:duration_s = stable; vowel:identity:margin = unexpected change
- COLLATERAL beyond the no-op floor but within same-letter variation: vowel:identity:margin; beyond same-letter variation: none
- ELSEWHERE IN THE AYAH: 0 verdict flips 

## sad_fatha/f0|{"semitones": 3}
- TRANSFORM: f0 {"semitones": 3} (WORLD f0 x 2^(s/12) (world_lab.world/synth))
- PHYSICS: f0_hz requested direction +1, predicted 32.7, measured 32.25 -> VERIFIED
- UNRELATED: head:ghonna:margin = stable; head:hams_or_jahr:margin = stable; head:istitala:margin = stable; head:itbaq:margin = stable; head:qalqla:margin = stable; head:safeer:margin = stable; head:shidda_or_rakhawa:margin = stable; head:tafashie:margin = stable; head:tafkheem_or_taqeeq:margin = stable; head:tikraar:margin = stable; identity:margin = stable; letter:duration_s = stable; rule:ikhfa:counts = stable; vowel:duration_s = stable; vowel:identity:margin = stable
- COLLATERAL beyond the no-op floor but within same-letter variation: none; beyond same-letter variation: none
- ELSEWHERE IN THE AYAH: 1 verdict flips ['70:5:L12 tafkheem_or_taqeeq']

## sad_fatha/f0|{"semitones": 6}
- TRANSFORM: f0 {"semitones": 6} (WORLD f0 x 2^(s/12) (world_lab.world/synth))
- PHYSICS: f0_hz requested direction +1, predicted 71.7, measured 71.05 -> VERIFIED
- UNRELATED: head:ghonna:margin = stable; head:hams_or_jahr:margin = stable; head:istitala:margin = stable; head:itbaq:margin = stable; head:qalqla:margin = stable; head:safeer:margin = stable; head:shidda_or_rakhawa:margin = stable; head:tafashie:margin = stable; head:tafkheem_or_taqeeq:margin = stable; head:tikraar:margin = stable; identity:margin = stable; letter:duration_s = stable; rule:ikhfa:counts = stable; vowel:duration_s = stable; vowel:identity:margin = stable
- COLLATERAL beyond the no-op floor but within same-letter variation: none; beyond same-letter variation: none
- ELSEWHERE IN THE AYAH: 1 verdict flips ['70:5:L12 tafkheem_or_taqeeq']

## ghunnah/determinism
- TRANSFORM: determinism {} (Engine.analyze twice)
- PHYSICS: — requested direction +0, predicted None, measured None -> n/a (control)
- UNRELATED: head:ghonna:margin = stable; head:hams_or_jahr:margin = stable; head:istitala:margin = stable; head:itbaq:margin = stable; head:qalqla:margin = stable; head:safeer:margin = stable; head:shidda_or_rakhawa:margin = stable; head:tafashie:margin = stable; head:tafkheem_or_taqeeq:margin = stable; head:tikraar:margin = stable; identity:margin = stable; letter:duration_s = stable; rule:ghunnah:counts = stable; rule:madd_tabii:counts = stable
- ELSEWHERE IN THE AYAH: 0 verdict flips 

## ghunnah/noop_splice
- TRANSFORM: noop_splice {} (perturb.splice(original samples))
- PHYSICS: — requested direction +0, predicted None, measured None -> n/a (control)
- UNRELATED: head:ghonna:margin = stable; head:hams_or_jahr:margin = stable; head:istitala:margin = stable; head:itbaq:margin = stable; head:qalqla:margin = stable; head:safeer:margin = stable; head:shidda_or_rakhawa:margin = stable; head:tafashie:margin = stable; head:tafkheem_or_taqeeq:margin = stable; head:tikraar:margin = stable; identity:margin = stable; letter:duration_s = stable; rule:ghunnah:counts = stable; rule:madd_tabii:counts = stable
- ELSEWHERE IN THE AYAH: 0 verdict flips 

## ghunnah/noop_world
- TRANSFORM: noop_world {} (world_lab.world -> synth (no edit))
- PHYSICS: — requested direction +0, predicted None, measured None -> n/a (control)
- UNRELATED: head:ghonna:margin = stable; head:hams_or_jahr:margin = stable; head:istitala:margin = stable; head:itbaq:margin = stable; head:qalqla:margin = stable; head:safeer:margin = stable; head:shidda_or_rakhawa:margin = stable; head:tafashie:margin = stable; head:tafkheem_or_taqeeq:margin = stable; head:tikraar:margin = stable; identity:margin = stable; letter:duration_s = stable; rule:ghunnah:counts = stable; rule:madd_tabii:counts = stable
- ELSEWHERE IN THE AYAH: 0 verdict flips 

## ghunnah/noop_pv
- TRANSFORM: noop_pv {} (perturb.stretch (librosa phase vocoder))
- PHYSICS: — requested direction +0, predicted None, measured None -> n/a (control)
- UNRELATED: head:ghonna:margin = stable; head:hams_or_jahr:margin = stable; head:istitala:margin = stable; head:itbaq:margin = stable; head:qalqla:margin = stable; head:safeer:margin = stable; head:shidda_or_rakhawa:margin = stable; head:tafashie:margin = stable; head:tafkheem_or_taqeeq:margin = stable; head:tikraar:margin = stable; identity:margin = stable; letter:duration_s = stable; rule:ghunnah:counts = stable; rule:madd_tabii:counts = stable
- ELSEWHERE IN THE AYAH: 0 verdict flips 

## ghunnah/boundary-0.02|nasal
- TRANSFORM: nasal {"g_db": -12} (world_lab.nasal)
- PHYSICS: — requested direction +0, predicted None, measured None -> n/a (control)
- UNRELATED: head:ghonna:margin = stable; head:hams_or_jahr:margin = stable; head:istitala:margin = stable; head:itbaq:margin = stable; head:qalqla:margin = stable; head:safeer:margin = stable; head:shidda_or_rakhawa:margin = stable; head:tafashie:margin = unexpected change; head:tafkheem_or_taqeeq:margin = stable; head:tikraar:margin = stable; identity:margin = stable; letter:duration_s = stable; rule:ghunnah:counts = stable; rule:madd_tabii:counts = stable
- ELSEWHERE IN THE AYAH: 0 verdict flips 

## ghunnah/boundary+0.02|nasal
- TRANSFORM: nasal {"g_db": -12} (world_lab.nasal)
- PHYSICS: — requested direction +0, predicted None, measured None -> n/a (control)
- UNRELATED: head:ghonna:margin = unexpected change; head:hams_or_jahr:margin = unexpected change; head:istitala:margin = stable; head:itbaq:margin = unexpected change; head:qalqla:margin = stable; head:safeer:margin = unexpected change; head:shidda_or_rakhawa:margin = unexpected change; head:tafashie:margin = unexpected change; head:tafkheem_or_taqeeq:margin = stable; head:tikraar:margin = unexpected change; identity:margin = unexpected change; letter:duration_s = stable; rule:ghunnah:counts = stable; rule:madd_tabii:counts = stable
- ELSEWHERE IN THE AYAH: 0 verdict flips 

## ghunnah/nasal|{"g_db": -6}
- TRANSFORM: nasal {"g_db": -6} (world_lab.nasal)
- PHYSICS: nasal_ratio_db requested direction -1, predicted None, measured -10.54 -> VERIFIED
- TARGET head:ghonna:margin: EXPECTED decrease; OBSERVED delta 0.056 (noise 1.0) -> stable
- UNRELATED: head:hams_or_jahr:margin = stable; head:istitala:margin = stable; head:itbaq:margin = stable; head:qalqla:margin = stable; head:safeer:margin = stable; head:shidda_or_rakhawa:margin = stable; head:tafashie:margin = unexpected change; head:tafkheem_or_taqeeq:margin = stable; head:tikraar:margin = stable; identity:margin = stable; letter:duration_s = stable; rule:ghunnah:counts = stable; rule:madd_tabii:counts = stable
- ELSEWHERE IN THE AYAH: 0 verdict flips 

## ghunnah/nasal|{"g_db": -12}
- TRANSFORM: nasal {"g_db": -12} (world_lab.nasal)
- PHYSICS: nasal_ratio_db requested direction -1, predicted None, measured -20.69 -> VERIFIED
- TARGET head:ghonna:margin: EXPECTED decrease; OBSERVED delta -1.592 (noise 1.0) -> expected change
- UNRELATED: head:hams_or_jahr:margin = unexpected change; head:istitala:margin = unexpected change; head:itbaq:margin = unexpected change; head:qalqla:margin = stable; head:safeer:margin = unexpected change; head:shidda_or_rakhawa:margin = unexpected change; head:tafashie:margin = unexpected change; head:tafkheem_or_taqeeq:margin = stable; head:tikraar:margin = unexpected change; identity:margin = unexpected change; letter:duration_s = stable; rule:ghunnah:counts = stable; rule:madd_tabii:counts = stable
- ELSEWHERE IN THE AYAH: 0 verdict flips 

## ghunnah/nasal|{"g_db": -18}
- TRANSFORM: nasal {"g_db": -18} (world_lab.nasal)
- PHYSICS: nasal_ratio_db requested direction -1, predicted None, measured -30.4 -> VERIFIED
- TARGET head:ghonna:margin: EXPECTED decrease; OBSERVED delta -4.791 (noise 1.0) -> expected change
- UNRELATED: head:hams_or_jahr:margin = unexpected change; head:istitala:margin = unexpected change; head:itbaq:margin = unexpected change; head:qalqla:margin = unexpected change; head:safeer:margin = unexpected change; head:shidda_or_rakhawa:margin = unexpected change; head:tafashie:margin = unexpected change; head:tafkheem_or_taqeeq:margin = unexpected change; head:tikraar:margin = unexpected change; identity:margin = unexpected change; letter:duration_s = stable; rule:ghunnah:counts = stable; rule:madd_tabii:counts = stable
- ELSEWHERE IN THE AYAH: 0 verdict flips 

## ghunnah/duration|{"k": 0.5}
- TRANSFORM: duration {"k": 0.5} (perturb.stretch (librosa phase vocoder))
- PHYSICS: span_duration_s requested direction -1, predicted -0.42, measured -0.42 -> VERIFIED
- TARGET rule:ghunnah:counts: EXPECTED decrease; OBSERVED delta -0.89 (noise 0.25) -> expected change
- UNRELATED: head:ghonna:margin = unexpected change; head:hams_or_jahr:margin = unexpected change; head:istitala:margin = unexpected change; head:itbaq:margin = unexpected change; head:qalqla:margin = unexpected change; head:safeer:margin = unexpected change; head:shidda_or_rakhawa:margin = unexpected change; head:tafashie:margin = unexpected change; head:tafkheem_or_taqeeq:margin = unexpected change; head:tikraar:margin = unexpected change; identity:margin = unexpected change; letter:duration_s = unexpected change; rule:madd_tabii:counts = stable
- ELSEWHERE IN THE AYAH: 0 verdict flips 

## ghunnah/duration|{"k": 1.5}
- TRANSFORM: duration {"k": 1.5} (perturb.stretch (librosa phase vocoder))
- PHYSICS: span_duration_s requested direction +1, predicted 0.42, measured 0.42 -> VERIFIED
- TARGET rule:ghunnah:counts: EXPECTED increase; OBSERVED delta 0.85 (noise 0.25) -> expected change
- UNRELATED: head:ghonna:margin = unexpected change; head:hams_or_jahr:margin = unexpected change; head:istitala:margin = unexpected change; head:itbaq:margin = unexpected change; head:qalqla:margin = stable; head:safeer:margin = unexpected change; head:shidda_or_rakhawa:margin = unexpected change; head:tafashie:margin = unexpected change; head:tafkheem_or_taqeeq:margin = stable; head:tikraar:margin = unexpected change; identity:margin = unexpected change; letter:duration_s = unexpected change; rule:madd_tabii:counts = stable
- ELSEWHERE IN THE AYAH: 0 verdict flips 

## ghunnah/duration|{"k": 2.0}
- TRANSFORM: duration {"k": 2.0} (perturb.stretch (librosa phase vocoder))
- PHYSICS: span_duration_s requested direction +1, predicted 0.84, measured 0.84 -> VERIFIED
- TARGET rule:ghunnah:counts: EXPECTED increase; OBSERVED delta 1.73 (noise 0.25) -> expected change
- UNRELATED: head:ghonna:margin = unexpected change; head:hams_or_jahr:margin = unexpected change; head:istitala:margin = unexpected change; head:itbaq:margin = unexpected change; head:qalqla:margin = stable; head:safeer:margin = unexpected change; head:shidda_or_rakhawa:margin = unexpected change; head:tafashie:margin = unexpected change; head:tafkheem_or_taqeeq:margin = stable; head:tikraar:margin = unexpected change; identity:margin = unexpected change; letter:duration_s = unexpected change; rule:madd_tabii:counts = stable
- ELSEWHERE IN THE AYAH: 0 verdict flips 
