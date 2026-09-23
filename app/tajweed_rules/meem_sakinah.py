"""Meem Sakinah: Ikhfa' Shafawi (before ب), Idghaam Shafawi / Mithlayn (before م), Izhaar Shafawi.

* Ikhfa' Shafawi: a 2-harakah nasal hum with a *light* labial contact, not a full silent closure.
* Idghaam Shafawi: one mushaddad meem carrying 2–2.5 harakat of ghunnah.
* Izhaar Shafawi: a clear meem released within 1.0 harakah; a hold beyond 1.3 harakat fails,
  which matters most before و and ف (where the lips are tempted to linger).
"""

from __future__ import annotations

from app.models import RuleDiagnostic, RuleInstance, RuleType, Status
from app.tajweed_rules.base import EvalContext, skipped
from app.tajweed_rules.noon_sakinah import clear_letter, nasal_hold, nasal_window, validate_iqlab_or_ikhfa_shafawi


def validate_ikhfa_shafawi(rule: RuleInstance, ev: EvalContext) -> RuleDiagnostic:
    return validate_iqlab_or_ikhfa_shafawi(rule, ev)


def validate_idgham_shafawi(rule: RuleInstance, ev: EvalContext) -> RuleDiagnostic:
    window = nasal_window(rule, ev, include_target=True)
    if window is None:
        return skipped(rule, f"Idghaam Shafawi on '{rule.word}' could not be located in the audio.")
    return nasal_hold(rule, ev, window, 2.0, 2.5)


def validate_izhar_shafawi(rule: RuleInstance, ev: EvalContext) -> RuleDiagnostic:
    diag = clear_letter(rule, ev, "Meem")
    if diag.status is Status.FAIL and rule.detail:
        diag.feedback += " Before و/ف the lips must part immediately."
    return diag


VALIDATORS = {
    RuleType.IKHFA_SHAFAWI: validate_ikhfa_shafawi,
    RuleType.IDGHAM_SHAFAWI: validate_idgham_shafawi,
    RuleType.IZHAR_SHAFAWI: validate_izhar_shafawi,
}
