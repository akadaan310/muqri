"""Orchestrator CLI: gates, budget and the cloud plan.

    .venv/bin/python -m research_agency_lab.orchestrator status
    .venv/bin/python -m research_agency_lab.orchestrator check              # run tests + julia, record the gate
    .venv/bin/python -m research_agency_lab.orchestrator authorise TAG      # before any cloud launch
    .venv/bin/python -m research_agency_lab.orchestrator confirm-bill USD CONTAINER_S   # calibrate billing
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from research_agency_lab.orchestrator.budget import CPU_CORE_S, MEM_GIB_S, BudgetExceeded, Ledger
from research_agency_lab.orchestrator.gates import METRICS, ROOT, STAGES, plan, status

JULIA = Path.home() / "julia-1.11.5" / "bin" / "julia"


def cmd_status(ledger: Ledger) -> None:
    print(f"budget: cap ${ledger.cap_usd:.2f}, spent ${ledger.spent:.2f}, remaining ${ledger.remaining:.2f}; "
          f"billing factor {ledger.billing_factor} ({'confirmed' if ledger.billing_factor_confirmed else 'UNCONFIRMED'})")
    for w, r in ledger.rates.items():
        print(f"rate {w}: {r.mean:.2f} ± {r.var ** 0.5:.2f} s/row (upper {r.upper():.2f})")
    print("\ngates:")
    for name, (ok, why) in status().items():
        print(f"  {'PASS' if ok else 'FAIL' if ok is False else '----'}  {name:15s} {why}")
    print("\ncloud plan (value per upper-bound $):")
    for r in plan(ledger):
        state = f"blocked by {','.join(r['blocked_by'])}" if r["blocked_by"] else \
            ("OK within budget" if r.get("fits_budget") else "over budget")
        print(f"  {r['tag']:26s} ${r['expected_usd']:>7} (≤ ${r['upper_usd']}) {r['container_hours']:>6} h  {state}")


def cmd_check() -> None:
    py = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x"], cwd=ROOT, capture_output=True, text=True)
    P = ROOT / "research_agency_lab/substrate_library/julia"
    jl = subprocess.run([str(JULIA), f"--project={P}", str(P / "test/test_frontier.jl")], capture_output=True, text=True)
    METRICS.mkdir(exist_ok=True)
    (METRICS / "tests.json").write_text(json.dumps({"pytest_ok": py.returncode == 0, "julia_ok": jl.returncode == 0,
                                                    "pytest_tail": py.stdout.strip().splitlines()[-1:]}) + "\n")
    print(f"pytest rc={py.returncode}, julia rc={jl.returncode}")


def main(argv: list[str]) -> int:
    ledger = Ledger.load()
    cmd = argv[0] if argv else "status"
    if cmd == "status":
        cmd_status(ledger)
    elif cmd == "check":
        cmd_check()
    elif cmd == "authorise":
        stage = next(s for s in STAGES if s.tag == argv[1])
        blocked = [g for g in stage.requires if status()[g][0] is not True]
        if blocked:
            print(f"refused: gates not passed: {blocked}")
            return 2
        try:
            print(ledger.authorise(stage.tag, stage.workload, stage.rows, cpu=stage.cpu, mem_gib=stage.mem_gib,
                                   gpu=stage.gpu))
        except BudgetExceeded as exc:
            print(f"refused: {exc}")
            return 2
        ledger.save()
    elif cmd == "confirm-bill":
        usd, container_s = float(argv[1]), float(argv[2])
        ledger.billing_factor = round(usd / (container_s * (2 * CPU_CORE_S + 6 * MEM_GIB_S)), 3)
        ledger.billing_factor_confirmed = True
        ledger.save()
        print(f"billing factor set to {ledger.billing_factor}")
    else:
        print(__doc__)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
