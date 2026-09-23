"""Cloud spend control: a cost model with a Kalman-filtered rate, a chance-constrained budget gate
and a persistent ledger.

Cost of a batch = rows × (seconds per row) × (price per container-second) × billing factor, where
the seconds-per-row rate is uncertain and is tracked with a scalar Kalman filter (a new pilot run is a
noisy measurement of it). A launch is authorised only if the **upper** ``CONFIDENCE`` bound of its
cost fits the remaining budget (a chance constraint: P(cost > remaining) ≤ 1 − CONFIDENCE).

``billing_factor`` maps list-price estimates onto the real bill. Modal bills at list price, so it is 1.0.
(An earlier session read "150" — the studio-all run's 150 shard *tasks* — as a $150 bill and set it to
26.4; there was no such bill. The account limit is $30 a month; the cap here is $20, the rest padding.)
"""

from __future__ import annotations

import datetime
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

LEDGER = Path(__file__).resolve().parent / "ledger.json"
# Modal list prices (USD per second); verify at modal.com/pricing and adjust billing_factor instead.
CPU_CORE_S = 0.0000131
MEM_GIB_S = 0.00000222
GPU_S = {"T4": 0.000164, "L4": 0.000222, "A10G": 0.000306}
CONFIDENCE_Z = 1.645  # one-sided 95 %


@dataclass
class RateEstimate:
    """Scalar Kalman state for seconds-per-row of one workload (x̂, P) with process noise Q."""

    mean: float
    var: float
    q: float = 0.05  # (s/row)² drift per update: code or model changes move the rate

    def update(self, measured: float, r: float) -> None:
        p = self.var + self.q
        k = p / (p + r)
        self.mean += k * (measured - self.mean)
        self.var = (1 - k) * p

    def upper(self) -> float:
        return self.mean + CONFIDENCE_Z * math.sqrt(self.var + self.q)


@dataclass
class Ledger:
    cap_usd: float = 20.0
    billing_factor: float = 1.0
    billing_factor_confirmed: bool = True
    rates: dict[str, RateEstimate] = field(default_factory=dict)
    entries: list[dict[str, object]] = field(default_factory=list)

    # -- persistence -----------------------------------------------------------------------------
    @classmethod
    def load(cls, path: Path = LEDGER) -> Ledger:
        if not path.exists():
            return cls.default()
        d = json.loads(path.read_text())
        d["rates"] = {k: RateEstimate(**v) for k, v in d.get("rates", {}).items()}
        return cls(**d)

    @classmethod
    def default(cls) -> Ledger:
        # studio-all (2026-09-23): 143 699 container-s / 37 411 rows, cpu=2, 6 GiB; shard spread gives the variance
        return cls(rates={"engine_cpu2": RateEstimate(mean=3.84, var=0.6 ** 2)})

    def save(self, path: Path = LEDGER) -> None:
        d = asdict(self)
        path.write_text(json.dumps(d, indent=1) + "\n")

    # -- cost model ------------------------------------------------------------------------------
    @staticmethod
    def price_per_s(cpu: float, mem_gib: float, gpu: str | None = None) -> float:
        return cpu * CPU_CORE_S + mem_gib * MEM_GIB_S + (GPU_S[gpu] if gpu else 0.0)

    def estimate(self, workload: str, rows: int, cpu: float = 2.0, mem_gib: float = 6.0,
                 gpu: str | None = None) -> dict[str, float]:
        rate = self.rates[workload]
        unit = self.price_per_s(cpu, mem_gib, gpu) * self.billing_factor
        return {"expected_usd": round(rows * rate.mean * unit, 2), "upper_usd": round(rows * rate.upper() * unit, 2),
                "container_hours": round(rows * rate.mean / 3600, 1)}

    @property
    def spent(self) -> float:
        return float(sum(float(e.get("actual_usd", e.get("authorised_usd", 0.0))) for e in self.entries))  # type: ignore[arg-type]

    @property
    def remaining(self) -> float:
        return self.cap_usd - self.spent

    # -- control ---------------------------------------------------------------------------------
    def authorise(self, tag: str, workload: str, rows: int, **resources: object) -> dict[str, object]:
        """Record and return the authorisation, or raise if the upper cost bound breaks the cap."""
        est = self.estimate(workload, rows, **resources)  # type: ignore[arg-type]
        if est["upper_usd"] > self.remaining:
            raise BudgetExceeded(f"{tag}: upper cost ${est['upper_usd']} > remaining ${self.remaining:.2f} "
                                 f"(cap ${self.cap_usd}, factor {self.billing_factor}"
                                 f"{'' if self.billing_factor_confirmed else ', unconfirmed'})")
        entry = {"tag": tag, "workload": workload, "rows": rows, "authorised_usd": est["upper_usd"],
                 "expected_usd": est["expected_usd"], "at": datetime.datetime.now(datetime.UTC).isoformat()}
        self.entries.append(entry)
        return entry

    def max_rows(self, workload: str, cpu: float = 2.0, mem_gib: float = 6.0, gpu: str | None = None) -> int:
        """Largest batch whose upper cost bound fits the remaining budget."""
        per_row = self.rates[workload].upper() * self.price_per_s(cpu, mem_gib, gpu) * self.billing_factor
        return max(0, int(self.remaining / per_row))

    def observe(self, workload: str, container_s: float, rows: int, shard_secs: list[float] | None = None) -> None:
        """Feed a finished run back: its seconds/row is a measurement with variance from the shard spread."""
        measured = container_s / max(1, rows)
        if shard_secs and len(shard_secs) > 1:
            m = sum(shard_secs) / len(shard_secs)
            var_shard = sum((s - m) ** 2 for s in shard_secs) / (len(shard_secs) - 1)
            r = var_shard / (rows / len(shard_secs)) ** 2 / len(shard_secs)
        else:
            r = (0.25 * measured) ** 2
        self.rates.setdefault(workload, RateEstimate(measured, r)).update(measured, r)


class BudgetExceeded(RuntimeError):
    pass
