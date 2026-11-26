from __future__ import annotations

import math
from dataclasses import dataclass

from device.fuel_gauge import DeviceReading

K_E_LO = "smart-e-lo"
K_E_HI = "smart-e-hi"
K_BATCHES = "smart-batches"
K_DEADLINE = "smart-deadline-s"
K_SOC_MIN = "smart-soc-min"
K_PROBE = "smart-probe-samples"
K_PARTIAL = "smart-partial-work"
K_BRULE = "smart-batch-rule"
BATCH_RULES = ("steps", "old", "grow")

MOMENTUM = 0.9


@dataclass(frozen=True)
class Envelope:
    e_lo: int
    e_hi: int
    batches: tuple[int, ...]
    deadline_s: float
    soc_min: float
    probe_samples: int
    partial_work: bool = False
    batch_rule: str = "old"

    def to_config(self) -> dict:
        cfg = {
            K_E_LO: int(self.e_lo),
            K_E_HI: int(self.e_hi),
            K_BATCHES: [int(b) for b in self.batches],
            K_DEADLINE: float(self.deadline_s),
            K_SOC_MIN: float(self.soc_min),
            K_PROBE: int(self.probe_samples),
            K_BRULE: str(self.batch_rule),
        }
        if self.partial_work:
            cfg[K_PARTIAL] = 1
        return cfg

    @classmethod
    def from_config(cls, cfg) -> "Envelope":
        return cls(
            e_lo=int(cfg[K_E_LO]),
            e_hi=int(cfg[K_E_HI]),
            batches=tuple(int(b) for b in cfg[K_BATCHES]),
            deadline_s=float(cfg[K_DEADLINE]),
            soc_min=float(cfg[K_SOC_MIN]),
            probe_samples=int(cfg[K_PROBE]),
            partial_work=bool(int(cfg[K_PARTIAL])) if K_PARTIAL in cfg else False,
            batch_rule=str(cfg[K_BRULE]) if K_BRULE in cfg else "old",
        )

    @staticmethod
    def present_in(cfg) -> bool:
        return K_E_LO in cfg


@dataclass(frozen=True)
class Plan:
    optout: bool
    epochs: int = 0
    batch: int = 0
    time_s: float = 0.0
    soc_end: float = 0.0
    deadline_ok: bool = True
    utility: float = 0.0
    e_target: int = 0
    reason: str = ""
    steps_cap: int = 0


def statistical_utility(loss: float, n_classes: int = 10) -> float:
    if not math.isfinite(loss):
        return 1.0
    return float(min(1.0, max(0.0, loss / math.log(n_classes))))


def headroom(reading: DeviceReading, soc_min: float) -> float:
    if reading.charging:
        return 1.0
    return float(min(1.0, max(0.0, (reading.soc - soc_min) / max(1e-9, 1.0 - soc_min))))


def _energy_ok(reading: DeviceReading, soc_end: float, soc_min: float) -> bool:
    return reading.charging or soc_end >= soc_min


def can_afford_minimum(reading: DeviceReading, env: Envelope, n_samples: int) -> bool:
    if reading.charging:
        return True
    for b in env.batches:
        _, soc_end = reading.predict(env.e_lo, b, n_samples, env.probe_samples)
        if soc_end >= env.soc_min:
            return True
    return False


def accepts_proposal(reading: DeviceReading, env: Envelope, n_samples: int) -> bool:
    return can_afford_minimum(reading, env, n_samples)


def decide(reading: DeviceReading, env: Envelope, n_samples: int,
           loss: float, n_classes: int = 10) -> Plan:
    u = statistical_utility(loss, n_classes)
    h = headroom(reading, env.soc_min)
    span = max(0, env.e_hi - env.e_lo)
    e_target = env.e_lo + int(round(span * u * h))
    p = env.probe_samples

    def options(e: int):
        out = []
        for b in env.batches:
            t, soc_end = reading.predict(e, b, n_samples, p)
            if _energy_ok(reading, soc_end, env.soc_min):
                out.append((b, t, soc_end))
        return out

    for e, b in batch_plan(env, n_samples, e_target, u):
        t, soc_end = reading.predict(e, b, n_samples, p)
        if _energy_ok(reading, soc_end, env.soc_min) and t <= env.deadline_s:
            return Plan(False, e, b, t, soc_end, True, u, e_target,
                        "target" if e == e_target else "ridotto")

    feasible = options(env.e_lo)
    if feasible and env.partial_work:
        for b in sorted(env.batches):
            k = deadline_steps(reading, env, n_samples, b)
            t, soc_end = reading.predict(k / steps_per_epoch(n_samples, b), b, n_samples, p)
            if _energy_ok(reading, soc_end, env.soc_min):
                return Plan(False, env.e_lo, b, t, soc_end, t <= env.deadline_s, u, e_target,
                            "parziale", steps_cap=k)

    if feasible:
        b, t, soc_end = min(feasible, key=lambda o: o[1])
        return Plan(False, env.e_lo, b, t, soc_end, False, u, e_target,
                    "fuori-deadline")
    return Plan(True, utility=u, e_target=e_target, reason="riserva")


def batch_plan(env: Envelope, n_samples: int, e_target: int,
               u: float) -> list[tuple[int, int]]:
    es = range(e_target, env.e_lo - 1, -1)
    bs = sorted(env.batches)
    if env.batch_rule == "old":
        return [(e, b) for b in bs for e in es]
    if env.batch_rule == "grow":
        cap = bs[0] / u if u > 0 else math.inf
        b = max(x for x in bs if x <= cap)
        return [(e, b) for e in es]
    h = env.e_lo * steps_per_epoch(n_samples, bs[0])
    return [(e, next((b for b in reversed(bs) if e * steps_per_epoch(n_samples, b) >= h), bs[0]))
            for e in es]


def steps_per_epoch(n_samples: int, batch: int) -> int:
    return max(1, math.ceil(n_samples / batch))


def deadline_steps(reading: DeviceReading, env: Envelope, n_samples: int, batch: int) -> int:
    spe = steps_per_epoch(n_samples, batch)

    def fits(k: int) -> bool:
        return reading.work_time_s(k / spe, batch, n_samples, env.probe_samples) <= env.deadline_s

    hi = env.e_lo * spe
    if fits(hi):
        return hi
    lo = 0
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if fits(mid):
            lo = mid
        else:
            hi = mid
    return max(1, lo)


def affordable_steps(reading: DeviceReading, env: Envelope, n_samples: int,
                     batch: int, cap: int) -> int:
    if reading.charging:
        return cap
    spe = max(1, math.ceil(n_samples / batch))

    def ok(steps: int) -> bool:
        _, soc_end = reading.predict(steps / spe, batch, n_samples,
                                     env.probe_samples)
        return soc_end >= env.soc_min

    if ok(cap):
        return cap
    lo, hi = 0, cap
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if ok(mid):
            lo = mid
        else:
            hi = mid
    return lo
