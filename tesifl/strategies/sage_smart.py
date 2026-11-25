from __future__ import annotations

import math
from logging import INFO

import numpy as np
import pulp
from flwr.app import Array, ArrayRecord, ConfigRecord, Message, MessageType, RecordDict
from flwr.common import log

from device.battery_model import energy_wh_linear
from device.communication_model import comm_power_w, comm_time_s
from device.compute_model import probe_time_s, training_time_s
from device.communication_model import ctrl_latency_s
from device.constants import CHARGE_C_RATE, RECHARGE_RATE_PER_H, V_NOMINAL
from device.fuel_gauge import DeviceReading
from device.workload import stationary_utilization
from device.world_state import (
    BATTERY_NAMES, battery_delta_soc, effective_profile, idle_power_w,
    training_power_w,
)
from strategies.sage_ablation import PhysicalSAGEAblation
from tesiFL.client_policy import BATCH_RULES, Envelope

_ESTIMATE_STEP_S = 300.0


class PhysicalSAGESmart(PhysicalSAGEAblation):
    def __init__(
        self,
        *args,
        stale_weight: float = 0.3,
        stale_max: int = 25,
        soc_min: float = 0.20,
        epochs_cap_mult: int = 2,
        epochs_min: int = 2,
        partition_sizes: dict[int, int] | None = None,
        deadline_mult: float = 1.0,
        probe_samples: int = 256,
        max_waves: int = 2,
        query_timeout_s: float = 600.0,
        calib_alpha: float = 0.3,
        batch_min: int = 16,
        batch_max: int = 64,
        forecast: bool = True,
        negotiation: bool = True,
        partial_work: bool = True,
        shortlist: float = 1.5,
        load_tau_s: float = 450.0,
        server_battery: str | None = None,
        batch_rule: str = "steps",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if shortlist < 1.0:
            raise ValueError(f"shortlist >= 1, non {shortlist}")
        if batch_rule not in BATCH_RULES:
            raise ValueError(f"batch_rule: una fra {BATCH_RULES}, non {batch_rule!r}")
        self.batch_rule = batch_rule
        if server_battery is not None and server_battery not in BATTERY_NAMES:
            raise ValueError(f"server_battery: una fra {tuple(BATTERY_NAMES)}, non {server_battery!r}")
        self.server_battery = server_battery
        self.forecast = bool(forecast)
        self.negotiation = bool(negotiation)
        self.partial_work = bool(partial_work)
        self.shortlist = float(shortlist)
        self.load_tau_s = float(load_tau_s)
        self.proposal_nonresp = not self.broadcast
        self.stale_w = float(stale_weight)
        self.stale_max = int(stale_max)
        self.soc_min = float(soc_min)
        if not 1 <= int(batch_min) <= int(batch_max):
            raise ValueError(f"batch: serve 1 <= min <= max, non {batch_min}, {batch_max}")
        choices, b = [], int(batch_min)
        while b <= int(batch_max):
            choices.append(b)
            b *= 2
        self.batch_choices = tuple(choices)
        self.epochs_cap = int(epochs_cap_mult) * int(self.epochs)
        self.epochs_min = int(epochs_min)
        self.deadline_mult = float(deadline_mult)
        self.probe_samples = int(probe_samples)
        self.max_waves = max(1, int(max_waves))
        self.query_timeout_s = float(query_timeout_s)
        self.calib_alpha = float(calib_alpha)
        self._n: dict[int, int] = {int(c): int(n) for c, n in
                                   (partition_sizes or {}).items()}
        self.w_busy = self.world.workload.max_utilization()
        wl = self.world.workload
        self.w_enroll = (float(np.mean(list(stationary_utilization(wl.scale).values())))
                         if wl.enabled else 0.0)

        self._last_seen: dict[int, int] = {}
        self._plan: dict[int, Envelope] = {}

        self._subscribed = False
        self._deadline_s: float = math.inf
        self._clock_s = 0.0
        self._soc_rep: dict[int, float] = {}
        self._t_rep: dict[int, float] = {}
        self._w_rep: dict[int, float] = {}
        self._w_sum: dict[int, float] = {}
        self._w_cnt: dict[int, int] = {}
        self._chg_rep: dict[int, bool] = {}
        self._soc_known: dict[int, float] = {}
        self._chg_known: dict[int, bool] = {}
        self._w_known: dict[int, float] = {}
        self._k_cal: dict[int, float] = {}
        self._loss_known: dict[int, float] = {}
        self._bound: dict[int, str] = {}
        self._reports: dict[int, dict] = {}
        self._grid = None
        self._queried: set[int] = set()
        self._overhead_s = 0.0
        self._waves = 0
        self._n_queried = 0
        self._n_refused = 0
        self._info_err = float("nan")
        self._n_deferred = 0
        self._n_noreply = 0
        self._extra_wait_s = 0.0
        self._w_err = float("nan")

    @property
    def _protocol_cols(self) -> bool:
        return self.forecast or self.negotiation or self.proposal_nonresp

    @property
    def variant(self) -> str:
        off = [name for name, on in (
            ("anzianita'", self.stale_w > 0),
            ("previsione del SoC", self.forecast),
            ("negoziazione (broadcast dello stato)" if self.broadcast else "negoziazione",
             self.negotiation),
            ("lavoro parziale", self.partial_work)) if not on]
        return (f"sage-smart-{self.world.battery}"
                + (f" SENZA {', '.join(off)}" if off else "")
                + (f" CON previsione del server su batteria {BATTERY_NAMES[self.server_battery]}"
                   if self.server_battery else ""))

    def configure_train(self, server_round, arrays, config, grid):
        self._grid = grid
        return super().configure_train(server_round, arrays, config, grid)

    def _round_overhead_s(self) -> float:
        return self._overhead_s + super()._round_overhead_s()

    def _sensor_soc(self, cid: int) -> float:
        return float(self.world.snapshot(cid).soc)

    def _subscribe(self) -> None:
        cids = list(self._node_to_cid.values())
        for c in cids:
            self._record(c, self._sensor_soc(c), self.w_enroll, False, 0.0,
                         measured=False)
            self._k_cal[c] = 1.0
        nominal = [
            training_time_s(self.world.profile(c), int(self.epochs),
                            self._n[c], int(self.batch_size))
            for c in cids if self._n.get(c)
        ]
        if nominal:
            self._deadline_s = self.deadline_mult * float(np.median(nominal))
        self._subscribed = True
        log(INFO, "%s: %s client iscritti | deadline %.1f s "
            "(%.2f x mediana nominale, E=%s B=%s) | max %s ondate | batteria %s",
            self.variant, len(cids), self._deadline_s, self.deadline_mult,
            self.epochs, self.batch_size, self.max_waves,
            BATTERY_NAMES[self.world.battery])

    def _record(self, cid: int, soc: float, w: float, chg: bool, t: float,
                measured: bool = True) -> None:
        if measured:
            self._w_sum[cid] = self._w_sum.get(cid, 0.0) + float(w)
            self._w_cnt[cid] = self._w_cnt.get(cid, 0) + 1
        self._soc_rep[cid], self._w_rep[cid] = float(soc), float(w)
        self._chg_rep[cid], self._t_rep[cid] = bool(chg), float(t)
        self._soc_known[cid], self._w_known[cid] = float(soc), float(w)
        self._chg_known[cid] = bool(chg)

    def _forecast(self, cid: int) -> tuple[float, float]:
        soc0 = float(self._soc_rep[cid])
        dt = max(0.0, self._clock_s - self._t_rep.get(cid, 0.0))
        w_rep = float(self._w_rep.get(cid, self.w_enroll))
        w_med = (self._w_sum[cid] / self._w_cnt[cid]) if self._w_cnt.get(cid) else self.w_enroll
        if dt <= 0.0:
            return soc0, w_rep
        prof = self.world.profile(cid)
        bat = self.server_battery or self.world.battery
        recharge = bool(getattr(self.world, "_recharge_enabled", False))
        n = max(1, math.ceil(dt / _ESTIMATE_STEP_S))
        h = dt / n
        if self._chg_rep.get(cid, False):
            m_d, s_d, m_c, s_c = 0.0, soc0, 1.0, soc0
        else:
            m_d, s_d, m_c, s_c = 1.0, soc0, 0.0, soc0
        m_f, s_f = 0.0, 1.0
        for i in range(n):
            w = w_med + (w_rep - w_med) * math.exp(-(i + 0.5) * h / self.load_tau_s)
            p_idle = idle_power_w(prof, w)
            if recharge and m_d > 0.0:
                moved = m_d * (1.0 - math.exp(-RECHARGE_RATE_PER_H * (1.0 - s_d) * h / 3600.0))
                if moved > 0.0:
                    s_c = (m_c * s_c + moved * s_d) / (m_c + moved)
                    m_c, m_d = m_c + moved, m_d - moved
            if m_d > 0.0:
                s_d = max(0.0, s_d + battery_delta_soc(bat, p_idle, h, prof, s_d))
            if m_f > 1e-9:
                s_f = max(0.0, s_f + battery_delta_soc(bat, p_idle, h, prof, s_f))
            if m_c > 0.0:
                s_c = min(1.0, s_c + CHARGE_C_RATE * h / 3600.0)
                if s_c >= 1.0:
                    s_f = (m_f * s_f + m_c) / (m_f + m_c)
                    m_f, m_c = m_f + m_c, 0.0
        soc = m_d * s_d + m_c * s_c + m_f * s_f
        w_now = w_med + (w_rep - w_med) * math.exp(-dt / self.load_tau_s)
        return float(min(1.0, max(0.0, soc))), float(w_now)

    def _refresh_state(self, available: list[int]) -> None:
        for c in self._node_to_cid.values():
            if self.forecast:
                self._soc_known[c], self._w_known[c] = self._forecast(c)
            else:
                self._soc_known[c] = self._soc_rep[c]
                self._w_known[c] = self._w_rep[c]
            self._chg_known[c] = self._chg_rep[c]
        pool = [self._node_to_cid[n] for n in available]
        if pool:
            self._info_err = float(np.mean(
                [abs(self._soc_known[c] - self._sensor_soc(c)) for c in pool]))
            if self._protocol_cols:
                self._w_err = float(np.mean(
                    [abs(self._w_known[c] - self.world.workload.utilization(c))
                     for c in pool]))

    def _propose(self, nodes: list[int],
                 envs: dict[int, Envelope]) -> dict[int, Message]:
        if self.proposal_nonresp:
            lost = {n for n in nodes if self._bcast_rng.random() < self.bcast_nonresp}
            if lost:
                self._n_noreply += len(lost)
                self._extra_wait_s += max(0.0, self.bcast_timeout_s - ctrl_latency_s())
                nodes = [n for n in nodes if n not in lost]
        msgs: list[Message] = []
        for n in nodes:
            cid = self._node_to_cid[n]
            cfg = ConfigRecord({"cid": int(cid)})
            for key, val in envs[cid].to_config().items():
                cfg[key] = val
            rec = RecordDict({self.configrecord_key: cfg,
                              "device": self._device_record(cid)})
            msgs.extend(self._construct_messages(rec, [int(n)], MessageType.QUERY))
        self._queried.update(nodes)
        self._n_queried += len(nodes)
        if self._grid is None or not msgs:
            return {}
        out: dict[int, Message] = {}
        for r in self._grid.send_and_receive(msgs, timeout=self.query_timeout_s):
            if not r.has_content():
                continue
            n = int(r.metadata.src_node_id)
            cid = self._node_to_cid.get(n)
            if cid is None:
                continue
            self._record(cid, float(self._rep(r, "smart-soc", self._soc_known[cid])),
                         float(self._rep(r, "smart-workload", self.w_busy)),
                         bool(int(self._rep(r, "smart-charging", 0))),
                         self._clock_s)
            out[n] = r
        return out

    def _charge_control(self, selected: list[int]) -> None:
        sel = set(selected)
        for n in self._queried:
            self.world.apply_control(self._node_to_cid[n],
                                     merged=(n in sel) or self.broadcast)

    def _resource(self, cid: int) -> float:
        return self._soc_known.get(cid, float(self.world.initial_soc(cid)))

    def _n_of(self, cid: int) -> int | None:
        if cid in self._n:
            return self._n[cid]
        return int(np.median(list(self._n.values()))) if self._n else None

    def _batch_for(self, cid: int) -> int:
        return int(self.batch_choices[0])

    def _staleness(self, cid: int, server_round: int) -> float:
        last = self._last_seen.get(cid)
        age = server_round if last is None else server_round - last
        return float(min(1.0, age / max(1, self.stale_max)))

    def _select_clients(
        self, available: list[int], k: int, server_round: int
    ) -> list[int]:
        if not self._subscribed:
            self._subscribe()
        self._reports = {}
        self._bound = {}
        self._queried = set()
        self._overhead_s = 0.0
        self._waves = 0
        self._n_queried = 0
        self._n_refused = 0
        self._n_deferred = 0
        self._n_noreply = 0
        self._extra_wait_s = 0.0
        if not k or not available:
            log(INFO, "%s round %s: pool vuoto", self.variant, server_round)
            return []
        self._refresh_state(available)
        cid_of = self._node_to_cid
        pool = available
        m = len(pool)
        hi = min(self.k + server_round - 1,
                 int(np.floor(self.max_frac * max(m, self._n_alive))), m)
        lo = self.k

        score = {
            n: self.a * self._resource(cid_of[n])
            + self.b * self._divergence(cid_of[n])
            + self.stale_w * self._staleness(cid_of[n], server_round)
            for n in pool
        }

        prob = pulp.LpProblem("SAGE_Smart_Select", pulp.LpMaximize)
        y = [pulp.LpVariable(f"y_{i}", cat="Binary") for i in range(m)]
        prob += pulp.lpSum(y[i] * score[pool[i]] for i in range(m))
        prob += pulp.lpSum(y) >= lo
        prob += pulp.lpSum(y) <= hi
        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        selected = [pool[i] for i in range(m) if int(pulp.value(y[i]) or 0) == 1]
        if not selected:
            self._charge_control([])
            log(INFO, "%s round %s: ILP infeasible con pool %s (lo=%s, hi=%s): "
                "pool esaurito", self.variant, server_round, m, lo, hi)
            return []

        ranking = sorted(pool, key=lambda n: -score[n])
        if self.negotiation:
            selected = self._negotiate_pick(ranking, len(selected), server_round)
        else:
            selected = ranking[:len(selected)]
            if self.proposal_nonresp:
                lost = {n for n in selected
                        if self._bcast_rng.random() < self.bcast_nonresp}
                self._n_noreply += len(lost)
                selected = [n for n in selected if n not in lost]
        if not selected:
            self._charge_control([])
            log(INFO, "%s round %s: nessun candidato ha accettato in %s "
                "ondate: pool esaurito", self.variant, server_round,
                self._waves)
            return []

        self._charge_control(selected)
        self._overhead_s = self._waves * ctrl_latency_s() + self._extra_wait_s
        for n in selected:
            self._last_seen[cid_of[n]] = server_round
        self._plan = {cid_of[n]: self._assign_work(cid_of[n]) for n in selected}

        E = [env.e_hi for env in self._plan.values()]
        never = sum(1 for c in cid_of.values() if c not in self._last_seen)
        log(INFO, "%s round %s: pool %s, %s selezionati (cap %s) | "
            "E_hi %s-%s (medio %.1f) | mai visti %s | interrogati %s, "
            "rifiuti %s, ondate %s%s",
            self.variant, server_round, m, len(selected), hi, min(E), max(E),
            float(np.mean(E)), never, self._n_queried, self._n_refused,
            self._waves,
            (f" | rinviati {self._n_deferred}, non raggiunti {self._n_noreply}, "
             f"errore w {self._w_err:.3f}") if self._protocol_cols else "")
        return selected

    def _fresh_score(self, cid: int, server_round: int) -> float:
        soc = 1.0 if self._chg_rep.get(cid, False) else float(self._soc_rep[cid])
        base = (self.a * soc + self.b * self._divergence(cid)
                + self.stale_w * self._staleness(cid, server_round))
        return base * (1.0 - float(self._w_rep.get(cid, self.w_enroll)))

    def _negotiate_pick(self, ranking: list[int], slots: int,
                        server_round: int) -> list[int]:
        cid_of = self._node_to_cid
        accepted: list[int] = []
        asked: set[int] = set()
        need = slots
        while need > 0 and self._waves < self.max_waves:
            batch = [n for n in ranking if n not in asked][: int(math.ceil(self.shortlist * need))]
            if not batch:
                break
            self._waves += 1
            asked.update(batch)
            envs = {cid_of[n]: self._assign_work(cid_of[n]) for n in batch}
            replies = self._propose(batch, envs)
            for n in batch:
                r = replies.get(n)
                if r is None:
                    continue
                if int(self._rep(r, "smart-accept", 0)) == 1:
                    accepted.append(n)
                else:
                    self._n_refused += 1
                    self._last_seen[cid_of[n]] = server_round
            need = slots - len(accepted)
        picked = sorted(accepted,
                        key=lambda n: -self._fresh_score(cid_of[n], server_round))[:slots]
        self._n_deferred = len(accepted) - len(picked)
        return picked

    def _mark_half(self, selected: list[int]) -> None:
        self._half = set()

    def _batches_for(self, cid: int) -> tuple[int, ...]:
        return self.batch_choices

    def _assign_work(self, cid: int) -> Envelope:
        prof = self.world.profile(cid)
        batches = self._batches_for(cid)
        b_ref = self._batch_for(cid)
        peff = effective_profile(prof, self._w_known.get(cid, self.w_enroll))
        e_lo, e_cap = int(self.epochs_min), int(self.epochs_cap)
        n = self._n_of(cid)

        e_dl = e_cap
        if n and math.isfinite(self._deadline_s):
            t_ep = min(training_time_s(peff, 1, n, b) for b in batches)
            e_dl = int(self._deadline_s // max(t_ep, 1e-9))

        e_en = e_cap
        if n and not self._chg_known.get(cid, False):
            cap_wh = prof.battery_capacity_mah / 1000.0 * V_NOMINAL
            k = self._k_cal.get(cid, 1.0)
            d_ep = k * energy_wh_linear(
                training_power_w(prof), training_time_s(peff, 1, n, b_ref)
            ) / cap_wh
            d_up = k * energy_wh_linear(comm_power_w(), comm_time_s()) / cap_wh
            spare = self._resource(cid) - self.soc_min - d_up
            e_en = int(spare // d_ep) if d_ep > 0 else e_cap

        e_hi = int(np.clip(min(e_cap, e_dl, e_en), e_lo, e_cap))
        if e_hi >= e_cap:
            self._bound[cid] = "cap"
        elif e_dl <= e_en:
            self._bound[cid] = "deadline"
        else:
            self._bound[cid] = "energy"
        return Envelope(
            e_lo=e_lo, e_hi=e_hi, batches=batches,
            deadline_s=float(self._deadline_s) if math.isfinite(self._deadline_s)
            else 1e12,
            soc_min=float(self.soc_min), probe_samples=self.probe_samples,
            partial_work=self.partial_work, batch_rule=self.batch_rule,
        )

    def _per_client_config(
        self, cid: int, cfg: ConfigRecord, server_round: int
    ) -> None:
        env = self._plan.get(cid)
        if env is None:
            return
        for key, val in env.to_config().items():
            cfg[key] = val
        cfg["epochs"] = int(env.e_hi)
        cfg["batch-size"] = int(self._batch_for(cid))

    def _device_record(self, cid: int) -> ConfigRecord | None:
        rec = DeviceReading.from_world(self.world, cid).to_record()
        rec["dataset-size"] = int(self._n_of(cid) or 0)
        return ConfigRecord(rec)

    def _rep(self, reply: Message, key: str, default=None):
        return self._metric(reply, key, default=default)

    def _is_optout(self, reply: Message) -> bool:
        return int(self._rep(reply, "smart-optout", 0) or 0) == 1

    def _charge_communication(self, cid: int, reply: Message) -> tuple[float, float]:
        if not self._is_optout(reply):
            return super()._charge_communication(cid, reply)
        e_dl, l_dl = self.world.apply_download(cid)
        e_ct, l_ct = self.world.apply_control(cid, merged=False)
        return e_dl + e_ct, l_dl + l_ct

    def _work_done(self, cid: int, n_ex: int, reply: Message) -> tuple[float, int]:
        b = int(self._rep(reply, "smart-batch", self._batch_for(cid)))
        steps = int(self._rep(reply, "smart-steps", 0))
        probe = int(self._rep(reply, "smart-probe", 0))
        e_eff = steps / math.ceil(n_ex / b) if steps > 0 and n_ex > 0 else 0.0
        dt = (self.world.round_duration_s(cid, epochs=e_eff, batch_size=b,
                                          dataset_size_local=n_ex)
              + self.world.probe_duration_s(cid, probe))
        return dt, b

    def _in_envelope(self, cid: int, reply: Message) -> bool:
        env = self._plan.get(cid)
        if env is None:
            return False
        e = int(self._rep(reply, "smart-epochs", 0))
        b = int(self._rep(reply, "smart-batch", 0))
        return env.e_lo <= e <= env.e_hi and b in env.batches

    def _aggregatable(self, replies: list[Message]) -> list[Message]:
        out = []
        for r in replies:
            if not r.has_content():
                out.append(r)
                continue
            if self._is_optout(r):
                continue
            cid = self._node_to_cid.get(int(r.metadata.src_node_id))
            if cid is not None and not self._in_envelope(cid, r):
                continue
            out.append(r)
        return out

    def _steps(self, reply: Message) -> float:
        if self._is_optout(reply):
            return 0.0
        return float(self._rep(reply, "smart-steps", 0))

    def _on_round_result(
        self, cid: int, n_ex: int, dt_s: float, e_wh: float, e_lin_wh: float,
        reply: Message, server_round: int,
    ) -> None:
        super()._on_round_result(cid, n_ex, dt_s, e_wh, e_lin_wh, reply,
                                 server_round)
        optout = self._is_optout(reply)
        soc0 = float(self._rep(reply, "smart-soc-start", self._resource(cid)))
        soc1 = float(self._rep(reply, "smart-soc-final", soc0))
        chg = bool(int(self._rep(reply, "smart-charging", 0)))
        w = float(self._rep(reply, "smart-workload", self.w_busy))
        b = int(self._rep(reply, "smart-batch", self._batch_for(cid)))
        steps = int(self._rep(reply, "smart-steps", 0))
        probe = int(self._rep(reply, "smart-probe", 0))

        self._record(cid, soc1, w, chg, self._clock_s + self._overhead_s + dt_s)
        if n_ex > 0:
            self._n[cid] = int(n_ex)
        loss = self._rep(reply, "smart-loss-pre", None)
        if loss is not None:
            self._loss_known[cid] = float(loss)

        if not chg and steps > 0 and n_ex > 0:
            prof = self.world.profile(cid)
            peff = effective_profile(prof, w)
            e_eff = steps / math.ceil(n_ex / b)
            t = training_time_s(peff, e_eff, n_ex, b) + probe_time_s(peff, probe)
            cap_wh = prof.battery_capacity_mah / 1000.0 * V_NOMINAL
            lin = (energy_wh_linear(training_power_w(prof), t)
                   + energy_wh_linear(comm_power_w(), comm_time_s())) / cap_wh
            drop = soc0 - soc1
            if lin > 0 and drop > 0:
                a = self.calib_alpha
                self._k_cal[cid] = (1 - a) * self._k_cal.get(cid, 1.0) + a * drop / lin

        self._reports[cid] = {
            "optout": optout,
            "epochs": int(self._rep(reply, "smart-epochs", 0)),
            "e_eff": (steps / math.ceil(n_ex / b)) if steps > 0 and n_ex > 0 else 0.0,
            "batch": b,
            "steps": steps,
            "budget_hit": int(self._rep(reply, "smart-budget-hit", 0)),
            "deadline_ok": int(self._rep(reply, "smart-deadline-ok", 1)),
            "charging": chg,
            "loss_pre": float(loss) if loss is not None else float("nan"),
            "loss_post": float(self._rep(reply, "smart-loss-post", float("nan"))),
            "workload": w,
            "soc_err": abs(float(self.world.snapshot(cid).soc) - soc1),
            "rejected": (not optout) and not self._in_envelope(cid, reply),
            "partial": int(self._rep(reply, "smart-partial", 0)),
        }
        if self._reports[cid]["rejected"]:
            log(INFO, "%s round %s: update del client %s fuori envelope, "
                "scartato", self.variant, server_round, cid)

    def _policy_stats(self) -> dict:
        rep = list(self._reports.values())
        done = [r for r in rep if not r["optout"]]
        envs = list(self._plan.values())

        def mean(xs):
            xs = [x for x in xs if x == x]
            return float(np.mean(xs)) if xs else float("nan")

        st = {
            "mean_epochs": mean([r["e_eff"] for r in done]),
            "max_epochs": float(max((r["e_eff"] for r in done), default=0.0)),
            "mean_batch": mean([r["batch"] for r in done]),
            "n_charging_sel": int(sum(1 for r in rep if r["charging"])),
            "smart_min_epochs": float(min((r["e_eff"] for r in done), default=0.0)),
            "smart_mean_e_hi": mean([e.e_hi for e in envs]),
            "smart_n_bound_energy": int(sum(1 for v in self._bound.values() if v == "energy")),
            "smart_n_bound_deadline": int(sum(1 for v in self._bound.values() if v == "deadline")),
            "smart_n_optout": int(sum(1 for r in rep if r["optout"])),
            "smart_n_rejected": int(sum(1 for r in rep if r["rejected"])),
            "smart_n_deadline_miss": int(sum(1 for r in done if not r["deadline_ok"])),
            "smart_n_budget_hit": int(sum(1 for r in done if r["budget_hit"])),
            "smart_mean_loss_pre": mean([r["loss_pre"] for r in rep]),
            "smart_mean_loss_post": mean([r["loss_post"] for r in done]),
            "smart_mean_workload": mean([r["workload"] for r in rep]),
            "smart_soc_err": float(max((r["soc_err"] for r in rep), default=0.0)),
            "smart_mean_k": mean(list(self._k_cal.values())),
            "smart_deadline_s": float(self._deadline_s),
            "smart_info_err": float(self._info_err),
            "smart_n_queried": int(self._n_queried),
            "smart_n_refused": int(self._n_refused),
            "smart_waves": int(self._waves),
            "smart_overhead_s": float(self._overhead_s),
        }
        if self.partial_work:
            st["smart_n_partial"] = int(sum(1 for r in done if r.get("partial")))
        if self._protocol_cols:
            st["smart_n_deferred"] = int(self._n_deferred)
            st["smart_n_noreply"] = int(self._n_noreply)
            st["smart_w_err"] = float(self._w_err)
        log(INFO, "%s: %s risposte | E eseguite %.1f (min %.2f, max %.2f) "
            "| E_hi medio %.1f (energia %s, deadline %s) | opt-out %s | "
            "fuori deadline %s | errore SoC %.1e | errore stato noto %.3f",
            self.variant, len(rep), st["mean_epochs"], st["smart_min_epochs"],
            st["max_epochs"], st["smart_mean_e_hi"], st["smart_n_bound_energy"],
            st["smart_n_bound_deadline"], st["smart_n_optout"],
            st["smart_n_deadline_miss"], st["smart_soc_err"],
            st["smart_info_err"])
        return st

    def aggregate_train(self, server_round, replies):
        arrays, metrics = super().aggregate_train(server_round, replies)
        self._clock_s += self._last_round_s

        if self._plan:
            self._round_stats.setdefault(int(server_round), {}).update(
                self._policy_stats())

        if arrays is None or self._global_arrays is None:
            return arrays, metrics

        keys = list(self._global_arrays.keys())
        rows = []
        wts = []
        for r in self._aggregatable(replies):
            if not r.has_content():
                continue
            if self._node_to_cid.get(int(r.metadata.src_node_id)) is None:
                continue
            n = float(self._metric(r, "num-examples", "num_examples", default=0))
            tau = self._steps(r)
            if n <= 0 or tau <= 0:
                continue
            key = sorted(r.content.array_records)[0]
            rec = r.content.array_records[key]
            rows.append((n, tau, [rec[k].numpy() for k in keys]))
            wts.append(float(self._metric(r, "smart-weight", default=n)))

        if len(rows) < 2:
            return arrays, metrics

        tot = sum(n for n, _, _ in rows)
        p = [n / tot for n, _, _ in rows]
        tau_eff = sum(pi * tau for pi, (_, tau, _) in zip(p, rows))
        taus = [tau for _, tau, _ in rows]
        self._round_stats.setdefault(int(server_round), {}).update({
            "tau_min": int(min(taus)),
            "tau_max": int(max(taus)),
            "tau_eff": float(tau_eff),
            "tau_spread": float(max(taus) / max(1.0, min(taus))),
        })
        log(INFO, "round %s: FedAvg su %s client | tau %s-%s, tau_eff %.1f, "
            "spread %.1f", server_round, len(rows), int(min(taus)),
            int(max(taus)), tau_eff, max(taus) / max(1.0, min(taus)))

        if self.partial_work and any(w < n for w, (n, _, _) in zip(wts, rows)):
            tot_w = sum(wts)
            q = [w / tot_w for w in wts]
            merged = ArrayRecord()
            for j, k in enumerate(keys):
                ref = self._global_arrays[k].numpy()
                avg = np.asarray(sum(qi * row[2][j].astype(np.float64) for qi, row in zip(q, rows)),
                                 dtype=np.float64).reshape(ref.shape)
                if np.issubdtype(ref.dtype, np.integer):
                    avg = np.rint(avg)
                merged[k] = Array(avg.astype(ref.dtype))
            self._round_stats.setdefault(int(server_round), {}).update({
                "smart_weight_frac": float(tot_w / sum(n for n, _, _ in rows)),
            })
            return merged, metrics

        return arrays, metrics
