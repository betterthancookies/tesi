# [LIMITE DICHIARATO] all'interno del singolo round la potenza resta costante

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from device.battery_model import (
    delta_soc, energy_wh_from_delta_soc, energy_wh_linear,
)
from device.battery_model_peukert import delta_soc as peukert_delta_soc
from device.communication_model import (
    comm_power_w, comm_time_s, ctrl_power_w, ctrl_time_s,
)
from device.compute_model import probe_time_s, training_time_s
from device.constants import (
    DCDC_EFFICIENCY,
    BATCH_SIZE_REF,
    CHARGE_C_RATE,
    IDLE_POWER_W,
    INITIAL_SOC_MEAN,
    INITIAL_SOC_RANGE,
    INITIAL_SOC_STD,
    POWER_BATCH_MULT_MAX,
    POWER_BATCH_MULT_MIN,
    POWER_EXPONENT_BATCH,
    RECHARGE_AVAILABLE_DEFAULT,
    RECHARGE_PROB_BASE,
    TRAIN_POWER_W,
    V_NOMINAL,
)
from device.device_profile import DeviceProfile, generate_profiles
from device.workload import WorkloadModel


# [B] funzioni di modulo, non metodi: le usa anche il CLIENT di sage_smart
# (device/fuel_gauge.py) per prevedere il proprio consumo. Se client e mondo
# avessero due copie della stessa formula, prima o poi divergerebbero.
def effective_profile(p: DeviceProfile, w: float) -> DeviceProfile:
    """Profilo col throughput ridotto dal workload w."""
    if w <= 0.0:
        return p
    return DeviceProfile(
        cid=p.cid, battery_capacity_mah=p.battery_capacity_mah,
        macs_per_s=p.macs_per_s * max(1e-3, 1.0 - w),
        peukert_n=p.peukert_n, tier=p.tier,
    )


def training_power_w(batch_size: int) -> float:
    """P(B) = P0 * (B/B_ref)^gamma, limitata in [mult_min, mult_max] * P0."""
    mult = (batch_size / BATCH_SIZE_REF) ** POWER_EXPONENT_BATCH
    mult = min(max(mult, POWER_BATCH_MULT_MIN), POWER_BATCH_MULT_MAX)
    return TRAIN_POWER_W * mult


@dataclass(frozen=True)
class WorldStateSnapshot:
    cid: int
    soc: float
    available: bool
    recharging: bool   # sempre False: niente ricarica. Tenuto per le strategie.
    failed: bool


class WorldState:
    """Mondo fisico: tre tier di device, workload concorrente, scarica Peukert.

    Un device che tocca SoC 0 e' morto in modo permanente e non torna
    disponibile.

    CONSUMO FRA UN ROUND E L'ALTRO (modello di ESCS, Sez. 4.2)
    La durata di un round e' il training del selezionato PIU' LENTO. Chi non e'
    selezionato sta in idle per tutta quella durata; chi e' selezionato paga il
    training piu' l'idle per il tempo in cui aspetta gli altri. Cosi' non serve
    inventare un intervallo fra i round: lo determina la simulazione.
    [B] ne discende un effetto che nessuno dei due algoritmi modella: il device
    veloce finisce prima e poi brucia energia aspettando i lenti.

    WORKLOAD
    w e' la frazione di calcolo occupata da altre applicazioni, e agisce su due
    grandezze senza coefficienti nuovi:
        throughput disponibile = (1 - w) * macs_per_s
        potenza fra i round    = P_idle + w * (P_train - P_idle)
    [B] il workload NON tocca la potenza di training: durante il training il
    device e' gia' a pieno carico, il lavoro concorrente lo rallenta (si
    contende la CPU) ma non aggiunge un consumo sopra il massimo.

    ABLAZIONE
    Con tiers_enabled=False, workload_enabled=False e idle_enabled=False il
    mondo torna esattamente a quello delle campagne precedenti. E' il test che
    separa l'effetto dell'eterogeneita' da quello della politica.

    DUE CONTABILITA' ENERGETICHE, tenute separate di proposito:
      - Peukert:  Wh = V * C_rated * |dSoC|, la carica che la batteria paga
                  davvero. E' quella dei contatori globali (stats, CSV).
      - lineare:  Wh = P/eta * dt, potenza per tempo. E' quella che vedono
                  SAGE ed ESCS originali, che sottraggono i Wh spesi dalla
                  capacita' nominale: sottostimano il consumo reale.
    apply_round/apply_communication restituiscono entrambe, cosi' una
    strategia "orig" puo' usare la lineare e il confronto con la variante SoC
    misura davvero l'errore del modello lineare, non altro.

    TRE MODELLI DI BATTERIA, per il confronto lin / peuk / nm
      nm    (datasheet, battery_model.py) e' la batteria VERA: _soc.
      lin   fuel gauge lineare, soc_linear():
                SoC_lin <- SoC_lin - (P/eta * dt) / (V_nom * C_nom)
      peuk  fuel gauge di Peukert, soc_peukert() (battery_model_peukert.py):
                SoC_peuk <- SoC_peuk - dt / (C' / I^n)
    I due gauge sono il SoC che il device CREDEREBBE di avere se stimasse la
    carica con quel modello. Sono aggiornati a OGNI scarica (training,
    comunicazione, idle) con la stessa potenza e la stessa durata della
    batteria vera, e a ogni carica; stesso clamp in [0, 1]. A fine carica (SoC
    vero = 1) si riallineano a 1, come fa un fuel gauge vero quando rileva la
    carica completa.
    [B] differiscono dal SoC vero SOLO per il modello di batteria: e' cio'
    che serve ai bracci peuk (sage_peuk, sage_smart_peuk, escs_*_peuk) e a
    sage_smart_lin per isolare l'errore del modello. Le varianti lineari di
    SAGE ed ESCS invece tengono una contabilita' propria che ignora idle e
    ricarica, come nei loro paper.
    """

    def __init__(self, profiles: list[DeviceProfile], seed: int,
                 workload: WorkloadModel | None = None,
                 idle_enabled: bool = True,
                 recharge_enabled: bool = False,
                 recharge_available: bool = RECHARGE_AVAILABLE_DEFAULT) -> None:
        self._profiles = {p.cid: p for p in profiles}
        self._rng = np.random.default_rng(seed)
        self._idle_enabled = bool(idle_enabled)
        self._recharge_enabled = bool(recharge_enabled)
        # True = modalita' (a): in carica il device resta selezionabile e si
        # allena a costo zero. False = (b): e' fuori dalla federazione.
        self._recharge_available = bool(recharge_available)
        # [B] seed diverso da quello del SoC: con lo stesso, la permutazione
        # dei profili di workload sarebbe allineata a quella dei tier.
        self.workload = workload if workload is not None else WorkloadModel(
            n_clients=len(profiles), seed=seed + 7919, enabled=False)
        # unica asimmetria fisica iniziale fra device altrimenti identici
        self._soc: dict[int, float] = {
            cid: self._sample_initial_soc() for cid in self._profiles
        }
        # [B] il SoC iniziale e' un dato noto anche ai paper di riferimento
        # (SAGE: E_i iniziale da gaussiana; ESCS: battery iniziale 30-100%).
        # Le strategie "orig" partono da qui, non da 1.
        self._soc0: dict[int, float] = dict(self._soc)
        # fuel gauge lineare: parte dallo stesso SoC, vedi docstring
        self._soc_lin: dict[int, float] = dict(self._soc)
        self._soc_peuk: dict[int, float] = dict(self._soc)
        self._failed: dict[int, bool] = {cid: False for cid in self._profiles}
        self._energy_wh: dict[int, float] = {cid: 0.0 for cid in self._profiles}
        self._energy_comm_wh: dict[int, float] = {cid: 0.0 for cid in self._profiles}
        # messaggi di controllo (stato, proposte): solo sage_smart li usa
        self._energy_ctrl_wh: dict[int, float] = {cid: 0.0 for cid in self._profiles}
        self._energy_lin_wh: dict[int, float] = {cid: 0.0 for cid in self._profiles}
        self._energy_idle_wh: dict[int, float] = {cid: 0.0 for cid in self._profiles}
        # energia presa dalla RETE durante le cariche: non pesa sulla batteria
        # ma va contabilizzata, altrimenti allenare un device in carica
        # sembrerebbe gratuito in assoluto e non solo per la sua batteria.
        self._energy_grid_wh: dict[int, float] = {cid: 0.0 for cid in self._profiles}
        self._charging: dict[int, bool] = {cid: False for cid in self._profiles}
        self._n_recharges: dict[int, int] = {cid: 0 for cid in self._profiles}

    def _sample_initial_soc(self) -> float:
        """SoC iniziale: normale troncata, spostata verso la carica piena.

        [B] l'uniforme su [0.3, 1.0] metteva il 29% dei client sotto il 50%
        di carica gia' al round 1, cioe' una popolazione mediamente scarica
        prima ancora di iniziare. Con mu=0.75 quella quota scende al 5%: resta
        una coda bassa -- che serve, perche' e' cio' che gli algoritmi
        battery-aware devono saper distinguere -- ma il caso tipico diventa un
        device carico, che e' la situazione realistica.

        [B] RIFIUTO IN CICLO, NON CLIP. Un clip agli estremi accumulerebbe
        massa su 0.30 e 1.00, creando due picchi artificiali proprio dove la
        soglia di riserva e la carica piena contano di piu'.

        [B] usa self._rng, quindi la popolazione iniziale dipende dal seed
        come prima: seed diversi danno popolazioni diverse, lo stesso seed
        la stessa popolazione.
        """
        lo, hi = INITIAL_SOC_RANGE
        for _ in range(10000):
            x = float(self._rng.normal(INITIAL_SOC_MEAN, INITIAL_SOC_STD))
            if lo <= x <= hi:
                return x
        # irraggiungibile con parametri sensati; se mu cade fuori dal
        # troncamento e' meglio accorgersene che restare in ciclo
        raise RuntimeError(
            f"SoC iniziale: nessun campione in [{lo}, {hi}] con "
            f"mu={INITIAL_SOC_MEAN}, sigma={INITIAL_SOC_STD}"
        )

    # ------------------------------------------------------------- lettura
    def snapshot(self, cid: int) -> WorldStateSnapshot:
        """[B] `available` dipende dalla modalita' di ricarica:
        (a) recharge_available=True  -> in carica il device resta selezionabile
            e si allena a costo zero (il telefono in carica funziona);
        (b) recharge_available=False -> in carica esce dalla federazione, che
            per quel round lavora su N-1 client.
        """
        failed = self._failed[cid]
        charging = self._charging[cid]
        available = (not failed) and (self._recharge_available or not charging)
        return WorldStateSnapshot(
            cid=cid, soc=self._soc[cid], available=available,
            recharging=charging, failed=failed,
        )

    def profile(self, cid: int) -> DeviceProfile:
        return self._profiles[cid]

    def initial_soc(self, cid: int) -> float:
        return self._soc0[cid]

    def soc_linear(self, cid: int) -> float:
        """SoC secondo il fuel gauge LINEARE del device (vedi docstring)."""
        return self._soc_lin[cid]

    def soc_peukert(self, cid: int) -> float:
        """SoC secondo il fuel gauge di PEUKERT del device (vedi docstring)."""
        return self._soc_peuk[cid]

    def capacity_wh(self, cid: int) -> float:
        """Capacita' nominale in Wh, V_nom * C_nom: il denominatore lineare."""
        return self._profiles[cid].battery_capacity_mah / 1000.0 * V_NOMINAL

    def energy_wh(self, cid: int) -> float:
        """Energia cumulata (train + comm) del client, in Wh (Peukert)."""
        return (self._energy_wh[cid] + self._energy_comm_wh[cid]
                + self._energy_ctrl_wh[cid])

    def energy_lin_wh(self, cid: int) -> float:
        """Energia cumulata (train + comm) del client, in Wh LINEARI (P*dt)."""
        return self._energy_lin_wh[cid]

    def energy_idle_wh(self, cid: int) -> float:
        return self._energy_idle_wh[cid]

    def tier(self, cid: int) -> str:
        return self._profiles[cid].tier

    def utilization(self, cid: int) -> float:
        return self.workload.utilization(cid)

    def is_charging(self, cid: int) -> bool:
        return self._charging[cid]

    def energy_grid_wh(self, cid: int) -> float:
        return self._energy_grid_wh[cid]

    def begin_round(self) -> None:
        """Inizio round: avanza il workload e decide chi va in carica.

        [B] la probabilita' cresce al calare del SoC, p = base * (1 - SoC):
        e' il comportamento dell'utente, che mette in carica quando vede la
        batteria bassa. Un device gia' in carica non viene riestratto: esce
        dalla carica solo al 100%.
        """
        self.workload.step()
        if not self._recharge_enabled:
            return
        for cid in self._profiles:
            if self._failed[cid] or self._charging[cid]:
                continue
            p = RECHARGE_PROB_BASE * (1.0 - self._soc[cid])
            if self._rng.random() < p:
                self._charging[cid] = True
                self._n_recharges[cid] += 1

    def tick_all(self, dt_s: float = 0.0) -> None:
        """Compatibilita': l'evoluzione ora sta in begin_round/apply_idle."""
        return None

    # -------------------------------------------------------------- tempi
    def effective_profile(self, cid: int) -> DeviceProfile:
        """Profilo col throughput ridotto dal workload corrente."""
        return effective_profile(self._profiles[cid],
                                 self.workload.utilization(cid))

    def round_duration_s(
        self, cid: int, epochs: int, batch_size: int, dataset_size_local: int,
        include_workload: bool = True,
    ) -> float:
        """Durata del training locale.

        include_workload=False da' la durata NOMINALE, che dipende dal solo
        tier. Serve a ESCS, che nel paper usa la latenza di profilo come dato
        statico noto al server: il workload e' proprio cio' che il server non
        puo' conoscere in anticipo.
        """
        p = self.effective_profile(cid) if include_workload else self._profiles[cid]
        return training_time_s(p, epochs, dataset_size_local, batch_size)

    def probe_duration_s(self, cid: int, n_samples: int) -> float:
        """Forward di sola valutazione su n_samples, col workload corrente.

        E' la stima della loss locale che il client di sage_smart fa prima
        di decidere (E, B): costa poco, ma non e' gratis e va addebitata.
        """
        return probe_time_s(self.effective_profile(cid), n_samples)

    def communication_time_s(self, cid: int) -> float:
        return comm_time_s()

    def training_power_w(self, cid: int, batch_size: int) -> float:
        return training_power_w(batch_size)

    # ------------------------------------------------------------ consumo
    def _drain(self, cid: int, power_w: float, dt_s: float) -> tuple[float, float]:
        """Applica power_w per dt_s. Ritorna (Wh Peukert, Wh lineari) pagati."""
        if self._failed[cid]:
            return 0.0, 0.0
        if self._charging[cid]:
            # collegato alla rete: la batteria non si scarica. L'energia c'e'
            # lo stesso e la contabilizziamo a parte.
            self._energy_grid_wh[cid] += (power_w / DCDC_EFFICIENCY) * dt_s / 3600.0
            return 0.0, 0.0
        profile = self._profiles[cid]
        soc = self._soc[cid]
        d_req = delta_soc(power_w, dt_s, profile, soc)
        new_soc = max(0.0, min(1.0, soc + d_req))
        # [B] energia dal delta EFFETTIVO, non da quello richiesto: il SoC e'
        # clampato a 0 ma il delta no, quindi un client che muore nel round
        # verrebbe addebitato per carica che non aveva.
        d = new_soc - soc
        self._soc[cid] = new_soc
        if new_soc <= 0.0:
            self._failed[cid] = True
        # la lineare va scalata con la stessa frazione effettiva: se il
        # client muore a meta' round paga meta' anche nella contabilita' P*dt
        frac = d / d_req if d_req < 0.0 else 0.0
        e_lin = energy_wh_linear(power_w, dt_s) * frac
        self._energy_lin_wh[cid] += e_lin
        self._soc_lin[cid] = max(
            0.0, min(1.0, self._soc_lin[cid] - e_lin / self.capacity_wh(cid)))
        # gauge di Peukert: stessa potenza, stessa frazione di tempo eseguita
        d_peuk = peukert_delta_soc(power_w, dt_s, profile) * frac
        self._soc_peuk[cid] = max(0.0, min(1.0, self._soc_peuk[cid] + d_peuk))
        return energy_wh_from_delta_soc(d, profile), e_lin

    def apply_round(self, cid: int, dt_s: float, batch_size: int) -> tuple[float, float]:
        e, e_lin = self._drain(cid, self.training_power_w(cid, batch_size), dt_s)
        self._energy_wh[cid] += e
        return e, e_lin

    def apply_control(self, cid: int, merged: bool = False) -> tuple[float, float]:
        """Uno scambio di controllo (richiesta + risposta) con il server.

        merged=True se lo scambio cade nella sessione radio del download del
        modello che segue (client selezionato): paga il solo payload. Vedi
        communication_model.
        """
        e, e_lin = self._drain(cid, ctrl_power_w(merged), ctrl_time_s(merged))
        self._energy_ctrl_wh[cid] += e
        return e, e_lin

    def apply_communication(self, cid: int) -> tuple[float, float]:
        e, e_lin = self._drain(cid, comm_power_w(), comm_time_s())
        self._energy_comm_wh[cid] += e
        return e, e_lin

    def idle_power_w(self, cid: int) -> float:
        """P_idle + w * (P_train - P_idle): interpola fra riposo e pieno carico."""
        w = self.workload.utilization(cid)
        return IDLE_POWER_W + w * (TRAIN_POWER_W - IDLE_POWER_W)

    def apply_idle(self, durations: dict[int, float]) -> tuple[float, float]:
        """Idle di fine round, modello ESCS Sez. 4.2.

        durations: cid -> durata del training, per i soli SELEZIONATI.
        Il round dura quanto il piu' lento fra loro; ogni client sta in idle
        per la differenza fra quella durata e il proprio training (zero per il
        piu' lento, tutta la durata per chi non e' stato selezionato).

        [B] va chiamata DOPO apply_round, altrimenti un client gia' a zero
        pagherebbe l'idle prima del training e l'ordine cambierebbe chi muore.
        """
        if not self._idle_enabled:
            return 0.0, 0.0
        t_round = max(durations.values()) if durations else 0.0
        if t_round <= 0.0:
            return 0.0, 0.0
        self._charge_step(t_round)
        tot_e = tot_lin = 0.0
        for cid in self._profiles:
            if self._failed[cid]:
                continue
            idle_s = t_round - float(durations.get(cid, 0.0))
            if idle_s <= 0.0:
                continue
            e, lin = self._drain(cid, self.idle_power_w(cid), idle_s)
            self._energy_idle_wh[cid] += e
            tot_e += e
            tot_lin += lin
        return tot_e, tot_lin

    def _charge_step(self, t_round: float) -> None:
        """Carica lineare a C-rate costante per la durata del round.

        [B] lineare per scelta: la curva reale e' CC-CV e rallenta sopra l'80%,
        ma modellarla aggiungerebbe due parametri per un effetto che qui non
        cambia nulla. La carica finisce al 100% e il device torna normale.
        """
        if not self._recharge_enabled:
            return
        d_soc = CHARGE_C_RATE * t_round / 3600.0
        for cid in self._profiles:
            if not self._charging[cid]:
                continue
            self._soc[cid] = min(1.0, self._soc[cid] + d_soc)
            self._soc_lin[cid] = min(1.0, self._soc_lin[cid] + d_soc)
            self._soc_peuk[cid] = min(1.0, self._soc_peuk[cid] + d_soc)
            if self._soc[cid] >= 1.0:
                self._charging[cid] = False
                # carica completa rilevata: i fuel gauge si riallineano
                self._soc_lin[cid] = 1.0
                self._soc_peuk[cid] = 1.0

    # ------------------------------------------------------------ metriche
    def total_energy_train_wh(self) -> float:
        return sum(self._energy_wh.values())

    def total_energy_comm_wh(self) -> float:
        return sum(self._energy_comm_wh.values())

    def total_energy_grid_wh(self) -> float:
        return sum(self._energy_grid_wh.values())

    def total_energy_idle_wh(self) -> float:
        return sum(self._energy_idle_wh.values())

    def total_energy_ctrl_wh(self) -> float:
        return sum(self._energy_ctrl_wh.values())

    def total_energy_wh(self) -> float:
        return (self.total_energy_train_wh() + self.total_energy_comm_wh()
                + self.total_energy_ctrl_wh() + self.total_energy_idle_wh())

    def total_energy_lin_wh(self) -> float:
        return sum(self._energy_lin_wh.values())

    def stats(self) -> dict:
        soc_all = list(self._soc.values())
        return {
            "total_energy_wh": self.total_energy_wh(),
            "energy_train_wh": self.total_energy_train_wh(),
            "energy_comm_wh": self.total_energy_comm_wh(),
            "energy_ctrl_wh": self.total_energy_ctrl_wh(),
            "energy_idle_wh": self.total_energy_idle_wh(),
            "energy_grid_wh": sum(self._energy_grid_wh.values()),
            "energy_lin_wh": self.total_energy_lin_wh(),
            "mean_workload": self.workload.mean_utilization(),
            "n_failed": sum(self._failed.values()),
            "n_recharging": int(sum(1 for v in self._charging.values() if v)),
            "total_recharges": int(sum(self._n_recharges.values())),
            # [B] distribuzione del SoC su TUTTA la popolazione, morti inclusi
            # col loro zero: la media dei soli sopravvissuti sarebbe ottimista.
            "mean_soc": float(np.mean(soc_all)),
            "median_soc": float(np.median(soc_all)),
            "min_soc": float(np.min(soc_all)),
            "max_soc": float(np.max(soc_all)),
            "p25_soc": float(np.percentile(soc_all, 25)),
            "p75_soc": float(np.percentile(soc_all, 75)),
            "n_soc_zero": int(sum(1 for v in soc_all if v <= 0.0)),
            "n_clients": len(soc_all),
            # [B] SoC medio e morti PER TIER: senza queste colonne non si puo'
            # verificare che l'eterogeneita' produca davvero un effetto.
            **{f"soc_{t}": float(np.mean([self._soc[c] for c in self._profiles
                                          if self._profiles[c].tier == t]))
               for t in sorted({p.tier for p in self._profiles.values()})},
            **{f"dead_{t}": int(sum(1 for c in self._profiles
                                    if self._profiles[c].tier == t
                                    and self._soc[c] <= 0.0))
               for t in sorted({p.tier for p in self._profiles.values()})},
        }


def build_world_state(n_clients: int, seed: int,
                      capacity_mah: float | None = None,
                      tiers_enabled: bool = True,
                      workload_enabled: bool = True,
                      idle_enabled: bool = True,
                      recharge_enabled: bool = False,
                      recharge_available: bool = True,
                      **kwargs) -> WorldState:
    """`battery_variant` accettato e ignorato: resta solo Peukert.

    I tre flag disattivano tier, workload e idle. Tutti a False riproducono
    esattamente il mondo delle campagne precedenti.
    """
    profiles = generate_profiles(n_clients, seed=seed, capacity_mah=capacity_mah,
                                 tiers_enabled=tiers_enabled)
    workload = WorkloadModel(n_clients=n_clients, seed=seed + 7919,
                             enabled=workload_enabled)
    return WorldState(profiles=profiles, seed=seed, workload=workload,
                      idle_enabled=idle_enabled,
                      recharge_enabled=recharge_enabled,
                      recharge_available=recharge_available)


if __name__ == "__main__":
    N, K, R = 30, 6, 250
    ws = build_world_state(n_clients=N, seed=42)
    rng = np.random.default_rng(0)
    for r in range(1, R + 1):
        alive = [c for c in range(N) if ws.snapshot(c).available]
        if not alive:
            print(f"popolazione esaurita al round {r}")
            break
        sel = rng.choice(alive, size=min(K, len(alive)), replace=False)
        for cid in sel:
            dt = ws.round_duration_s(cid, epochs=5, batch_size=128,
                                     dataset_size_local=1333)
            ws.apply_round(cid, dt_s=dt, batch_size=128)
            ws.apply_communication(cid)
        if r % 50 == 0:
            s = ws.stats()
            print(f"round {r:3d} | {s['total_energy_wh']:6.3f} Wh "
                  f"(train {s['energy_train_wh']:.3f} + comm {s['energy_comm_wh']:.3f}) "
                  f"| SoC medio {s['mean_soc']:.3f} | morti {s['n_soc_zero']}/{N}")