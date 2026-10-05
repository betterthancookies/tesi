from __future__ import annotations

import math
from logging import INFO

import numpy as np
from flwr.app import ConfigRecord, Message
from flwr.common import log

from device.battery_model import energy_wh_linear
from device.communication_model import comm_power_w, comm_time_s
from device.compute_model import probe_time_s, training_time_s
from device.constants import V_NOMINAL, WORKLOAD_STATES
from device.fuel_gauge import DeviceReading
from device.world_state import effective_profile, training_power_w
from strategies.sage_smart import PhysicalSAGESmart
from tesiFL.client_policy import Envelope


class PhysicalSAGESmart2(PhysicalSAGESmart):
    """SAGE-smart con (E, B) decisi dal CLIENT: lo schema della proposta.

    In `sage_smart` il server sceglie chi partecipa E quanto lavora. Qui il
    server sceglie solo chi partecipa e propone un intervallo; quanto lavorare
    lo decide il client, un istante prima di iniziare, con cio' che solo lui
    sa. Le fasi sono quelle della proposta (Sez. 4).

    FASE 0 -- ISCRIZIONE (una volta, al primo round)
      Il server registra i dati STATICI di ogni client: profilo hardware
      (capacita', velocita', curva di potenza), taglia del dataset, SoC
      iniziale. Ne ricava la deadline del round e il batch ottimo per device.

    FASE 2 -- HEARTBEAT (ogni round, prima della selezione; disattivabile)
      Ogni client comunica SoC, stato di carica e workload. E' l'istantanea
      che il server usa per selezionare e per costruire l'envelope.
      [B] simulato come lettura diretta del mondo, senza un messaggio Flower:
      e' un pacchetto di pochi byte e il suo costo energetico e' trascurabile.
      [B] il workload del heartbeat e' quello del round PRECEDENTE: il mondo
      lo fa avanzare dopo la selezione (WorldState.begin_round). Lo scarto fra
      cio' che il server vede e cio' che il client trova e' esattamente il
      motivo del late binding.
      Con heartbeat=False il server conosce solo cio' che i client riportano
      a fine round (SoC finale) e, per il workload, assume il caso peggiore.

    FASE 3 -- SELEZIONE (invariata rispetto a sage_smart)
          score = a*SoC + b*D + c*min(1, eta / stale_max)
      con SoC quello NOTO AL SERVER (heartbeat o ultimo riportato), non quello
      vero del mondo. c e' `stale_weight`.

    FASE 4 -- PROPOSTA (envelope), per ogni selezionato
      E in [E_lo, E_hi]   B in {batch ammessi}   + deadline, soglia SoC
      E_lo = epochs_min: sotto, l'update non porta segnale utile.
      E_hi = min(2*E_nom, vincolo di deadline, vincolo di energia), dove
        deadline: E * t_epoca(B, w_noto) <= T_round
        energia:  E * dSoC_epoca <= SoC_noto - soc_min - dSoC_upload
                  con dSoC stimato LINEARMENTE (P*t/capacita') e corretto da
                  un fattore k_i appreso dai consumi che il client riporta.
      B: il batch ottimo del device (compute_model.optimal_batch_size) e i
      due adiacenti nella lista delle taglie ammesse.
      [B] e' una stima volutamente LASCA: il server non conosce il workload
      attuale ne' la curva vera della batteria. Il compito di essere precisi
      e' del client.
      [B] T_round = deadline_mult * mediana dei tempi nominali (E_nom, B_nom,
      senza workload). Limita lo straggler, quindi anche l'idle che gli altri
      pagano aspettandolo (modello ESCS), e tiene i tau comparabili per
      FedNova: un client in carica non puo' piu' ricevere 10 epoche e
      arrivare a tau 30 volte gli altri, che era il guasto di sage_smart.

    FASE 5 -- DECISIONE DEL CLIENT: vedi tesiFL/client_policy.py.
      Il client rilegge SoC, workload e carica, stima la loss del modello
      globale sui propri dati e sceglie (E, B) dentro l'envelope, oppure si
      ritira (opt-out) se nemmeno il minimo rispetta la sua riserva.

    FASE 6 -- TRAINING con guardia: il client si ferma se finisce il budget
      energetico, anche a meta' epoca, e lo dichiara.

    FASE 7 -- RISPOSTA E AGGREGAZIONE
      Il client comunica (E, B) scelti, passi eseguiti, SoC iniziale e
      FINALE, loss prima e dopo. Il server:
        - verifica che (E, B) stiano nell'envelope: se no scarta l'update;
        - addebita al mondo fisico il lavoro DICHIARATO (Fedavg._work_done);
        - aggrega con FedNova, tau_i = passi dichiarati dal client.

    FASE 8 -- AGGIORNAMENTO DELLO STATO
      SoC noto <- SoC finale riportato; k_i <- media mobile del rapporto fra
      calo di SoC osservato e stima lineare; divergenza D_i dalla
      distribuzione delle classi (come sage_smart).

    COSA SA IL SERVER, E DA DOVE
      Le decisioni della strategia usano solo: profili statici (Fase 0),
      heartbeat (Fase 2), metriche riportate dai client (Fase 7). Il mondo
      fisico entra in due soli punti, entrambi fuori dalle decisioni:
        - `_device_record`: i SENSORI del device, allegati al messaggio per
          conto del device stesso (in un telefono vero li legge dall'OS);
        - la colonna `s2_soc_err`, che confronta il SoC dichiarato dal
          client con quello del mondo: e' una verifica, non un input.

    PARAMETRI NUOVI rispetto a sage_smart
      deadline_mult  scala della deadline sulla mediana nominale
      probe_samples  campioni per stimare la loss prima di decidere
      heartbeat      Fase 2 attiva o no (ablazione: server cieco)
      calib_alpha    peso della media mobile su k_i
    soc_min ed epochs_min esistono gia' in sage_smart, qui cambiano ruolo:
    soc_min e' la riserva che il CLIENT protegge, epochs_min e' E_lo.
    """

    def __init__(
        self,
        *args,
        partition_sizes: dict[int, int] | None = None,
        epochs_min: int = 2,
        deadline_mult: float = 1.0,
        probe_samples: int = 256,
        heartbeat: bool = True,
        calib_alpha: float = 0.3,
        **kwargs,
    ) -> None:
        super().__init__(*args, epochs_min=epochs_min, **kwargs)
        self.deadline_mult = float(deadline_mult)
        self.probe_samples = int(probe_samples)
        self.heartbeat = bool(heartbeat)
        self.calib_alpha = float(calib_alpha)
        # Fase 0: taglia dei dataset. Se manca, la si impara dalle risposte.
        self._n: dict[int, int] = {int(c): int(n) for c, n in
                                   (partition_sizes or {}).items()}
        # [B] workload ignoto = caso peggiore (proposta, Fase 1): meglio
        # proporre poco lavoro a un client che si scopre libero che troppo a
        # uno che si scopre occupato.
        self.w_busy = float(max(WORKLOAD_STATES.values()))

        # stato che il server tiene per ogni client
        self._subscribed = False
        self._deadline_s: float = math.inf
        self._soc_known: dict[int, float] = {}
        self._chg_known: dict[int, bool] = {}
        self._w_known: dict[int, float] = {}
        self._k_cal: dict[int, float] = {}
        self._loss_known: dict[int, float] = {}
        # per round: vincolo che ha fissato E_hi e risposte dei client
        self._bound: dict[int, str] = {}
        self._reports: dict[int, dict] = {}

    # ===================================================== Fase 0 e Fase 2
    def _subscribe(self) -> None:
        """Iscrizione: dati statici e SoC iniziale di ogni client."""
        cids = list(self._node_to_cid.values())
        for c in cids:
            self._soc_known[c] = float(self.world.initial_soc(c))
            self._chg_known[c] = False
            self._w_known[c] = self.w_busy
            self._k_cal[c] = 1.0
        nominal = [
            training_time_s(self.world.profile(c), int(self.epochs),
                            self._n[c], int(self.batch_size))
            for c in cids if self._n.get(c)
        ]
        if nominal:
            self._deadline_s = self.deadline_mult * float(np.median(nominal))
        self._subscribed = True
        log(INFO, "sage-smart2: %s client iscritti | deadline %.1f s "
            "(%.2f x mediana nominale, E=%s B=%s) | heartbeat %s",
            len(cids), self._deadline_s, self.deadline_mult, self.epochs,
            self.batch_size, "ON" if self.heartbeat else "OFF")

    def _heartbeat(self) -> None:
        """Fase 2: stato dinamico fresco, prima della selezione."""
        if not self.heartbeat:
            return
        for c in self._node_to_cid.values():
            snap = self.world.snapshot(c)
            self._soc_known[c] = float(snap.soc)
            self._chg_known[c] = bool(snap.recharging)
            self._w_known[c] = float(self.world.utilization(c))

    def _resource(self, cid: int) -> float:
        """SoC usato nello score: quello NOTO al server, non quello vero."""
        return self._soc_known.get(cid, float(self.world.initial_soc(cid)))

    def _n_of(self, cid: int) -> int | None:
        if cid in self._n:
            return self._n[cid]
        return int(np.median(list(self._n.values()))) if self._n else None

    # ============================================================ Fase 3
    def _select_clients(
        self, available: list[int], k: int, server_round: int
    ) -> list[int]:
        if not self._subscribed:
            self._subscribe()
        self._heartbeat()
        self._reports = {}
        self._bound = {}
        return super()._select_clients(available, k, server_round)

    # ============================================================ Fase 4
    def _batches_for(self, cid: int) -> tuple[int, ...]:
        """Batch ottimo del device e i due adiacenti fra le taglie ammesse."""
        ch = self.batch_choices
        i = ch.index(self._batch_for(cid))
        return tuple(ch[max(0, i - 1): i + 2])

    def _assign_work(self, cid: int) -> Envelope:
        prof = self.world.profile(cid)          # statico: dalla Fase 0
        batches = self._batches_for(cid)
        b_ref = self._batch_for(cid)
        peff = effective_profile(prof, self._w_known.get(cid, self.w_busy))
        e_lo, e_cap = int(self.epochs_min), int(self.epochs_cap)
        n = self._n_of(cid)

        # vincolo di deadline: col batch piu' veloce dell'envelope
        e_dl = e_cap
        if n and math.isfinite(self._deadline_s):
            t_ep = min(training_time_s(peff, 1, n, b) for b in batches)
            e_dl = int(self._deadline_s // max(t_ep, 1e-9))

        # vincolo di energia: stima lineare corretta da k_i
        e_en = e_cap
        if n and not self._chg_known.get(cid, False):
            cap_wh = prof.battery_capacity_mah / 1000.0 * V_NOMINAL
            k = self._k_cal.get(cid, 1.0)
            d_ep = k * energy_wh_linear(
                training_power_w(b_ref), training_time_s(peff, 1, n, b_ref)
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
        )

    def _plan_epochs(self) -> list[int]:
        return [env.e_hi for env in self._plan.values()]

    def _per_client_config(
        self, cid: int, cfg: ConfigRecord, server_round: int
    ) -> None:
        env = self._plan.get(cid)
        if env is None:
            return
        for key, val in env.to_config().items():
            cfg[key] = val
        # [B] epochs e batch-size restano nel config come TETTO e batch di
        # riferimento: il client di sage_smart2 non li usa per allenarsi.
        cfg["epochs"] = int(env.e_hi)
        cfg["batch-size"] = int(self._batch_for(cid))

    def _device_record(self, cid: int) -> ConfigRecord | None:
        """Sensori del device: letti dal mondo PER CONTO del client.

        [B] chiamato dopo WorldState.begin_round: il client vede il workload
        e lo stato di carica di QUESTO round, il server nella selezione aveva
        visto quelli del precedente.
        """
        return ConfigRecord(DeviceReading.from_world(self.world, cid).to_record())

    # ============================================================ Fase 7
    def _rep(self, reply: Message, key: str, default=None):
        return self._metric(reply, key, default=default)

    def _is_optout(self, reply: Message) -> bool:
        return int(self._rep(reply, "s2-optout", 0) or 0) == 1

    def _work_done(self, cid: int, n_ex: int, reply: Message) -> tuple[float, int]:
        """Lavoro dichiarato dal client -> durata, per l'addebito fisico.

        Stessa formula che il client usa per prevedere il proprio SoC
        (DeviceReading.work_time_s), quindi il SoC che il mondo applica
        coincide con quello che il client ha comunicato.
        """
        b = int(self._rep(reply, "s2-batch", self._batch_for(cid)))
        steps = int(self._rep(reply, "s2-steps", 0))
        probe = int(self._rep(reply, "s2-probe", 0))
        e_eff = steps / math.ceil(n_ex / b) if steps > 0 and n_ex > 0 else 0.0
        dt = (self.world.round_duration_s(cid, epochs=e_eff, batch_size=b,
                                          dataset_size_local=n_ex)
              + self.world.probe_duration_s(cid, probe))
        return dt, b

    def _in_envelope(self, cid: int, reply: Message) -> bool:
        env = self._plan.get(cid)
        if env is None:
            return False
        e = int(self._rep(reply, "s2-epochs", 0))
        b = int(self._rep(reply, "s2-batch", 0))
        return env.e_lo <= e <= env.e_hi and b in env.batches

    def _aggregatable(self, replies: list[Message]) -> list[Message]:
        """Esclude chi si e' ritirato e chi e' uscito dall'envelope.

        [B] il client non dovrebbe mai uscire dall'envelope: la sua policy
        sceglie dentro i limiti. Il controllo rende esplicito il contratto:
        fuori dai limiti il server non accetta l'update (ma il lavoro e'
        stato fatto, quindi l'energia viene addebitata lo stesso).
        """
        out = []
        for r in replies:
            if not r.has_content():
                out.append(r)      # gli errori li gestisce FedAvg, come prima
                continue
            if self._is_optout(r):
                continue
            cid = self._node_to_cid.get(int(r.metadata.src_node_id))
            if cid is not None and not self._in_envelope(cid, r):
                continue
            out.append(r)
        return out

    def _steps(self, cid: int, n: float, reply) -> float:
        """tau_i di FedNova: i passi che il client dichiara di aver fatto."""
        if self._is_optout(reply):
            return 0.0
        return float(self._rep(reply, "s2-steps", 0))

    # ============================================================ Fase 8
    def _on_round_result(
        self, cid: int, n_ex: int, dt_s: float, e_wh: float, e_lin_wh: float,
        reply: Message, server_round: int,
    ) -> None:
        super()._on_round_result(cid, n_ex, dt_s, e_wh, e_lin_wh, reply,
                                 server_round)
        optout = self._is_optout(reply)
        soc0 = float(self._rep(reply, "s2-soc-start", self._resource(cid)))
        soc1 = float(self._rep(reply, "s2-soc-final", soc0))
        chg = bool(int(self._rep(reply, "s2-charging", 0)))
        w = float(self._rep(reply, "s2-workload", self.w_busy))
        b = int(self._rep(reply, "s2-batch", self._batch_for(cid)))
        steps = int(self._rep(reply, "s2-steps", 0))
        probe = int(self._rep(reply, "s2-probe", 0))

        # stato noto al server: cio' che il client ha comunicato
        self._soc_known[cid] = soc1
        self._chg_known[cid] = chg
        if not self.heartbeat:
            self._w_known[cid] = w
        if n_ex > 0:
            self._n[cid] = int(n_ex)
        loss = self._rep(reply, "s2-loss-pre", None)
        if loss is not None:
            self._loss_known[cid] = float(loss)

        # calibrazione k_i: calo di SoC vero / stima lineare dello stesso
        # lavoro. Il server ricostruisce la durata con il workload DICHIARATO.
        if not chg and steps > 0 and n_ex > 0:
            prof = self.world.profile(cid)
            peff = effective_profile(prof, w)
            e_eff = steps / math.ceil(n_ex / b)
            t = training_time_s(peff, e_eff, n_ex, b) + probe_time_s(peff, probe)
            cap_wh = prof.battery_capacity_mah / 1000.0 * V_NOMINAL
            lin = (energy_wh_linear(training_power_w(b), t)
                   + energy_wh_linear(comm_power_w(), comm_time_s())) / cap_wh
            drop = soc0 - soc1
            if lin > 0 and drop > 0:
                a = self.calib_alpha
                self._k_cal[cid] = (1 - a) * self._k_cal.get(cid, 1.0) + a * drop / lin

        self._reports[cid] = {
            "optout": optout,
            "epochs": int(self._rep(reply, "s2-epochs", 0)),
            "e_eff": (steps / math.ceil(n_ex / b)) if steps > 0 and n_ex > 0 else 0.0,
            "batch": b,
            "steps": steps,
            "budget_hit": int(self._rep(reply, "s2-budget-hit", 0)),
            "deadline_ok": int(self._rep(reply, "s2-deadline-ok", 1)),
            "charging": chg,
            "loss_pre": float(loss) if loss is not None else float("nan"),
            "loss_post": float(self._rep(reply, "s2-loss-post", float("nan"))),
            "workload": w,
            # [B] VERIFICA, non input: SoC dichiarato contro SoC del mondo
            # dopo l'addebito di training + upload (l'idle viene dopo).
            "soc_err": abs(float(self.world.snapshot(cid).soc) - soc1),
            "rejected": (not optout) and not self._in_envelope(cid, reply),
        }
        if self._reports[cid]["rejected"]:
            log(INFO, "sage-smart2 round %s: update del client %s fuori "
                "envelope, scartato", server_round, cid)

    def _policy_stats(self) -> dict:
        rep = list(self._reports.values())
        done = [r for r in rep if not r["optout"]]
        envs = list(self._plan.values())

        def mean(xs):
            xs = [x for x in xs if x == x]          # scarta i NaN
            return float(np.mean(xs)) if xs else float("nan")

        st = {
            # stessi nomi di sage_smart: le epoche sono quelle ESEGUITE
            "mean_epochs": mean([r["e_eff"] for r in done]),
            "max_epochs": float(max((r["e_eff"] for r in done), default=0.0)),
            "mean_batch": mean([r["batch"] for r in done]),
            "n_charging_sel": int(sum(1 for r in rep if r["charging"])),
            # specifiche dello schema client-driven
            "s2_min_epochs": float(min((r["e_eff"] for r in done), default=0.0)),
            "s2_mean_e_hi": mean([e.e_hi for e in envs]),
            "s2_n_bound_energy": int(sum(1 for v in self._bound.values() if v == "energy")),
            "s2_n_bound_deadline": int(sum(1 for v in self._bound.values() if v == "deadline")),
            "s2_n_optout": int(sum(1 for r in rep if r["optout"])),
            "s2_n_rejected": int(sum(1 for r in rep if r["rejected"])),
            "s2_n_deadline_miss": int(sum(1 for r in done if not r["deadline_ok"])),
            "s2_n_budget_hit": int(sum(1 for r in done if r["budget_hit"])),
            "s2_mean_loss_pre": mean([r["loss_pre"] for r in rep]),
            "s2_mean_loss_post": mean([r["loss_post"] for r in done]),
            "s2_mean_workload": mean([r["workload"] for r in rep]),
            "s2_soc_err": float(max((r["soc_err"] for r in rep), default=0.0)),
            "s2_mean_k": mean(list(self._k_cal.values())),
            "s2_deadline_s": float(self._deadline_s),
        }
        log(INFO, "sage-smart2: %s risposte | E eseguite %.1f (min %.2f, max %.2f) "
            "| E_hi medio %.1f (energia %s, deadline %s) | opt-out %s | "
            "fuori deadline %s | errore SoC %.1e",
            len(rep), st["mean_epochs"], st["s2_min_epochs"], st["max_epochs"],
            st["s2_mean_e_hi"], st["s2_n_bound_energy"],
            st["s2_n_bound_deadline"], st["s2_n_optout"],
            st["s2_n_deadline_miss"], st["s2_soc_err"])
        return st
