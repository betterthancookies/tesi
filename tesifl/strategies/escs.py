from __future__ import annotations

import math
from logging import INFO
import numpy as np
from flwr.app import Message
from flwr.common import log

from device.battery_model import V_NOMINAL
from strategies.fedavg import PhysicalFedAvg


class PhysicalESCS(PhysicalFedAvg):
    """ESCS (Maciel et al., 2024), allineato al paper e ai sorgenti degli autori.

    Round 1:     tutti i client disponibili (Alg. 1 righe 5-6; escs_sd.py
                 riga 108). Disattivabile con first_round_all=False.
    Filtro R:    (batteria - consumo previsto) >= min_battery  and  nq >= nq_min
    Utilita' U:
      "s" sistemica  U = (a*bat + b*(1 - cpu)*th + g*nq)^(1 - loss/max_loss)
      "m" modello    U = (loss/max_loss)^(1 - base)
    Penalita' th:  exp(|cpu - lim|) - 1  se cpu > lim,  1 altrimenti
    Selezione:     "d" i top-k per utilita', "p" i client con rand() <= U.

    da cui escs-sd, escs-sp, escs-md, escs-mp.

    LATENZA (cpu). Nel paper cpu = total_train_latency / max_training_latency
    e' un rapporto fra dati di PROFILO, noti per ogni client prima di partire,
    quindi cpu in (0, 1]. Qui la latenza di profilo e' predetta dal modello di
    calcolo del world state sulla VERA taglia della partizione locale
    (`partition_sizes`, ricostruita dal server con lo stesso partitioner dei
    client). Con device omogenei e' identica a quella misurata a fine round,
    quindi non serve aggiornarla. lim = percentile 95 / max, calcolato una
    volta sola su tutti i client (escs_sd.py righe 86-93; Table 4 dice "95%
    of maximum", il codice usa il percentile).

    [B] una latenza di riferimento su un dataset nominale diverso da quello
    vero manda cpu sopra 1, il termine (1-cpu)*th diventa molto negativo e
    l'utilita' sistemica collassa a zero: batteria e rete smettono di contare.
    E' il motivo per cui il riferimento DEVE essere calcolato sulle taglie
    reali.

    LOSS. [B] 23/10, allineata ai SORGENTI degli autori (escs_*.py,
    aggregate_fit righe 142-145): la loss di un client e' quella che riporta
    a fine TRAINING (fit_res.metrics["loss"]), aggiornata solo quando viene
    selezionato. Il paper parla di loss di valutazione (c_l, Table 2), ma il
    loro aggregate_evaluate registra solo l'accuracy. Qui: "train_loss" del
    nostro client (loss media del training locale). Prima la leggevamo da
    aggregate_evaluate, che con fraction_evaluate 0.2 la aggiornava a caso
    ogni ~5 round. Chi non ne ha ancora vale loss = max_loss, cioe' U = 1:
    priorita' all'esplorazione, che e' l'intento dell'inf iniziale nel
    codice originale (li' pero' inf/inf = nan escludeva per sempre i mai
    osservati; il bug non e' riprodotto).

    RISORSA DI BATTERIA. bat = SoC del device, letto dalla batteria del mondo
    (lineare, Peukert o datasheet: etichette escs_sd_lin / _peuk / _nm). Fra
    le tre etichette l'algoritmo e' identico e cambia SOLO la batteria
    montata sui device, come per sage_smart_lin/_peuk/_nm.
    La risorsa entra sia nell'utilita' sia nel filtro R, perche' nel paper
    sono la stessa grandezza; il consumo previsto e' il calo di SoC che la
    stessa batteria ha avuto nell'ultimo round del client.

    [B] la modalita' probabilistica non ha tetto: seleziona un numero
    variabile di client, ed e' il motivo per cui nel paper -sp/-mp sono le
    varianti piu' energivore. Il ramo deterministico e' un fallback che
    scatta solo se la probabilistica seleziona zero client. Con
    cap_probabilistic=True si tronca a k per il confronto a parita' di
    client per round, ma -sp diventa quasi indistinguibile da -sd.

    ADATTAMENTI DICHIARATI
    - il consumo previsto e' quello dell'ultimo round del client, in frazione
      di capacita'. Nell'originale e' delta_train_battery, costante per
      client; qui varia col carico locale, quindi l'ultimo osservato e' la
      stima. Chi non e' mai stato osservato usa la media degli altri.
    - min_battery e nq_min sono globali. Nell'originale sono campi per client
      del profilo device, che `device/` non ha. [ASSUNZIONE]
    - nq e' estratta U(0,1) una volta per client, come trans_prob nel profilo
      degli autori. `device/` modella il costo della comunicazione, non la
      probabilita' di fallimento, quindi non c'e' un corrispettivo fisico.
    - pesi a = b = g = 1/3: il paper usa pesi uguali senza specificarne la
      somma. [ASSUNZIONE] la somma a 1 e' nostra.
    - QUALITA' DI RETE statica e niente fallimenti di trasmissione. I loro
      sorgenti selezionano con trans_prob STATICA (has_client_resources) e
      fanno poi fallire la trasmissione con un profilo di rete per round
      (filter_clients_to_train_by_predicted_behavior): e' la simulazione del
      loro ambiente, che vale per tutti gli algoritmi. Il nostro ambiente
      (device/) non ha fallimenti di rete per nessun algoritmo: introdurli
      solo per ESCS non sarebbe un confronto equo. Conseguenza della nq
      statica: i client con nq < 0.2 non passano mai il filtro (13 / 22 / 22
      su 100 coi seed 2026 / 42 / 73), come nel loro codice.

    NUMERO DI SELEZIONATI dal round 2, come nei loro sorgenti (C = k / N):
      -sd/-md  idonei <= k: tutti; altrimenti i primi k per utilita'
      -sp/-mp  i sorteggiati; se nessuno, i primi round(len(idonei) * C)
      ultimo ripiego (nessun idoneo): per sola batteria, i primi
               round(len(con batteria) * C) per loss decrescente
    """

    def __init__(
        self,
        *args,
        utility_mode: str = "s",
        selection_mode: str = "d",
        partition_sizes: dict[int, int] | None = None,
        first_round_all: bool = True,
        battery_weight: float = 1 / 3,
        cpu_cost_weight: float = 1 / 3,
        link_prob_weight: float = 1 / 3,
        time_percentile: float = 95.0,
        min_battery: float = 0.20,
        min_network_quality: float = 0.20,
        cap_probabilistic: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        um, sm = utility_mode.lower(), selection_mode.lower()
        if um not in ("s", "m") or sm not in ("d", "p"):
            raise ValueError("utility_mode in {s,m}, selection_mode in {d,p}")
        if not partition_sizes:
            raise ValueError(
                "partition_sizes obbligatorio: senza le taglie reali la "
                "latenza di riferimento e' sbagliata e l'utilita' collassa"
            )
        self.utility_mode, self.selection_mode = um, sm
        self.first_round_all = bool(first_round_all)
        self.w_b, self.w_c, self.w_l = (
            float(battery_weight), float(cpu_cost_weight), float(link_prob_weight)
        )
        self.time_percentile = float(time_percentile)
        self.min_battery = float(min_battery)
        self.min_nq = float(min_network_quality)
        self.cap_prob = bool(cap_probabilistic)
        self._sizes = {int(c): int(n) for c, n in partition_sizes.items()}

        self._loss: dict[int, float] = {}     # cid -> eval loss ultimo round
        self._cons: dict[int, float] = {}     # cid -> consumo ultimo round (frazione)
        self._nq: dict[int, float] = {}       # cid -> qualita' rete, statica
        self._lat_ref: dict[int, float] = {}  # cid -> latenza di profilo
        self._max_lat = 0.0
        self._lim = 1.0
        self._cap_wh: dict[int, float] = {}

    @property
    def variant(self) -> str:
        return f"escs-{self.utility_mode}{self.selection_mode}-{self.world.battery}"

    # ----------------------------------------------------------- componenti
    def _ensure_profile(self) -> None:
        """Latenze di profilo e nq: una volta sola, su tutti i client."""
        if self._lat_ref:
            return
        for cid in sorted(self._node_to_cid.values()):
            if cid not in self._sizes:
                raise KeyError(f"partition_sizes non contiene il cid {cid}")
            # [B] durata NOMINALE, senza workload: nel paper la latenza di
            # training e' un dato di profilo statico, noto al server. Il
            # workload e' proprio cio' che il server non puo' conoscere in
            # anticipo, ed e' il motivo per cui la decisione andrebbe presa
            # dal client. Con i tier questa latenza ora varia per client, che
            # e' la situazione che il paper descrive.
            self._lat_ref[cid] = float(self.world.round_duration_s(
                cid, epochs=self.epochs, batch_size=self.batch_size,
                dataset_size_local=self._sizes[cid], include_workload=False,
            ))
            self._nq[cid] = float(self._rng.uniform(0.0, 1.0))
        lats = list(self._lat_ref.values())
        self._max_lat = max(lats) or 1.0
        self._lim = float(np.percentile(lats, self.time_percentile)) / self._max_lat
        log(INFO, "%s: latenze di profilo %.1f-%.1f s, lim=%.3f",
            self.variant, min(lats), self._max_lat, self._lim)

    def _capacity(self, cid: int) -> float:
        if cid not in self._cap_wh:
            mah = float(self.world.profile(cid).battery_capacity_mah)
            self._cap_wh[cid] = mah / 1000.0 * V_NOMINAL
        return self._cap_wh[cid]

    def _battery(self, cid: int) -> float:
        """La risorsa di batteria vista dal selettore: il SoC del device."""
        return float(self.world.snapshot(cid).soc)

    def _consumption(self, cid: int) -> float:
        if cid in self._cons:
            return self._cons[cid]
        return float(np.mean(list(self._cons.values()))) if self._cons else 0.0

    def _has_resources(self, cid: int) -> bool:
        return (self._battery(cid) - self._consumption(cid) >= self.min_battery
                and self._nq[cid] >= self.min_nq)

    # ------------------------------------------------------------ selezione
    def _select_clients(
        self, available: list[int], k: int, server_round: int
    ) -> list[int]:
        if not k:
            return []
        self._ensure_profile()
        cid_of = self._node_to_cid

        # Alg. 1 righe 5-6: al primo round partecipano tutti
        if server_round == 1 and self.first_round_all:
            log(INFO, "%s round 1: tutti i %s disponibili", self.variant,
                len(available))
            return list(available)

        ast = [n for n in available if self._has_resources(cid_of[n])]

        max_loss = max(self._loss.values()) if self._loss else 1.0
        util = {n: self._utility(cid_of[n], max_loss) for n in ast}
        # tie-break randomizzato: senza, il sort stabile esplora nell'ordine
        # di `available`, accoppiato alla partizione dei dati.
        ast = [ast[i] for i in self._rng.permutation(len(ast))] if ast else []

        # frazione di training C dei loro sorgenti (fit_fraction, costante):
        # client per round su N iscritti, non il k del round (che cala coi morti)
        frac = self.k_per_round / max(1, len(cid_of))
        rate = float("nan")
        selected: list[int] = []
        if ast and self.selection_mode == "p":
            selected = [n for n in ast if self._rng.random() <= util[n]]
            rate = len(selected) / len(ast)
            if self.cap_prob and len(selected) > k:
                idx = self._rng.choice(len(selected), size=k, replace=False)
                selected = [selected[i] for i in idx]

        if not selected and ast:
            ranked = sorted(ast, key=lambda n: -util[n])
            if self.selection_mode == "d":
                # escs_sd/md: tutti se non superano k, altrimenti i primi k
                selected = ast if len(ast) <= k else ranked[:k]
            else:
                # escs_sp/mp, ripiego quando il sorteggio non sceglie nessuno:
                # i primi round(len(idonei) * C), senza il ramo "tutti"
                selected = ranked[:round(len(ast) * frac)]

        if not selected:
            # terzo fallback degli autori: si ignora la qualita' del link e si
            # seleziona per sola batteria, ordinando per loss decrescente.
            batt = [n for n in available
                    if self._battery(cid_of[n]) - self._consumption(cid_of[n])
                    >= self.min_battery]
            batt.sort(key=lambda n: -self._loss.get(cid_of[n], max_loss))
            selected = batt[:round(len(batt) * frac)]

        log(INFO, "%s round %s: %s idonei su %s, %s selezionati (accept %.2f)",
            self.variant, server_round, len(ast), len(available),
            len(selected), rate)
        return selected

    def _utility(self, cid: int, max_loss: float) -> float:
        bat = self._battery(cid)
        cpu = self._lat_ref[cid] / self._max_lat          # in (0, 1]
        th = math.exp(abs(cpu - self._lim)) - 1 if cpu > self._lim else 1.0
        base = max(
            self.w_b * bat + self.w_c * (1.0 - cpu) * th + self.w_l * self._nq[cid],
            1e-12,
        )
        ratio = min(max(self._loss.get(cid, max_loss) / max_loss, 1e-12), 1.0)
        try:
            return float(base ** (1.0 - ratio) if self.utility_mode == "s"
                         else ratio ** (1.0 - base))
        except (OverflowError, ValueError):
            return 0.0

    # ---------------------------------------------------------------- update
    def _on_round_result(
        self, cid: int, n_ex: int, dt_s: float, e_wh: float, e_lin_wh: float,
        reply: Message, server_round: int,
    ) -> None:
        # consumo previsto: il calo di SoC di training + upload di questo
        # round, nella batteria del mondo (e_wh = V_nom * C_nom * |dSoC|)
        cap = self._capacity(cid)
        if cap > 0.0:
            self._cons[cid] = float(e_wh / cap)
        # loss del client per l'utilita': quella del suo training, come nei
        # sorgenti degli autori (aggregate_fit, solo se num_examples != 0)
        if n_ex > 0:
            loss = self._metric(reply, "train_loss", "loss", default=None)
            if loss is not None and math.isfinite(float(loss)):
                self._loss[cid] = float(loss)