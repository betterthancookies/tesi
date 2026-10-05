from __future__ import annotations

from logging import INFO

import numpy as np
import pulp
from flwr.app import ArrayRecord, ConfigRecord
from flwr.common import log
from flwr.app import Array, ArrayRecord, ConfigRecord
from device.compute_model import optimal_batch_size
from strategies.sage_ablation import PhysicalSAGEAblation


class PhysicalSAGESmart(PhysicalSAGEAblation):
    """SAGE-peuk piu' assegnazione dinamica di (E, B) e bonus di anzianita'.

    Identica a `PhysicalSAGEAblation` per il modello di batteria (SoC Peukert)
    e per la struttura della selezione. Cambiano tre cose, tutte lato server.

    1. BONUS DI ANZIANITA' NELLA SELEZIONE
       [B] SAGE sceglie i top-k per a*SoC + b*D, dove D e' STATICA: chi ha
       divergenza bassa non viene scelto mai, e chi consuma poco resta carico
       e viene riscelto sempre. Misurato: con i tier, 4 client su 30 non
       vengono mai selezionati e uno viene scelto 228 volte su 250, con un
       Gini delle selezioni di 0.62 contro lo 0.19 di FedAvg -- e il modello
       globale non vede mai i loro dati.
       Il termine di anzianita' rompe il ciclo: chi non partecipa da tanto
       guadagna punteggio finche' non viene provato. E' la Fase 8 della
       proposta, ed e' quello che fanno Oort e F3AST.

           score = a*SoC + b*D + c*min(1, eta / stale_max)

    2. EPOCHE PER CLIENT
       [B] un'epoca costa 2.8 volte di piu' sul tier lento che su quello
       veloce (a parita' di dati). Assegnare a tutti lo stesso E spreca
       batteria dove costa di piu'. Qui:

           E_i = clip( round( E_nom * v_i * h_i ), 1, 2*E_nom )
           v_i = macs_i / macs_medio            (velocita' relativa)
           h_i = (SoC_i - soc_min) / (1 - soc_min)   (riserva disponibile)

       Il primo fattore sposta il lavoro verso chi lo paga meno, il secondo
       protegge chi e' scarico. Un client scarico ma UTILE viene comunque
       selezionato -- e' il termine D a deciderlo -- ma riceve poche epoche:
       si preserva senza perderne il contributo.
       [B] soc_min riusa 0.2, cioe' il vincolo (2e) del paper SAGE, che
       l'ablazione aveva tolto come filtro rigido. Qui non torna come filtro
       ma come scala della riserva: nessun parametro nuovo.

    3. BATCH PER CLIENT
       B_i = optimal_batch_size(profilo), arrotondato alle taglie ammesse.
       [B] e' una funzione gia' presente in compute_model.py e mai usata da
       nessuna strategia: calcola analiticamente il batch che minimizza il
       costo di un'epoca per quel device, Peukert incluso. Zero parametri.
       Con i tier da 1x/2x/4x da' 32 / 64 / 128.

    CLIENT IN CARICA
    [B] un device collegato alla rete si allena a costo zero per la sua
    batteria, quindi riceve E massimo: e' il momento giusto per fargli fare
    lavoro. Attivo solo se la ricarica e' abilitata nel world state.

    4. AGGREGAZIONE STEP-NORMALIZED (FedNova)
       [B] con E e B per client gli update non sono piu' commensurabili: chi
       fa 8 epoche si allontana molto di piu' dal modello globale di chi ne fa
       1, e FedAvg lo media con lo stesso peso. Wang et al. (NeurIPS 2020)
       mostrano che cosi' si ottimizza implicitamente un obiettivo DIVERSO,
       sbilanciato verso chi ha lavorato di piu'. Qui sarebbe un bias
       correlato all'hardware, perche' le epoche crescono col tier.
       La correzione divide l'update per i passi compiuti prima di mediarlo.
       Vedi `aggregate_train` in fondo alla classe.

    LIMITE DICHIARATO
    La decisione e' del SERVER, con informazione che il server gia' possiede
    (SoC, tier, divergenza). Non e' lo schema client-driven del titolo della
    tesi: manca il late-binding, cioe' la decisione presa con cio' che solo il
    client sa -- il workload concorrente, che qui non entra. Serve come prova
    preliminare che assegnare (E, B) in modo adattivo porti un guadagno.
    """

    def __init__(
        self,
        *args,
        stale_weight: float = 0.3,
        stale_max: int = 25,
        soc_min: float = 0.20,
        batch_choices: tuple[int, ...] = (32, 64, 128, 256),
        epochs_cap_mult: int = 2,
        epochs_min: int = 3,
        boost_charging: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.stale_w = float(stale_weight)
        self.stale_max = int(stale_max)
        self.soc_min = float(soc_min)
        self.batch_choices = tuple(int(b) for b in batch_choices)
        self.epochs_cap = int(epochs_cap_mult) * int(self.epochs)
        self.epochs_min = int(epochs_min)
        self.boost_charging = bool(boost_charging)

        self._last_seen: dict[int, int] = {}   # cid -> ultimo round di selezione
        self._plan: dict[int, tuple[int, int]] = {}   # cid -> (E, B) del round
        self._macs_mean: float | None = None
        self._batch_of: dict[int, int] = {}

    # ------------------------------------------------------------ componenti
    def _mean_macs(self) -> float:
        if self._macs_mean is None:
            vals = [self.world.profile(c).macs_per_s
                    for c in self._node_to_cid.values()]
            self._macs_mean = float(np.mean(vals)) if vals else 1.0
        return self._macs_mean

    def _batch_for(self, cid: int) -> int:
        """Batch ottimo del device, calcolato una volta sola: dipende dal
        profilo hardware, che non cambia durante la run."""
        if cid not in self._batch_of:
            b = optimal_batch_size(self.world.profile(cid))
            self._batch_of[cid] = min(self.batch_choices,
                                      key=lambda x: abs(x - b))
        return self._batch_of[cid]

    def _staleness(self, cid: int, server_round: int) -> float:
        """Quanto tempo e' passato dall'ultima selezione, normalizzato.

        Chi non e' mai stato scelto ha anzianita' massima dal primo round:
        e' il modo per provarlo invece di escluderlo per sempre.
        """
        last = self._last_seen.get(cid)
        age = server_round if last is None else server_round - last
        return float(min(1.0, age / max(1, self.stale_max)))

    # ------------------------------------------------------------- selezione
    def _select_clients(
        self, available: list[int], k: int, server_round: int
    ) -> list[int]:
        if not k or not available:
            log(INFO, "sage-smart round %s: pool vuoto", server_round)
            return []
        cid_of = self._node_to_cid
        pool = available
        m = len(pool)
        hi = min(self.k + server_round - 1, int(np.floor(self.max_frac * m)))
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
            log(INFO, "sage-smart round %s: ILP infeasible con pool %s "
                "(lo=%s, hi=%s): pool esaurito", server_round, m, lo, hi)
            return []

        for n in selected:
            self._last_seen[cid_of[n]] = server_round
        self._plan = {cid_of[n]: self._assign_work(cid_of[n]) for n in selected}

        E = self._plan_epochs()
        never = sum(1 for c in cid_of.values() if c not in self._last_seen)
        log(INFO, "sage-smart round %s: pool %s, %s selezionati (cap %s) | "
            "E %s-%s (medio %.1f) | mai visti %s",
            server_round, m, len(selected), hi, min(E), max(E),
            float(np.mean(E)), never)
        return selected

    def _assign_work(self, cid: int) -> tuple[int, int]:
        b = self._batch_for(cid)
        soc = self._resource(cid)
        v = self.world.profile(cid).macs_per_s / self._mean_macs()
        h = (soc - self.soc_min) / max(1e-9, 1.0 - self.soc_min)
        h = float(np.clip(h, 0.0, 1.0))
        e = int(np.clip(round(self.epochs * v * h),
                        self.epochs_min, self.epochs_cap))
        return e, b

    def _plan_epochs(self) -> list[int]:
        """Epoche del piano del round, per il log. sage_smart2 lo sostituisce:
        li' il piano e' un envelope e le epoche le sceglie il client."""
        return [e for e, _ in self._plan.values()]

    def _policy_stats(self) -> dict:
        """Colonne della policy nel CSV per round."""
        E = [e for e, _ in self._plan.values()]
        B = [b for _, b in self._plan.values()]
        return {
            "mean_epochs": float(np.mean(E)),
            "max_epochs": int(max(E)),
            "mean_batch": float(np.mean(B)),
            "n_charging_sel": int(sum(
                1 for c in self._plan if self.world.is_charging(c))),
        }

    def _steps(self, cid: int, n: float, reply) -> float:
        """tau_i di FedNova: passi di gradiente del client nel round.

        Qui lo ricostruisce il server da (E, B) assegnati. sage_smart2 usa
        invece i passi che il client dichiara di aver eseguito.
        """
        e = int(self._epochs_sent.get(cid, self.epochs))
        b = int(self._batch_sent.get(cid, self.batch_size))
        return float(e * max(1, int(n) // max(1, b)))

    # ------------------------------------------------------ config per client
    def _per_client_config(
        self, cid: int, cfg: ConfigRecord, server_round: int
    ) -> None:
        # [B] sostituisce le mezze epoche di SAGE: la riduzione per SoC basso
        # e' gia' dentro h_i, in forma continua invece che a gradino.
        e, b = self._plan.get(cid, (int(self.epochs), int(self.batch_size)))
        cfg["epochs"] = int(e)
        cfg["batch-size"] = int(b)

    def _mark_half(self, selected: list[int]) -> None:
        """Disattivata: qui la modulazione di E e' continua."""
        self._half = set()

    # -------------------------------------------------- aggregazione FedNova
    def aggregate_train(self, server_round, replies):
        """Aggregazione step-normalized (FedNova, Wang et al. NeurIPS 2020).

            w_i     = p_i = |d_i| / |D|          peso dai soli dati
            tau_i   = E_i * floor(n_i / B_i)     passi di gradiente compiuti
            tau_eff = sum_i p_i * tau_i          passi efficaci
            x <- x + tau_eff * sum_i p_i * (Delta_i / tau_i)

        La divisione per tau_i e' la normalizzazione: sottopesa chi ha fatto
        piu' passi. tau_eff davanti evita che il passo globale si rimpicciolisca
        per effetto della divisione.

        [B] QUANDO tau E' UGUALE PER TUTTI QUESTA E' ESATTAMENTE FedAvg:
        tau_eff = tau e la formula si riduce alla media pesata per p_i.
        Verificato numericamente.

        [B] tau DIPENDE ANCHE DA n_i, non solo da E e B. Con partizioni
        Dirichlet n varia di 4-5 volte, quindi la distorsione esiste anche a
        E e B fissi: FedAvg standard e' gia' sbilanciato nelle baseline, e i
        paper originali non lo correggono. Qui la correzione e' applicata al
        solo SAGE-smart, dove le epoche variano per costruzione.

        [B] CIO' CHE CONTA E' LA DISPERSIONE DI tau, NON IL SUO VALORE.
        L'update di ogni client viene moltiplicato per tau_eff/tau_i: il
        client con tau minimo e' amplificato di quel rapporto. Nella campagna
        con la ricarica, un client collegato alla rete riceveva le epoche
        massime a prescindere dai suoi dati e arrivava a tau=390 contro i 13
        degli altri; tau_eff saliva a 120 e l'update del client marginale
        pesava trenta volte quello del client in carica. L'accuratezza
        globale e' crollata da 0.50 a 0.14 per i diciotto round in cui quel
        device e' rimasto in carica, risalendo appena e' uscito. Da qui la
        colonna tau_spread: e' la grandezza da sorvegliare.

        tau_i lo calcola il SERVER: conosce E_i e B_i perche' li ha assegnati,
        n_i arriva nelle metriche. Nessuna modifica al client.
        [LIMITE] assume che il client esegua E_i epoche complete. Se un giorno
        potra' fermarsi prima, dovra' riportare i passi effettivi.
        """
        arrays, metrics = super().aggregate_train(server_round, replies)

        # [B] metriche della policy nel CSV per round, PRIMA dell'uscita
        # anticipata: servono anche quando l'aggregazione non viene applicata.
        # super() ha gia' assegnato _round_stats[round], quindi qui si aggiunge
        # sopra con update() invece di sovrascrivere.
        if self._plan:
            self._round_stats.setdefault(int(server_round), {}).update(
                self._policy_stats())

        if arrays is None or self._global_arrays is None:
            return arrays, metrics

        # [B] le CHIAVI vanno conservate: costruire un ArrayRecord da una lista
        # gli fa assegnare nomi posizionali "0","1",... e il load_state_dict in
        # global_evaluate non li riconosce. Leggere per chiave garantisce anche
        # lo stesso ordine fra record globale e locali.
        keys = list(self._global_arrays.keys())
        glob = [self._global_arrays[k].numpy().astype(np.float64) for k in keys]
        rows = []
        # [B] stesse reply che super() ha mediato: per sage_smart sono tutte,
        # sage_smart2 esclude opt-out e update fuori envelope
        for r in self._aggregatable(replies):
            if not r.has_content():
                continue
            cid = self._node_to_cid.get(int(r.metadata.src_node_id))
            if cid is None:
                continue
            n = float(self._metric(r, "num-examples", "num_examples", default=0))
            tau = self._steps(cid, n, r)
            if n <= 0 or tau <= 0:
                continue
            key = sorted(r.content.array_records)[0]
            rec = r.content.array_records[key]
            rows.append((n, tau, [rec[k].numpy() for k in keys]))

        if len(rows) < 2:
            return arrays, metrics

        tot = sum(n for n, _, _ in rows)
        p = [n / tot for n, _, _ in rows]
        tau_eff = sum(pi * tau for pi, (_, tau, _) in zip(p, rows))

        # [B] se tutti hanno lo stesso tau, questa e' identica a FedAvg:
        # lo si verifica lasciando E fisso e confrontando i due CSV.
        out = [g.copy() for g in glob]
        for pi, (_, tau, w) in zip(p, rows):
            for j, wj in enumerate(w):
                out[j] += tau_eff * pi * (wj.astype(np.float64) - glob[j]) / tau

        taus = [tau for _, tau, _ in rows]
        self._round_stats.setdefault(int(server_round), {}).update({
            "tau_min": int(min(taus)),
            "tau_max": int(max(taus)),
            "tau_eff": float(tau_eff),
            # rapporto di amplificazione del client marginale: sopra ~6-8
            # l'aggregazione comincia a essere dominata da un solo update
            "tau_spread": float(max(taus) / max(1.0, min(taus))),
        })

        log(INFO, "round %s: FedNova su %s client | tau %s-%s, tau_eff %.1f, "
            "spread %.1f", server_round, len(rows), int(min(taus)),
            int(max(taus)), tau_eff, max(taus) / max(1.0, min(taus)))

        merged = ArrayRecord()
        for k, o in zip(keys, out):
            merged[k] = Array(o.astype(np.float32))
        return merged, metrics