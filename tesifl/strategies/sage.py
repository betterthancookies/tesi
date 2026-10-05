from __future__ import annotations

from logging import INFO

import numpy as np
import pulp
from flwr.app import ConfigRecord, Message
from flwr.common import log

from device.battery_model import V_NOMINAL
from strategies.fedavg import PhysicalFedAvg


def _js(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon in base 2, con smoothing additivo come nel paper."""
    eps = 1e-9
    p, q = (p + eps) / (p + eps).sum(), (q + eps) / (q + eps).sum()
    m = 0.5 * (p + q)
    kl = lambda a, b: float(np.sum(a * (np.log(a) - np.log(b))) / np.log(2.0))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


class PhysicalSAGE(PhysicalFedAvg):
    """SAGE (Savoia et al., Applied Energy 410, 2026) sul mondo fisico.

        max  sum_i y_i * (a*E_i + b*D_i + c*R_i)          Eq. (2)
        s.t. k <= sum_i y_i <= min(k + r, 0.2*N)          (2a) (2b) (2c)
             E_i >= 0.2                                   (2e)
             a + b + c = 1                                (2g)

    E_i energia residua normalizzata sulla capacita' nominale, D_i JSD media
    verso gli altri client, R_i quota rinnovabile. r e' 0-indicizzato nel
    paper, server_round e' 1-indicizzato in Flower: da cui k + r - 1. N e' il
    numero TOTALE di client, come nel paper (non i soli eleggibili).

    Meta' epoche locali ai selezionati con E_i < 0.3 (Alg. 1, righe 17-20).

    [B] con score non negativi e soli vincoli di cardinalita' l'ottimo
    dell'ILP coincide con i top-k per score. La formulazione e' mantenuta per
    fedelta' al paper, non perche' aggiunga capacita' di selezione.

    CONTABILITA' DELL'ENERGIA (il punto del confronto con `sage_soc`):
      E_i(0) = SoC_0, come nel paper, dove l'energia iniziale e' un dato noto
      (estratta da una gaussiana e normalizzata sulla capacita').
      E_i(t) = SoC_0 - sum(Wh LINEARI spesi) / capacita' nominale.
    I Wh lineari sono P/eta * dt: e' il modello "potenza per tempo" del
    paper, che ignora Peukert e quindi sottostima il consumo reale di
    ~1.26x (training) e ~1.66x (comunicazione). Il SoC vero del world state
    NON entra nella selezione: quella e' la variante della tesi.

    ADATTAMENTI DICHIARATI
    - R_i e' simulato stocasticamente in [0,1], come nel paper (Sez. 5.2): li'
      non e' una misura, e' una simulazione di irraggiamento/vento.
    - D_i: il paper la assegna a tutti i client all'inizializzazione (Alg. 1
      riga 2). Se `class_distributions` non e' fornita al costruttore, si
      ricade sull'accumulo incrementale dalle metriche di training, e i mai
      osservati ereditano la D media dei noti per non restare esclusi.
    - il paper ricarica ogni 10 round; il world state non ha ricarica, quindi
      E_i e' monotona decrescente.

    [LIMITE DICHIARATO] nel paper anche i client non selezionati consumano in
    idle e il loro E_i cala. Qui non c'e' drain di background, quindi E_i cala
    solo per chi lavora.
    """

    def __init__(
        self,
        *args,
        sage_a: float = 0.5,
        sage_b: float = 0.2,
        sage_c: float = 0.3,
        sage_k: int = 2,
        eligibility_threshold: float = 0.2,
        half_epochs_threshold: float = 0.3,
        max_fraction: float = 0.2,
        class_distributions: dict[int, np.ndarray] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.a, self.b, self.c = float(sage_a), float(sage_b), float(sage_c)
        if abs(self.a + self.b + self.c - 1.0) > 1e-9:
            raise ValueError("vincolo (2g): a + b + c = 1")
        self.k = int(sage_k)
        self.elig_th = float(eligibility_threshold)
        self.half_th = float(half_epochs_threshold)
        self.max_frac = float(max_fraction)

        self._dist: dict[int, np.ndarray] = {}   # cid -> distribuzione classi
        self._div: dict[int, float] = {}         # cid -> JSD media
        self._renew: dict[int, float] = {}       # cid -> quota rinnovabile
        self._used_wh: dict[int, float] = {}     # cid -> Wh LINEARI cumulati
        self._cap_wh: dict[int, float] = {}
        self._half: set[int] = set()

        if class_distributions:
            for cid, d in class_distributions.items():
                arr = np.asarray(d, dtype=float)
                self._dist[int(cid)] = arr / arr.sum()
            self._recompute_div()

    # ------------------------------------------------------------ componenti
    def _capacity(self, cid: int) -> float:
        if cid not in self._cap_wh:
            mah = float(self.world.profile(cid).battery_capacity_mah)
            self._cap_wh[cid] = mah / 1000.0 * V_NOMINAL
        return self._cap_wh[cid]

    def _energy(self, cid: int) -> float:
        """E_i = SoC_0 - Wh_lineari / capacita', clampata in [0, 1]."""
        cap = self._capacity(cid)
        if cap <= 0.0:
            return 1.0
        soc0 = float(self.world.initial_soc(cid))
        return float(np.clip(soc0 - self._used_wh.get(cid, 0.0) / cap, 0.0, 1.0))

    def _divergence(self, cid: int) -> float:
        if cid in self._div:
            return self._div[cid]
        return float(np.mean(list(self._div.values()))) if self._div else 0.0

    def _step_renewable(self) -> None:
        """R_i evolve stocasticamente in [0,1], un passo per round."""
        for cid in self._node_to_cid.values():
            cur = self._renew.get(cid)
            if cur is None:
                cur = float(self._rng.uniform(0.0, 1.0))
            self._renew[cid] = float(
                np.clip(cur + self._rng.uniform(-0.1, 0.1), 0.0, 1.0)
            )

    # ------------------------------------------------------------- selezione
    def _select_clients(
        self, available: list[int], k: int, server_round: int
    ) -> list[int]:
        if not k:
            return []
        cid_of = self._node_to_cid
        self._step_renewable()

        # (2e): eleggibilita' sull'energia residua stimata
        elig = [n for n in available if self._energy(cid_of[n]) >= self.elig_th]
        if not elig:
            log(INFO, "sage round %s: nessun eleggibile (soglia %.2f)",
                server_round, self.elig_th)
            return []

        # (2a) k + r con r 0-indicizzato, (2b) 0.2*N con N TOTALE (paper),
        # e comunque non piu' degli eleggibili
        m = len(elig)
        n_total = len(cid_of)
        hi = min(self.k + server_round - 1, int(np.floor(self.max_frac * n_total)), m)
        lo = min(self.k, hi)

        score = {
            n: self.a * self._energy(cid_of[n])
            + self.b * self._divergence(cid_of[n])
            + self.c * self._renew.get(cid_of[n], 0.0)
            for n in elig
        }

        prob = pulp.LpProblem("SAGE_Select", pulp.LpMaximize)
        y = [pulp.LpVariable(f"y_{i}", cat="Binary") for i in range(m)]
        prob += pulp.lpSum(y[i] * score[elig[i]] for i in range(m))
        prob += pulp.lpSum(y) >= lo
        prob += pulp.lpSum(y) <= hi
        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        selected = [elig[i] for i in range(m) if int(pulp.value(y[i]) or 0) == 1]
        if not selected:  # solver fallito: top-lo per score
            selected = sorted(elig, key=lambda n: -score[n])[:lo]

        self._mark_half(selected)
        log(INFO, "sage round %s: %s eleggibili su %s, %s selezionati "
            "(cap %s, %s a epoche dimezzate)", server_round, m, len(available),
            len(selected), hi, len(self._half))
        return selected

    def _mark_half(self, selected: list[int]) -> None:
        self._half = {
            self._node_to_cid[n] for n in selected
            if self._energy(self._node_to_cid[n]) < self.half_th
        }

    def _per_client_config(
        self, cid: int, cfg: ConfigRecord, server_round: int
    ) -> None:
        if cid in self._half:
            cfg["epochs"] = max(1, int(self.epochs) // 2)

    # ---------------------------------------------------------------- update
    def _on_round_result(
        self, cid: int, n_ex: int, dt_s: float, e_wh: float, e_lin_wh: float,
        reply: Message, server_round: int,
    ) -> None:
        # contabilita' LINEARE: e' quella del paper, non la carica Peukert
        self._used_wh[cid] = self._used_wh.get(cid, 0.0) + e_lin_wh

        # fallback: distribuzioni non fornite al costruttore
        dist = self._metric(reply, "class-dist", default=None)
        if dist is None:
            return
        arr = np.asarray(list(dist), dtype=float)
        if arr.sum() <= 0:
            return
        self._dist[cid] = arr / arr.sum()
        self._recompute_div()

    def _recompute_div(self) -> None:
        cids = sorted(self._dist)
        n = len(cids)
        if n < 2:
            self._div = {c: 0.0 for c in cids}
            return
        mat = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                mat[i, j] = mat[j, i] = _js(self._dist[cids[i]], self._dist[cids[j]])
        self._div = {c: float(mat[i].sum() / (n - 1)) for i, c in enumerate(cids)}