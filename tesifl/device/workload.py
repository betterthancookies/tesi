"""Workload concorrente: quanto il device e' occupato da ALTRE applicazioni.

Tre catene di Markov a tre stati (idle / light / heavy), una per profilo
d'uso, assegnate ai client in parti uguali e in modo INDIPENDENTE dal tier
hardware: un telefono potente non e' per forza piu' occupato di uno modesto.

w = frazione di capacita' di calcolo presa da altre app. Con questa
definizione il workload entra nel modello senza coefficienti nuovi:

    throughput disponibile = (1 - w) * macs_per_s
    potenza fra i round    = P_idle + w * (P_train - P_idle)

La seconda interpola fra il consumo a riposo e quello a pieno carico.

[B] TRE CATENE INVECE DI UNA. Cosi' differisce anche la persistenza, non solo
la media: un device poco usato resta fermo a lungo, uno molto usato alterna in
fretta. Le diagonali sono alte di proposito -- con transizioni frequenti il
workload diventerebbe rumore bianco round su round, e si perderebbe l'unica
proprieta' che lo rende interessante, cioe' che uno stato dura.

[B] PERCHE' GIUSTIFICA IL LATE-BINDING. Il workload cambia a ogni round e il
server non lo conosce quando manda il modello: e' informazione che solo il
client ha, e solo un istante prima di iniziare. E' il terzo asse del gap
argument della proposta, e l'unico che il simulatore finora non aveva.
"""

from __future__ import annotations

import numpy as np

from device.constants import WORKLOAD_CHAINS, WORKLOAD_STATES

STATE_NAMES = list(WORKLOAD_STATES)          # idle, light, heavy
STATE_UTIL = np.array(list(WORKLOAD_STATES.values()), dtype=float)
PROFILE_NAMES = list(WORKLOAD_CHAINS)        # low, medium, high


class WorkloadModel:
    """Stato di occupazione di ogni client, un passo per round.

    enabled=False congela tutto a w=0: e' l'ablazione che riporta il mondo
    alla popolazione senza workload, per verificare che il resto non cambi.
    """

    def __init__(self, n_clients: int, seed: int, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self.n_clients = int(n_clients)
        self._rng = np.random.default_rng(seed)
        self._chains = {p: np.asarray(WORKLOAD_CHAINS[p], dtype=float)
                        for p in PROFILE_NAMES}
        for p, m in self._chains.items():
            if not np.allclose(m.sum(axis=1), 1.0):
                raise ValueError(f"la catena '{p}' ha righe che non sommano a 1")

        # profili in parti uguali, assegnati con una permutazione: senza,
        # il profilo sarebbe una funzione del cid e quindi correlato alla
        # partizione dei dati, che pure dipende dal cid.
        self._profile = self._assign_equal(PROFILE_NAMES)
        # tutti partono da idle: lo stato iniziale conta poco, le catene
        # convergono alla stazionaria in poche decine di round.
        self._state = {cid: 0 for cid in range(self.n_clients)}

    def _assign_equal(self, names: list[str]) -> dict[int, str]:
        base = [names[i % len(names)] for i in range(self.n_clients)]
        order = self._rng.permutation(self.n_clients)
        return {int(cid): base[i] for i, cid in enumerate(order)}

    # ------------------------------------------------------------ evoluzione
    def step(self) -> None:
        """Una transizione per ogni client. Da chiamare a inizio round."""
        if not self.enabled:
            return
        for cid in range(self.n_clients):
            row = self._chains[self._profile[cid]][self._state[cid]]
            self._state[cid] = int(self._rng.choice(len(row), p=row))

    # -------------------------------------------------------------- lettura
    def utilization(self, cid: int) -> float:
        """w in [0, 1): frazione di calcolo occupata da altre applicazioni."""
        if not self.enabled:
            return 0.0
        return float(STATE_UTIL[self._state[cid]])

    def profile(self, cid: int) -> str:
        return self._profile[cid] if self.enabled else "disabled"

    def state(self, cid: int) -> str:
        return STATE_NAMES[self._state[cid]] if self.enabled else "idle"

    def mean_utilization(self) -> float:
        if not self.enabled:
            return 0.0
        return float(np.mean([self.utilization(c) for c in range(self.n_clients)]))

    def counts_by_profile(self) -> dict[str, int]:
        out = {p: 0 for p in PROFILE_NAMES}
        for cid in range(self.n_clients):
            out[self._profile[cid]] += 1
        return out


def stationary_utilization() -> dict[str, float]:
    """Utilizzazione media attesa di ogni profilo, dall'autovettore della catena.

    Serve a verificare che le matrici diano le medie che ci si aspetta, senza
    doverle simulare: e' il primo dei controlli di validazione.
    """
    out = {}
    for name, m in WORKLOAD_CHAINS.items():
        m = np.asarray(m, dtype=float)
        vals, vecs = np.linalg.eig(m.T)
        i = int(np.argmin(np.abs(vals - 1.0)))
        pi = np.real(vecs[:, i])
        pi = pi / pi.sum()
        out[name] = float(pi @ STATE_UTIL)
    return out


if __name__ == "__main__":
    print("utilizzazione media attesa (dalla distribuzione stazionaria):")
    for k, v in stationary_utilization().items():
        print(f"  {k:>7}: {v:.4f}")

    print("\nverifica per simulazione (30 client, 20000 round):")
    wl = WorkloadModel(n_clients=30, seed=2026)
    acc = {p: [] for p in PROFILE_NAMES}
    for _ in range(20000):
        wl.step()
        for cid in range(30):
            acc[wl.profile(cid)].append(wl.utilization(cid))
    for p, v in acc.items():
        print(f"  {p:>7}: {np.mean(v):.4f}  (n={len(v)})")
    print("\nripartizione dei profili:", wl.counts_by_profile())