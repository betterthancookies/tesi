"""Cio' che il CLIENT sa del proprio device: la vista locale di sage_smart2.

In un telefono vero queste grandezze vengono dal sistema operativo: il SoC
dal fuel gauge della batteria, il workload dallo scheduler, il profilo
hardware da come e' fatto il device. Nel simulatore il mondo fisico vive nel
processo del server (WorldState), quindi la lettura arriva al client dentro
il messaggio di training, in un record separato chiamato "device".

[B] SEPARAZIONE DEI CANALI. Il messaggio di sage_smart2 porta due record:
    "config"  la PROPOSTA del server (envelope di E e B): e' protocollo FL.
    "device"  i SENSORI del device simulato: e' fisica, non protocollo.
La strategia non legge mai il record "device" per decidere: lo scrive il
livello fisico (WorldState) per conto del device. In un deployment reale il
record sparirebbe e il client leggerebbe le stesse grandezze dal sistema.

[B] LA PREVISIONE COINCIDE CON LA FISICA. `soc_after` replica esattamente
WorldState.apply_round + apply_communication: stesse funzioni (importate da
world_state), stesso ordine, stesso clamp a zero. Il SoC finale che il
client comunica al server e' quindi quello che il mondo applica davvero, e la
colonna `s2_soc_err` del CSV lo verifica a ogni round (deve stare a ~1e-12).
"""

from __future__ import annotations

from dataclasses import dataclass

from device.battery_model import delta_soc
from device.communication_model import comm_power_w, comm_time_s
from device.compute_model import probe_time_s, training_time_s
from device.device_profile import DeviceProfile
from device.world_state import WorldState, effective_profile, training_power_w

RECORD_KEY = "device"


@dataclass(frozen=True)
class DeviceReading:
    """Istantanea del device presa al momento dell'esecuzione (late binding)."""

    soc: float
    workload: float
    charging: bool
    profile: DeviceProfile

    # -------------------------------------------------- lato mondo (server)
    @classmethod
    def from_world(cls, world: WorldState, cid: int) -> "DeviceReading":
        return cls(
            soc=float(world.snapshot(cid).soc),
            workload=float(world.utilization(cid)),
            charging=bool(world.is_charging(cid)),
            profile=world.profile(cid),
        )

    def to_record(self) -> dict:
        p = self.profile
        return {
            "soc": float(self.soc),
            "workload": float(self.workload),
            "charging": bool(self.charging),
            "cid": int(p.cid),
            "capacity-mah": float(p.battery_capacity_mah),
            "macs-per-s": float(p.macs_per_s),
            "peukert-n": float(p.peukert_n),
            "tier": str(p.tier),
        }

    # -------------------------------------------------- lato device (client)
    @classmethod
    def from_record(cls, rec) -> "DeviceReading":
        return cls(
            soc=float(rec["soc"]),
            workload=float(rec["workload"]),
            charging=bool(rec["charging"]),
            profile=DeviceProfile(
                cid=int(rec["cid"]),
                battery_capacity_mah=float(rec["capacity-mah"]),
                macs_per_s=float(rec["macs-per-s"]),
                peukert_n=float(rec["peukert-n"]),
                tier=str(rec["tier"]),
            ),
        )

    # ------------------------------------------------------------ previsioni
    def work_time_s(self, epochs: float, batch_size: int, n_samples: int,
                    probe_samples: int = 0) -> float:
        """Durata del lavoro locale col workload ATTUALE.

        `epochs` puo' essere frazionario: un training interrotto a meta'
        epoca dura la frazione corrispondente (training_time_s e' lineare
        nelle epoche).
        """
        p = effective_profile(self.profile, self.workload)
        return (training_time_s(p, epochs, n_samples, batch_size)
                + probe_time_s(p, probe_samples))

    def soc_after(self, dt_s: float, batch_size: int) -> tuple[float, float]:
        """(SoC dopo il training, SoC dopo l'upload), come li applica il mondo.

        In carica la batteria non scende: il device e' collegato alla rete.
        """
        if self.charging:
            return self.soc, self.soc
        s1 = self._drain(self.soc, training_power_w(batch_size), dt_s)
        if s1 <= 0.0:
            # il mondo marca il device come morto e non addebita l'upload
            return 0.0, 0.0
        s2 = self._drain(s1, comm_power_w(), comm_time_s())
        return s1, s2

    def predict(self, epochs: float, batch_size: int, n_samples: int,
                probe_samples: int = 0) -> tuple[float, float]:
        """(durata, SoC finale dopo l'upload) di un piano (E, B)."""
        dt = self.work_time_s(epochs, batch_size, n_samples, probe_samples)
        return dt, self.soc_after(dt, batch_size)[1]

    def _drain(self, soc: float, power_w: float, dt_s: float) -> float:
        d = delta_soc(power_w, dt_s, self.profile, soc)
        return max(0.0, min(1.0, soc + d))
