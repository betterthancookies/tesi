"""Cio' che il CLIENT sa del proprio device: la vista locale di sage_smart.

In un telefono vero queste grandezze vengono dal sistema operativo: il SoC
dal fuel gauge della batteria, il workload dallo scheduler, il profilo
hardware da come e' fatto il device. Nel simulatore il mondo fisico vive nel
processo del server (WorldState), quindi la lettura arriva al client dentro
il messaggio di training, in un record separato chiamato "device".

[B] SEPARAZIONE DEI CANALI. Il messaggio di sage_smart porta due record:
    "config"  la PROPOSTA del server (envelope di E e B): e' protocollo FL.
    "device"  i SENSORI del device simulato: e' fisica, non protocollo.
La strategia non legge mai il record "device" per decidere: lo scrive il
livello fisico (WorldState) per conto del device. In un deployment reale il
record sparirebbe e il client leggerebbe le stesse grandezze dal sistema.

LA BATTERIA. Il device conosce la propria batteria: `model` e' quella del
mondo (WorldState.battery: "lin", "peuk" o "nm") e il SoC e' quello vero.
`soc_after` prevede il consumo con la stessa funzione con cui il mondo
scarica la batteria (world_state.battery_delta_soc), nello stesso ordine e
con lo stesso clamp: il SoC finale dichiarato coincide con quello vero
(colonna smart_soc_err ~0) in tutti e tre i mondi.
"""

from __future__ import annotations

from dataclasses import dataclass

from device.communication_model import comm_power_w, comm_time_s
from device.compute_model import probe_time_s, training_time_s
from device.device_profile import DeviceProfile
from device.world_state import (
    BATTERIES, BATTERY_NAMES, WorldState, battery_delta_soc,
    effective_profile, training_power_w,
)

RECORD_KEY = "device"
MODELS = BATTERIES
_NAMES = BATTERY_NAMES


@dataclass(frozen=True)
class DeviceReading:
    """Istantanea del device presa al momento dell'esecuzione (late binding)."""

    soc: float
    workload: float
    charging: bool
    profile: DeviceProfile
    model: str = "nm"

    # -------------------------------------------------- lato mondo (server)
    @classmethod
    def from_world(cls, world: WorldState, cid: int) -> "DeviceReading":
        return cls(
            soc=float(world.snapshot(cid).soc),
            workload=float(world.utilization(cid)),
            charging=bool(world.is_charging(cid)),
            profile=world.profile(cid),
            model=world.battery,
        )

    def to_record(self) -> dict:
        p = self.profile
        return {
            "soc": float(self.soc),
            "workload": float(self.workload),
            "charging": bool(self.charging),
            "battery-model": _NAMES[self.model],
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
            model={v: k for k, v in _NAMES.items()}[str(rec["battery-model"])],
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
        """(SoC dopo il training, SoC dopo l'upload) secondo la batteria del
        device. In carica la batteria non scende: e' collegato alla rete."""
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
        d = battery_delta_soc(self.model, power_w, dt_s, self.profile, soc)
        return max(0.0, min(1.0, soc + d))
