"""
Simulated IoT vitals stream generator.

Produces physiologically-plausible, temporally-correlated vital sign
readings using a bounded random walk, with configurable anomaly injection.
"""

import random
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Iterator


@dataclass
class VitalsReading:
    timestamp: str
    heart_rate_bpm: float
    spo2_percent: float
    temperature_celsius: float
    respiratory_rate_bpm: float
    is_anomalous: bool


class VitalsSimulator:
    BOUNDS = {
        "heart_rate_bpm": (40.0, 160.0),
        "spo2_percent": (85.0, 100.0),
        "temperature_celsius": (34.5, 40.0),
        "respiratory_rate_bpm": (8.0, 30.0),
    }

    NORMAL_BASELINE = {
        "heart_rate_bpm": 72.0,
        "spo2_percent": 98.0,
        "temperature_celsius": 36.8,
        "respiratory_rate_bpm": 16.0,
    }

    def __init__(self, anomaly_probability: float = 0.05, seed=None):
        self._rng = random.Random(seed)
        self._anomaly_probability = anomaly_probability
        self._state = dict(self.NORMAL_BASELINE)

    def _walk(self, key: str, step_size: float) -> float:
        low, high = self.BOUNDS[key]
        delta = self._rng.uniform(-step_size, step_size)
        new_value = self._state[key] + delta
        self._state[key] = max(low, min(high, new_value))
        return self._state[key]

    def _maybe_inject_anomaly(self) -> bool:
        if self._rng.random() < self._anomaly_probability:
            key = self._rng.choice(list(self.BOUNDS.keys()))
            low, high = self.BOUNDS[key]
            self._state[key] = self._rng.choice([low, high]) + self._rng.uniform(-2, 2)
            self._state[key] = max(low, min(high, self._state[key]))
            return True
        return False

    def next_reading(self) -> VitalsReading:
        anomalous = self._maybe_inject_anomaly()
        heart_rate = self._walk("heart_rate_bpm", step_size=2.0)
        spo2 = self._walk("spo2_percent", step_size=0.5)
        temperature = self._walk("temperature_celsius", step_size=0.1)
        resp_rate = self._walk("respiratory_rate_bpm", step_size=1.0)

        return VitalsReading(
            timestamp=datetime.now(timezone.utc).isoformat(),
            heart_rate_bpm=round(heart_rate, 1),
            spo2_percent=round(spo2, 1),
            temperature_celsius=round(temperature, 1),
            respiratory_rate_bpm=round(resp_rate, 1),
            is_anomalous=anomalous,
        )

    def stream(self, interval_seconds: float = 1.0, max_readings=None) -> Iterator[VitalsReading]:
        count = 0
        while max_readings is None or count < max_readings:
            yield self.next_reading()
            count += 1
            if max_readings is None or count < max_readings:
                time.sleep(interval_seconds)


if __name__ == "__main__":
    sim = VitalsSimulator(anomaly_probability=0.15, seed=42)
    for reading in sim.stream(interval_seconds=0, max_readings=10):
        flag = " <-- ANOMALY" if reading.is_anomalous else ""
        print(asdict(reading), flag)
