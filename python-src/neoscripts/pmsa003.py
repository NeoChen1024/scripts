#!/usr/bin/env python

"""Read Plantower PMSA003 particulate measurements from a serial port."""

import json
import struct
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import click
import serial
from serial import SerialException

FRAME_HEADER = b"\x42\x4d"
FRAME_SIZE = 32
FRAME_LENGTH = 28

# Taiwan MOENV AQI breakpoints effective from 2025-01-01.
AQI_RANGES = ((0, 50), (51, 100), (101, 150), (151, 200), (201, 300), (301, 400), (401, 500))
PM25_RANGES = ((0.0, 12.4), (12.5, 30.4), (30.5, 50.4), (50.5, 125.4), (125.5, 225.4), (225.5, 325.4), (325.5, 500.4))
PM10_RANGES = ((0, 30), (31, 75), (76, 190), (191, 354), (355, 424), (425, 504), (505, 604))


@dataclass(frozen=True)
class Measurement:
    standard_pm1: int
    standard_pm25: int
    standard_pm10: int
    pm1: int
    pm25: int
    pm10: int
    particles_03: int
    particles_05: int
    particles_10: int
    particles_25: int
    particles_50: int
    particles_100: int


def parse_frame(frame: bytes) -> Measurement:
    """Validate and decode one 32-byte PMSA003 active-mode frame."""
    if len(frame) != FRAME_SIZE:
        raise ValueError(f"expected {FRAME_SIZE} bytes, got {len(frame)}")
    if frame[:2] != FRAME_HEADER:
        raise ValueError("invalid frame header")
    if int.from_bytes(frame[2:4], "big") != FRAME_LENGTH:
        raise ValueError("invalid frame length")
    if sum(frame[:-2]) & 0xFFFF != int.from_bytes(frame[-2:], "big"):
        raise ValueError("checksum mismatch")

    values = struct.unpack(">13H", frame[4:30])
    return Measurement(
        standard_pm1=values[0],
        standard_pm25=values[1],
        standard_pm10=values[2],
        pm1=values[3],
        pm25=values[4],
        pm10=values[5],
        particles_03=values[6],
        particles_05=values[7],
        particles_10=values[8],
        particles_25=values[9],
        particles_50=values[10],
        particles_100=values[11],
    )


def read_frame(port: serial.Serial) -> Measurement:
    """Read through noise or corrupt frames until a valid frame is found."""
    buffer = bytearray()
    while True:
        byte = port.read(1)
        if not byte:
            raise TimeoutError("timed out waiting for PMSA003 data")
        buffer += byte

        header_at = buffer.find(FRAME_HEADER)
        if header_at < 0:
            buffer[:] = buffer[-1:] if buffer.endswith(FRAME_HEADER[:1]) else b""
            continue
        if header_at:
            del buffer[:header_at]
        if len(buffer) < FRAME_SIZE:
            continue

        candidate = bytes(buffer[:FRAME_SIZE])
        try:
            return parse_frame(candidate)
        except ValueError:
            del buffer[0]


def particulate_aqi(concentration: float, pollutant: str) -> int:
    """Calculate a Taiwan AQI particulate sub-index from one concentration."""
    if concentration < 0:
        raise ValueError("concentration cannot be negative")

    if pollutant == "pm25":
        # PM2.5 is defined to 0.1 ug/m3; sensor values are already integral.
        value = float(Decimal(str(concentration)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))
        ranges = PM25_RANGES
    elif pollutant == "pm10":
        value = int(Decimal(str(concentration)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
        ranges = PM10_RANGES
    else:
        raise ValueError(f"unsupported pollutant: {pollutant}")

    for (concentration_low, concentration_high), (aqi_low, aqi_high) in zip(ranges, AQI_RANGES):
        if value <= concentration_high:
            aqi = (aqi_high - aqi_low) / (concentration_high - concentration_low) * (value - concentration_low) + aqi_low
            return int(Decimal(str(aqi)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    return 500


class AQIHistory:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.hours = {}
        if path.exists():
            try:
                loaded = json.loads(path.read_text())
                self.hours = loaded if isinstance(loaded, dict) else {}
            except (OSError, json.JSONDecodeError):
                pass

    def add(self, measurement: Measurement, now: datetime | None = None) -> None:
        now = now or datetime.now().astimezone()
        key = now.replace(minute=0, second=0, microsecond=0).isoformat()
        bucket = self.hours.setdefault(key, {"pm25": 0, "pm10": 0, "count": 0})
        bucket["pm25"] += measurement.pm25
        bucket["pm10"] += measurement.pm10
        bucket["count"] += 1
        cutoff = now - timedelta(hours=13)
        self.hours = {key: value for key, value in self.hours.items() if datetime.fromisoformat(key) >= cutoff}

    def moving(self, now: datetime | None = None) -> tuple[float, float, int] | None:
        now = now or datetime.now().astimezone()
        current = now.replace(minute=0, second=0, microsecond=0)

        def hourly_averages(window: int) -> list[tuple[float, float]]:
            cutoff = current - timedelta(hours=window - 1)
            return [
                (value["pm25"] / value["count"], value["pm10"] / value["count"])
                for key, value in self.hours.items()
                if datetime.fromisoformat(key) >= cutoff and value["count"]
            ]

        last_4, last_12 = hourly_averages(4), hourly_averages(12)
        if len(last_4) < 2 or len(last_12) < 6:
            return None
        pm25 = 0.5 * sum(row[0] for row in last_12) / len(last_12) + 0.5 * sum(row[0] for row in last_4) / len(last_4)
        pm10 = 0.5 * sum(row[1] for row in last_12) / len(last_12) + 0.5 * sum(row[1] for row in last_4) / len(last_4)
        return pm25, pm10, len(last_12)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.hours, separators=(",", ":")))
        temporary.replace(self.path)


def format_measurement(measurement: Measurement, condition: str, moving: tuple[float, float, int] | None) -> str:
    if condition == "standard":
        pm1, pm25, pm10 = measurement.standard_pm1, measurement.standard_pm25, measurement.standard_pm10
    else:
        pm1, pm25, pm10 = measurement.pm1, measurement.pm25, measurement.pm10
    if moving:
        moving_pm25, moving_pm10, valid_hours = moving
        pm25_aqi = particulate_aqi(moving_pm25, "pm25")
        pm10_aqi = particulate_aqi(moving_pm10, "pm10")
        aqi = f"AQI={max(pm25_aqi, pm10_aqi)} (PM2.5={pm25_aqi} PM10={pm10_aqi} hours={valid_hours})"
    else:
        aqi = "AQI=N/A (insufficient hourly history)"
    return (
        f"{condition} ug/m3: PM1={pm1}, PM2.5={pm25}, PM10={pm10} | "
        f"count/0.1L >um(0.3={measurement.particles_03} 0.5={measurement.particles_05} "
        f"1.0={measurement.particles_10} 2.5={measurement.particles_25} "
        f"5.0={measurement.particles_50} 10={measurement.particles_100}) | "
        f"{aqi}"
    )


@click.command(help="Read PMSA003 measurements and estimate Taiwan AQI from each sample.")
@click.argument("port")
@click.option("--baudrate", "-b", default=9600, show_default=True, type=click.IntRange(min=1), help="Serial baud rate.")
@click.option("--timeout", "-t", default=5.0, show_default=True, type=click.FloatRange(min=0.1), help="Read timeout in seconds.")
@click.option("--count", "-n", default=0, show_default=True, type=click.IntRange(min=0), help="Number of readings; 0 runs forever.")
@click.option(
    "--condition",
    type=click.Choice(["atmospheric", "standard"]),
    default="atmospheric",
    show_default=True,
    help="Mass concentration fields to display; AQI always uses atmospheric values.",
)
@click.option(
    "--history-file",
    type=click.Path(path_type=Path, dir_okay=False),
    default=Path("/tmp/pmsa003-aqi.json"),
    show_default=True,
    help="Persistent hourly AQI history file.",
)
def main(port: str, baudrate: int, timeout: float, count: int, condition: str, history_file: Path) -> None:
    emitted = 0
    history = AQIHistory(history_file)
    last_saved_minute = None
    try:
        with serial.Serial(port, baudrate=baudrate, timeout=timeout) as serial_port:
            while count == 0 or emitted < count:
                measurement = read_frame(serial_port)
                history.add(measurement)
                current_minute = datetime.now().astimezone().replace(second=0, microsecond=0)
                if current_minute != last_saved_minute:
                    history.save()
                    last_saved_minute = current_minute
                click.echo(format_measurement(measurement, condition, history.moving()))
                emitted += 1
    except (SerialException, TimeoutError) as exc:
        raise click.ClickException(str(exc)) from exc
    except KeyboardInterrupt:
        pass
    finally:
        history.save()


if __name__ == "__main__":
    main()
