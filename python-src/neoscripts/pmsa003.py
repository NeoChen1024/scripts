#!/usr/bin/env python

"""Read Plantower PMSA003 particulate measurements from a serial port."""

import struct
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

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


def particulate_aqi(concentration: int, pollutant: str) -> int:
    """Calculate a Taiwan AQI particulate sub-index from one concentration."""
    if concentration < 0:
        raise ValueError("concentration cannot be negative")

    if pollutant == "pm25":
        # PM2.5 is defined to 0.1 ug/m3; sensor values are already integral.
        value = float(Decimal(concentration).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))
        ranges = PM25_RANGES
    elif pollutant == "pm10":
        value = int(Decimal(concentration).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
        ranges = PM10_RANGES
    else:
        raise ValueError(f"unsupported pollutant: {pollutant}")

    for (concentration_low, concentration_high), (aqi_low, aqi_high) in zip(ranges, AQI_RANGES):
        if value <= concentration_high:
            aqi = (aqi_high - aqi_low) / (concentration_high - concentration_low) * (value - concentration_low) + aqi_low
            return int(Decimal(str(aqi)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    return 500


def format_measurement(measurement: Measurement) -> str:
    pm25_aqi = particulate_aqi(measurement.pm25, "pm25")
    pm10_aqi = particulate_aqi(measurement.pm10, "pm10")
    return (
        f"ug/m3: PM1={measurement.pm1}, PM2.5={measurement.pm25}, PM10={measurement.pm10} | "
        f"count/0.1L >um(0.3={measurement.particles_03} 0.5={measurement.particles_05} "
        f"1.0={measurement.particles_10} 2.5={measurement.particles_25} "
        f"5.0={measurement.particles_50} 10={measurement.particles_100}) | "
        f"AQI={max(pm25_aqi, pm10_aqi)} (PM2.5={pm25_aqi} PM10={pm10_aqi})"
    )


@click.command(help="Read PMSA003 measurements and estimate Taiwan AQI from each sample.")
@click.argument("port")
@click.option("--baudrate", "-b", default=9600, show_default=True, type=click.IntRange(min=1), help="Serial baud rate.")
@click.option("--timeout", "-t", default=5.0, show_default=True, type=click.FloatRange(min=0.1), help="Read timeout in seconds.")
@click.option("--count", "-n", default=0, show_default=True, type=click.IntRange(min=0), help="Number of readings; 0 runs forever.")
def main(port: str, baudrate: int, timeout: float, count: int) -> None:
    emitted = 0
    try:
        with serial.Serial(port, baudrate=baudrate, timeout=timeout) as serial_port:
            while count == 0 or emitted < count:
                click.echo(format_measurement(read_frame(serial_port)))
                emitted += 1
    except (SerialException, TimeoutError) as exc:
        raise click.ClickException(str(exc)) from exc
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
