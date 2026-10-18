#!/usr/bin/env python

"""Inspect a GMC-1130, synchronize its clock, or watch its heartbeat counts."""

from collections import deque
from datetime import datetime
from math import isfinite
from struct import unpack_from
from time import sleep

import click
import serial
from serial import SerialException

BAUDRATE = 115200
TIMEOUT = 3.0
VERSION_MIN_LENGTH = 14
DATETIME_LENGTH = 7
HEARTBEAT_LENGTH = 2
WINDOW_TICKS = 60
CONFIG_LENGTH = 256
SERIAL_LENGTH = 7


def read_exact(port: serial.Serial, size: int, description: str) -> bytes:
    data = port.read(size)
    if len(data) != size:
        raise TimeoutError(f"{description}: expected {size} bytes, received {len(data)}")
    return data


def command(port: serial.Serial, payload: bytes) -> None:
    port.write(payload)
    port.flush()


def read_version(port: serial.Serial) -> str:
    command(port, b"<GETVER>>")
    version_raw = read_exact(port, VERSION_MIN_LENGTH, "GETVER")
    original_timeout = port.timeout
    try:
        port.timeout = 0.2
        version_raw += port.read(1)
    finally:
        port.timeout = original_timeout
    return version_raw.decode("ascii", errors="replace")


def open_device(port_name: str, timeout: float) -> serial.Serial:
    port = serial.Serial(port_name, baudrate=BAUDRATE, timeout=timeout, write_timeout=timeout)
    try:
        # A previous client may have left unsolicited heartbeat bytes enabled.
        command(port, b"<HEARTBEAT0>>")
        sleep(0.2)
        port.reset_input_buffer()
        version = read_version(port)
        if not version.startswith("GMC-1130"):
            raise ValueError(f"unexpected device version: {version!r}")
    except Exception:
        port.close()
        raise
    return port


def read_device_time(port: serial.Serial) -> datetime | None:
    command(port, b"<GETDATETIME>>")
    raw = read_exact(port, DATETIME_LENGTH, "GETDATETIME")
    if raw[-1] != 0xAA:
        raise ValueError(f"GETDATETIME returned an invalid terminator: {raw.hex()}")
    try:
        return datetime(2000 + raw[0], *raw[1:6])
    except ValueError:
        return None


def set_device_time(port: serial.Serial, when: datetime) -> None:
    if not 2000 <= when.year <= 2255:
        raise ValueError("host year is outside the GMC-1130 RTC range")
    fields = bytes((when.year - 2000, when.month, when.day, when.hour, when.minute, when.second))
    command(port, b"<SETDATETIME" + fields + b">>")
    acknowledgement = read_exact(port, 1, "SETDATETIME")
    if acknowledgement != b"\xaa":
        raise ValueError(f"SETDATETIME returned an invalid acknowledgement: {acknowledgement.hex()}")


def read_config(port: serial.Serial) -> bytes:
    command(port, b"<GETCFG>>")
    return read_exact(port, CONFIG_LENGTH, "GETCFG")


def read_voltage(port: serial.Serial) -> float:
    command(port, b"<GETVOLT>>")
    return read_exact(port, 1, "GETVOLT")[0] / 10


def format_config(config: bytes) -> list[str]:
    # The field order through offset 61 follows GQ's published 300/320 enum.
    # GMC-1130 places its last-save timestamp at 62..67; later models may differ.
    if len(config) != CONFIG_LENGTH:
        raise ValueError(f"expected {CONFIG_LENGTH} configuration bytes, got {len(config)}")

    lines = []

    def add(offset: int, name: str, value: str) -> None:
        lines.append(f"  0x{offset:02X}  {name:<29} {value}")

    def byte(offset: int, name: str, values: dict[int, str] | None = None) -> None:
        value = config[offset]
        description = values.get(value) if values else None
        add(offset, name, f"{description} ({value})" if description else f"{value} (0x{value:02X})")

    def number(offset: int, width: int, name: str, unit: str = "") -> None:
        data = config[offset : offset + width]
        value = int.from_bytes(data, "big")
        add(offset, name, "unset (all 0xFF)" if all(part == 0xFF for part in data) else f"{value}{unit}")

    def floating(offset: int, name: str) -> None:
        data = config[offset : offset + 4]
        value = unpack_from("<f", config, offset)[0]
        add(offset, name, f"{value:.3f} µSv/h" if isfinite(value) else f"raw {data.hex(' ')}")

    yes_no = {0: "off", 1: "on"}
    lines.append("Controls")
    byte(0, "Stored power flag")
    byte(1, "Alarm", yes_no)
    byte(2, "Speaker", yes_no)
    byte(3, "Graphic mode")
    byte(4, "Backlight timeout code")
    byte(5, "Idle title mode")
    byte(26, "Idle display mode")
    byte(33, "Swivel display", yes_no)
    byte(44, "Power saving mode")
    byte(45, "Sensitivity mode")
    number(46, 2, "Counter delay")
    byte(48, "Display contrast")
    byte(51, "Auto sensitivity threshold")
    byte(52, "Large font mode")
    byte(53, "LCD backlight level")
    byte(54, "Reverse display")
    byte(55, "Motion detection")
    byte(56, "Battery type code")
    byte(57, "Baud rate code")
    byte(58, "CPM speaker calibration")
    byte(59, "Graphic drawing mode")
    byte(60, "LED", yes_no)
    byte(61, "High CPM calibration")

    lines.append("Alarm")
    number(6, 2, "CPM threshold", " CPM")
    floating(27, "Dose threshold")
    byte(31, "Threshold type", {0: "CPM", 1: "µSv/h"})

    lines.append("Calibration")
    for index, offset in enumerate((8, 14, 20), start=1):
        number(offset, 2, f"Point {index} count", " CPM")
        floating(offset + 2, f"Point {index} dose")

    lines.append("History")
    byte(32, "Save interval", {0: "off", 1: "every second", 2: "every minute", 3: "every hour"})
    number(38, 3, "Flash write address")
    number(41, 3, "Flash read address")
    number(49, 2, "Maximum CPM", " CPM")
    try:
        stamp = datetime(2000 + config[62], *config[63:68]).isoformat(sep=" ")
    except ValueError:
        stamp = f"invalid (raw {config[62:68].hex(' ')})"
    add(62, "Last saved timestamp", stamp)
    return lines


@click.group()
@click.option("--port", "port_name", required=True, help="GMC-1130 serial port, preferably /dev/serial/by-id/...")
@click.option(
    "--timeout", default=TIMEOUT, show_default=True, type=click.FloatRange(min=0.1), help="Serial read timeout in seconds."
)
@click.pass_context
def main(ctx: click.Context, port_name: str, timeout: float) -> None:
    """GMC-1130 serial tools."""
    ctx.obj = (port_name, timeout)


@main.command()
@click.option("--raw", is_flag=True, help="Also print all 256 configuration bytes in hexadecimal.")
@click.pass_obj
def info(connection: tuple[str, float], raw: bool) -> None:
    """Show device settings and current battery voltage."""
    try:
        with open_device(*connection) as port:
            version = read_version(port)
            command(port, b"<GETSERIAL>>")
            serial_number = read_exact(port, SERIAL_LENGTH, "GETSERIAL").hex()
            voltage = read_voltage(port)
            config = read_config(port)
        click.echo(f"Device: {version}")
        click.echo(f"Serial: {serial_number}")
        click.echo(f"Battery: {voltage:.1f} V")
        click.echo(f"Configuration: {len(config)} bytes")
        for line in format_config(config):
            click.echo(line)
        if raw:
            click.echo("Raw configuration")
            for offset in range(0, len(config), 16):
                click.echo(f"  {offset:02X}: {config[offset:offset + 16].hex(' ')}")
    except (SerialException, TimeoutError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc


@main.command("sync-time")
@click.option("--check", is_flag=True, help="Show clock difference without setting the device RTC.")
@click.pass_obj
def sync_time(connection: tuple[str, float], check: bool) -> None:
    """Set the device RTC from the host's local clock."""
    try:
        with open_device(*connection) as port:
            before_host = datetime.now().astimezone()
            before_device = read_device_time(port)
            after_host = datetime.now().astimezone()
            midpoint = before_host + (after_host - before_host) / 2
            if before_device is None:
                click.echo("Device RTC before: invalid date/time")
            else:
                difference = (before_device - midpoint.replace(tzinfo=None)).total_seconds()
                click.echo(f"Device RTC before: {before_device.isoformat(sep=' ')} (host difference {difference:+.2f} s)")
            click.echo(f"Host local time:   {after_host.isoformat(sep=' ', timespec='seconds')}")
            if check:
                return

            target = datetime.now().astimezone()
            set_device_time(port, target)
            verify_before = datetime.now().astimezone()
            verified = read_device_time(port)
            verify_after = datetime.now().astimezone()
            if verified is None:
                raise ValueError("device RTC remains invalid after SETDATETIME")
            verify_midpoint = verify_before + (verify_after - verify_before) / 2
            difference = (verified - verify_midpoint.replace(tzinfo=None)).total_seconds()
            click.echo(f"Device RTC after:  {verified.isoformat(sep=' ')} (host difference {difference:+.2f} s)")
            if abs(difference) > 2:
                raise ValueError("device RTC verification differs from host by more than 2 seconds")
    except (SerialException, TimeoutError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc


@main.command()
@click.option(
    "--count",
    default=0,
    show_default=True,
    type=click.IntRange(min=0),
    help="Stop after this many ticks; 0 runs until interrupted.",
)
@click.pass_obj
def stream(connection: tuple[str, float], count: int) -> None:
    """Print CPS and rolling CPM over the latest 60 received ticks."""
    ticks: deque[int] = deque(maxlen=WINDOW_TICKS)
    try:
        with open_device(*connection) as port:
            command(port, b"<HEARTBEAT1>>")
            try:
                received = 0
                while count == 0 or received < count:
                    frame = read_exact(port, HEARTBEAT_LENGTH, "heartbeat")
                    cps = int.from_bytes(frame, "big") & 0x3FFF
                    ticks.append(cps)
                    received += 1
                    cpm = str(sum(ticks)) if len(ticks) == WINDOW_TICKS else f"warming {len(ticks)}/60"
                    click.echo(f"{datetime.now().astimezone().isoformat(timespec='seconds')}  CPS={cps}  CPM={cpm}")
            finally:
                command(port, b"<HEARTBEAT0>>")
    except KeyboardInterrupt:
        pass
    except (SerialException, TimeoutError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc


if __name__ == "__main__":
    main()
