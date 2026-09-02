#!/usr/bin/env python3
"""Partition a directory tree into capacity-bounded reflink/hardlink shards."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shutil
import stat
import statistics
import sys
import tempfile
from array import array
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Iterator

from tqdm import tqdm

TOOL = "dirshard"
VERSION = 1
FICLONE = 0x40049409
MAX_OPTIMAL_UNITS = 20
SCAN_PROGRESS_UPDATE_INTERVAL = 256
DP_PROGRESS_UPDATE_INTERVAL = 4096
HUMAN_SIZE_COLUMN_WIDTH = 11  # e.g. "1,023.9 GiB"
FILE_COUNT_COLUMN_WIDTH = 20
SIZE_MULTIPLIERS = {
    "": 1,
    "K": 1024,
    "M": 1024**2,
    "G": 1024**3,
    "T": 1024**4,
    "P": 1024**5,
    "E": 1024**6,
}


class DirshardError(Exception):
    """A user-facing error."""


@dataclass
class Unit:
    path: str
    kind: str
    size: int = 0
    files: int = 0
    directories: int = 0
    symlinks: int = 0


@dataclass
class PackingResult:
    bins: list[list[Unit]]
    requested: str
    implementation: str
    optimal_proven: bool
    lower_bound: int
    ffd_upper_bound: int
    dp_states: int = 0


@dataclass
class ScanState:
    special_policy: str
    special_paths: set[str] = field(default_factory=set)
    files: int = 0
    bytes: int = 0

    def handle_special(self, path: str) -> None:
        if self.special_policy == "error":
            raise DirshardError(f"special file encountered: {path!r}")
        self.special_paths.add(path)


def warn(message: str) -> None:
    print(f"{TOOL}: warning: {message}", file=sys.stderr)


def human_size(value: int | float) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB"):
        if abs(size) < 1024 or unit == "EiB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:,.1f} {unit}"
        size /= 1024
    raise AssertionError("unreachable")


def parse_size(text: str) -> int:
    """Parse K/M/G/... suffixes as IEC powers of 1024.

    For example, 10G, 10GB, and 10GiB all mean 10 * 1024**3 bytes.
    """
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([KMGTPE]?)(?:I?B)?\s*", text, re.IGNORECASE)
    if not match:
        raise argparse.ArgumentTypeError(f"invalid size: {text!r}")
    try:
        value = Decimal(match.group(1)) * SIZE_MULTIPLIERS[match.group(2).upper()]
    except (InvalidOperation, KeyError):
        raise argparse.ArgumentTypeError(f"invalid size: {text!r}") from None
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("size must be greater than zero")
    return result


def positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid integer: {text!r}") from None
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return value


def classify_mode(mode: int) -> str:
    if stat.S_ISDIR(mode):
        return "dir"
    if stat.S_ISREG(mode):
        return "file"
    if stat.S_ISLNK(mode):
        return "symlink"
    return "special"


def scandir_sorted(path: str) -> list[os.DirEntry[str]]:
    with os.scandir(path) as entries:
        return sorted(entries, key=lambda entry: entry.name)


def walk_tree(
    root: str,
    root_relpath: str = "",
    excluded_directories: set[str] | None = None,
) -> Iterator[tuple[str, str, os.stat_result, str]]:
    """Yield descendants as (relative path, kind, stat, absolute path).

    Paths are deterministic, symlinks are never followed, and excluded
    directories are neither yielded nor traversed.
    """
    exclusions = excluded_directories or set()
    stack: list[tuple[str, str]] = [("", root)]
    while stack:
        prefix, directory = stack.pop()
        children: list[tuple[str, str]] = []
        for entry in scandir_sorted(directory):
            relpath = f"{prefix}{entry.name}"
            source_relpath = f"{root_relpath}/{relpath}" if root_relpath else relpath
            entry_stat = entry.stat(follow_symlinks=False)
            kind = classify_mode(entry_stat.st_mode)
            if kind == "dir" and source_relpath in exclusions:
                continue
            yield relpath, kind, entry_stat, entry.path
            if kind == "dir":
                children.append((f"{relpath}/", entry.path))
        stack.extend(reversed(children))


def normalize_directory_points(src: str, values: list[str], option: str) -> set[str]:
    src_real = os.path.realpath(src)
    points: set[str] = set()

    for value in values:
        if os.path.isabs(value):
            raise DirshardError(f"{option} must be relative to SRC: {value!r}")
        normalized = os.path.normpath(value)
        if normalized in ("", "."):
            raise DirshardError(f"{option} cannot name SRC itself")
        if normalized == ".." or normalized.startswith(f"..{os.sep}"):
            raise DirshardError(f"{option} escapes SRC: {value!r}")

        current = src
        for component in normalized.split(os.sep):
            current = os.path.join(current, component)
            if os.path.islink(current):
                raise DirshardError(f"{option} cannot pass through a symlink: {value!r}")

        full = os.path.join(src, normalized)
        if not os.path.isdir(full):
            raise DirshardError(f"{option} is not a directory: {value!r}")
        full_real = os.path.realpath(full)
        if full_real != src_real and not full_real.startswith(f"{src_real}{os.sep}"):
            raise DirshardError(f"{option} escapes SRC: {value!r}")
        points.add(os.path.relpath(full_real, src_real).replace(os.sep, "/"))
    return points


def normalize_split_points(src: str, values: list[str]) -> set[str]:
    split_points = normalize_directory_points(src, values, "--split-at")
    ordered = sorted(split_points)
    for index, parent in enumerate(ordered):
        for child in ordered[index + 1 :]:
            if child.startswith(f"{parent}/"):
                raise DirshardError(f"overlapping --split-at values: {parent!r} already contains {child!r}")
    return split_points


def normalize_exclusions(src: str, values: list[str]) -> set[str]:
    exclusions = normalize_directory_points(src, values, "--exclude")
    return {path for path in exclusions if not any(path != parent and path.startswith(f"{parent}/") for parent in exclusions)}


def validate_split_exclusions(split_points: set[str], exclusions: set[str]) -> None:
    for split_point in sorted(split_points):
        for exclusion in sorted(exclusions):
            if split_point == exclusion or split_point.startswith(f"{exclusion}/"):
                raise DirshardError(f"--split-at {split_point!r} is inside excluded directory {exclusion!r}")


def discover_units(
    src: str,
    split_points: set[str],
    excluded_directories: set[str],
    scan: ScanState,
) -> list[Unit]:
    units: list[Unit] = []

    def is_split_ancestor(path: str) -> bool:
        return any(point.startswith(f"{path}/") for point in split_points)

    def descend(directory: str, prefix: str) -> None:
        for entry in scandir_sorted(directory):
            relpath = f"{prefix}{entry.name}"
            entry_stat = entry.stat(follow_symlinks=False)
            kind = classify_mode(entry_stat.st_mode)
            if kind == "dir" and relpath in excluded_directories:
                continue
            if kind == "dir" and (relpath in split_points or is_split_ancestor(relpath)):
                descend(entry.path, f"{relpath}/")
            elif kind == "dir":
                units.append(Unit(relpath, "tree"))
            elif kind == "file":
                units.append(Unit(relpath, "file"))
            elif kind == "symlink":
                units.append(Unit(relpath, "symlink"))
            else:
                scan.handle_special(relpath)

    descend(src, "")
    return units


def measure_units(
    src: str,
    units: list[Unit],
    excluded_directories: set[str],
    scan: ScanState,
) -> None:
    with tqdm(
        total=len(units),
        desc="scanning units",
        unit="unit",
        dynamic_ncols=True,
        disable=None,
    ) as progress:
        seen_entries = 0
        for unit in units:
            source = os.path.join(src, unit.path)
            source_stat = os.lstat(source)
            if unit.kind == "file":
                unit.size = source_stat.st_size
                unit.files = 1
            elif unit.kind == "symlink":
                unit.symlinks = 1
            else:
                unit.directories = 1
                for relpath, kind, entry_stat, _ in walk_tree(
                    source,
                    unit.path,
                    excluded_directories,
                ):
                    seen_entries += 1
                    full_relpath = f"{unit.path}/{relpath}"
                    if kind == "file":
                        unit.size += entry_stat.st_size
                        unit.files += 1
                    elif kind == "dir":
                        unit.directories += 1
                    elif kind == "symlink":
                        unit.symlinks += 1
                    else:
                        scan.handle_special(full_relpath)
                    if seen_entries % SCAN_PROGRESS_UPDATE_INTERVAL == 0:
                        progress.set_postfix(
                            files=scan.files + unit.files,
                            size=human_size(scan.bytes + unit.size),
                            refresh=True,
                        )
            scan.files += unit.files
            scan.bytes += unit.size
            progress.update(1)
            progress.set_postfix(files=scan.files, size=human_size(scan.bytes), refresh=False)


def calculate_unit_statistics(units: list[Unit]) -> dict[str, int | float]:
    if not units:
        return {"count": 0, "min": 0, "average": 0.0, "max": 0, "stddev": 0.0}
    sizes = [unit.size for unit in units]
    return {
        "count": len(sizes),
        "min": min(sizes),
        "average": statistics.fmean(sizes),
        "max": max(sizes),
        "stddev": statistics.pstdev(sizes),
    }


def report_unit_statistics(unit_statistics: dict[str, int | float]) -> None:
    print(
        f"{TOOL}: packing units={unit_statistics['count']}; "
        f"size min={human_size(unit_statistics['min'])}, "
        f"avg={human_size(unit_statistics['average'])}, "
        f"max={human_size(unit_statistics['max'])}, "
        f"stddev={human_size(unit_statistics['stddev'])}",
        file=sys.stderr,
    )


def ordered_units(units: list[Unit]) -> list[Unit]:
    return sorted(units, key=lambda unit: (-unit.size, unit.path))


def normalize_bins(bins: list[list[Unit]]) -> list[list[Unit]]:
    result = [bin_units for bin_units in bins if bin_units]
    for bin_units in result:
        bin_units.sort(key=lambda unit: unit.path)
    result.sort(
        key=lambda bin_units: (
            -sum(unit.size for unit in bin_units),
            tuple(unit.path for unit in bin_units),
        )
    )
    return result


def pack_ffd(units: list[Unit], capacity: int) -> list[list[Unit]]:
    bins: list[list[Unit]] = []
    loads: list[int] = []
    for unit in ordered_units(units):
        for index, load in enumerate(loads):
            if load + unit.size <= capacity:
                bins[index].append(unit)
                loads[index] += unit.size
                break
        else:
            bins.append([unit])
            loads.append(unit.size)
    return normalize_bins(bins)


def reconstruct_dp_bins(units: list[Unit], predecessor: array, started_new_bin: bytearray) -> list[list[Unit]]:
    mask = (1 << len(units)) - 1
    transitions: list[tuple[int, bool]] = []
    while mask:
        item = predecessor[mask]
        if item < 0:
            raise RuntimeError("subset DP predecessor chain is incomplete")
        transitions.append((item, bool(started_new_bin[mask])))
        mask ^= 1 << item
    transitions.reverse()

    bins: list[list[Unit]] = [[]]
    for item, starts_new in transitions:
        if starts_new:
            bins.append([])
        bins[-1].append(units[item])
    return normalize_bins(bins)


def pack_optimal(units: list[Unit], capacity: int, ffd_bins: list[list[Unit]]) -> PackingResult:
    if not units:
        return PackingResult([], "optimal", "lower-bound", True, 0, 0)

    total_size = sum(unit.size for unit in units)
    lower_bound = max(1, (total_size + capacity - 1) // capacity)
    upper_bound = len(ffd_bins)
    if lower_bound == upper_bound:
        return PackingResult(
            ffd_bins,
            "optimal",
            "lower-bound",
            True,
            lower_bound,
            upper_bound,
        )

    if len(units) > MAX_OPTIMAL_UNITS:
        raise DirshardError(
            f"--packing optimal needs subset DP for this input, but there are {len(units)} units; "
            f"the supported maximum is {MAX_OPTIMAL_UNITS} (use --packing ffd or add --split-at less aggressively)"
        )
    if capacity > (1 << 64) - 1:
        raise DirshardError("--packing optimal supports capacities up to 2^64-1 bytes")

    dp_units = ordered_units(units)
    state_count = 1 << len(dp_units)
    unreachable = 255
    bin_count = bytearray([unreachable]) * state_count
    last_load = array("Q", [0]) * state_count
    predecessor = array("b", [-1]) * state_count
    started_new_bin = bytearray(state_count)
    bin_count[0] = 1

    with tqdm(
        total=state_count,
        desc="optimal packing",
        unit="state",
        dynamic_ncols=True,
        disable=None,
    ) as progress:
        progress.set_postfix(lower=lower_bound, ffd=upper_bound, refresh=False)
        pending_updates = 0
        full_mask = state_count - 1
        for mask in range(state_count):
            used_bins = bin_count[mask]
            remaining = full_mask ^ mask
            while remaining:
                bit = remaining & -remaining
                item = bit.bit_length() - 1
                next_mask = mask | bit
                combined = last_load[mask] + dp_units[item].size
                if combined <= capacity:
                    candidate_bins = used_bins
                    candidate_load = combined
                    starts_new = 0
                else:
                    candidate_bins = used_bins + 1
                    candidate_load = dp_units[item].size
                    starts_new = 1
                if candidate_bins < bin_count[next_mask] or (
                    candidate_bins == bin_count[next_mask] and candidate_load < last_load[next_mask]
                ):
                    bin_count[next_mask] = candidate_bins
                    last_load[next_mask] = candidate_load
                    predecessor[next_mask] = item
                    started_new_bin[next_mask] = starts_new
                remaining ^= bit
            pending_updates += 1
            if pending_updates == DP_PROGRESS_UPDATE_INTERVAL:
                progress.update(pending_updates)
                pending_updates = 0
        if pending_updates:
            progress.update(pending_updates)

    bins = reconstruct_dp_bins(dp_units, predecessor, started_new_bin)
    if len(bins) != bin_count[-1]:
        raise RuntimeError("subset DP reconstruction disagrees with its optimum")
    return PackingResult(
        bins,
        "optimal",
        "subset-dp",
        True,
        lower_bound,
        upper_bound,
        state_count,
    )


def make_packing(units: list[Unit], capacity: int, requested: str) -> PackingResult:
    oversized = [unit for unit in units if unit.size > capacity]
    if oversized:
        details = ", ".join(f"{unit.path} ({human_size(unit.size)})" for unit in oversized[:5])
        if len(oversized) > 5:
            details += f", and {len(oversized) - 5} more"
        raise DirshardError(f"unit exceeds capacity {human_size(capacity)}: {details}; split it further with --split-at")

    ffd_bins = pack_ffd(units, capacity)
    total_size = sum(unit.size for unit in units)
    lower_bound = max(1, (total_size + capacity - 1) // capacity) if units else 0
    if requested == "ffd":
        return PackingResult(
            ffd_bins,
            "ffd",
            "first-fit-decreasing",
            len(ffd_bins) == lower_bound,
            lower_bound,
            len(ffd_bins),
        )
    return pack_optimal(units, capacity, ffd_bins)


def validate_prefix(prefix: str) -> None:
    separators = {os.sep}
    if os.altsep:
        separators.add(os.altsep)
    if any(separator in prefix for separator in separators):
        raise DirshardError("--prefix cannot contain a path separator")
    if prefix in (".", ".."):
        raise DirshardError("--prefix cannot be '.' or '..'")


def shard_names(count: int, prefix: str, digits: int) -> list[str]:
    return [f"{prefix}{number:0{digits}d}" for number in range(1, count + 1)]


def unit_record(unit: Unit) -> dict[str, object]:
    return {
        "path": unit.path,
        "kind": unit.kind,
        "size": unit.size,
        "files": unit.files,
        "directories": unit.directories,
        "symlinks": unit.symlinks,
    }


def build_manifest(
    src: str,
    dest: str,
    capacity: int,
    unit_statistics: dict[str, int | float],
    excluded_directories: set[str],
    prefix: str,
    digits: int,
    link_mode: str,
    special_policy: str,
    special_paths: set[str],
    packing: PackingResult,
) -> dict[str, object]:
    names = shard_names(len(packing.bins), prefix, digits)
    shards = []
    for name, bin_units in zip(names, packing.bins):
        shards.append(
            {
                "name": name,
                "size": sum(unit.size for unit in bin_units),
                "files": sum(unit.files for unit in bin_units),
                "units": [unit_record(unit) for unit in bin_units],
            }
        )
    return {
        "tool": TOOL,
        "version": VERSION,
        "source": os.path.realpath(src),
        "destination": os.path.abspath(dest),
        "capacity": capacity,
        "unit_statistics": unit_statistics,
        "excluded_directories": sorted(excluded_directories),
        "naming": {"prefix": prefix, "digits": digits},
        "link_mode": link_mode,
        "special_files": {
            "policy": special_policy,
            "skipped": sorted(special_paths) if special_policy == "skip" else [],
        },
        "packing": {
            "requested": packing.requested,
            "implementation": packing.implementation,
            "optimal_proven": packing.optimal_proven,
            "lower_bound": packing.lower_bound,
            "ffd_upper_bound": packing.ffd_upper_bound,
            "dp_states": packing.dp_states,
        },
        "total_size": sum(shard["size"] for shard in shards),
        "total_files": sum(shard["files"] for shard in shards),
        "shards": shards,
    }


def report_special_files(paths: set[str]) -> None:
    if not paths:
        return
    ordered = sorted(paths)
    for path in ordered[:20]:
        warn(f"skipping special file {path!r}")
    if len(ordered) > 20:
        warn(f"skipping {len(ordered) - 20} additional special files")


def print_plan(manifest: dict[str, object], verbose: bool) -> None:
    shards = manifest["shards"]
    capacity = manifest["capacity"]
    for shard in shards:
        efficiency = f"  packing_efficiency={shard['size'] / capacity * 100:.2f}%"
        print(
            f"{shard['name']}  {human_size(shard['size']):>{HUMAN_SIZE_COLUMN_WIDTH}}  "
            f"{shard['files']:>{FILE_COUNT_COLUMN_WIDTH}} files{efficiency}"
        )
        if verbose:
            for unit in shard["units"]:
                print(
                    f"    {unit['path']}  {human_size(unit['size']):>{HUMAN_SIZE_COLUMN_WIDTH}}  "
                    f"{unit['files']:>{FILE_COUNT_COLUMN_WIDTH}} files"
                )
    packing = manifest["packing"]
    print(f"total: {len(manifest['shards'])} shards, {human_size(manifest['total_size'])}, " f"{manifest['total_files']} files")
    print(
        f"packing: {packing['implementation']}, optimal_proven={str(packing['optimal_proven']).lower()}, "
        f"lower_bound={packing['lower_bound']}, ffd_upper_bound={packing['ffd_upper_bound']}"
    )


class Materializer:
    def __init__(
        self,
        src: str,
        link_mode: str,
        special_policy: str,
        special_paths: set[str],
        excluded_directories: set[str],
        verbose: bool,
        progress: tqdm,
    ) -> None:
        self.src = src
        self.link_mode = link_mode
        self.special_policy = special_policy
        self.special_paths = special_paths
        self.excluded_directories = excluded_directories
        self.verbose = verbose
        self.progress = progress
        self.linked_bytes = 0

    def handle_special(self, relpath: str) -> None:
        if self.special_policy == "error":
            raise DirshardError(f"special file encountered: {relpath!r}")
        self.special_paths.add(relpath)

    @staticmethod
    def copy_directory_metadata(directory_metadata: dict[str, str]) -> None:
        for destination, source in sorted(
            directory_metadata.items(),
            key=lambda item: item[0].count(os.sep),
            reverse=True,
        ):
            shutil.copystat(source, destination, follow_symlinks=False)

    def ensure_parent_directories(
        self,
        shard_dir: str,
        relpath: str,
        directory_metadata: dict[str, str],
    ) -> None:
        components = relpath.split("/")[:-1]
        source = self.src
        destination = shard_dir
        for component in components:
            source = os.path.join(source, component)
            destination = os.path.join(destination, component)
            if not os.path.exists(destination):
                os.mkdir(destination, 0o700)
            directory_metadata[destination] = source

    def create_regular_file(
        self,
        source: str,
        destination: str,
        source_stat: os.stat_result,
        inode_map: dict[tuple[int, int], str],
    ) -> None:
        if self.link_mode == "hardlink":
            os.link(source, destination)
        else:
            inode_key = (source_stat.st_dev, source_stat.st_ino)
            existing = inode_map.get(inode_key)
            if existing is not None:
                os.link(existing, destination)
            else:
                source_fd = os.open(source, os.O_RDONLY | os.O_CLOEXEC)
                try:
                    destination_fd = os.open(
                        destination,
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
                        0o600,
                    )
                    try:
                        fcntl.ioctl(destination_fd, FICLONE, source_fd)
                    finally:
                        os.close(destination_fd)
                finally:
                    os.close(source_fd)
                shutil.copystat(source, destination, follow_symlinks=False)
                inode_map[inode_key] = destination
        self.linked_bytes += source_stat.st_size
        self.progress.update(1)
        if self.progress.n % 128 == 0:
            self.progress.set_postfix(size=human_size(self.linked_bytes), refresh=False)

    @staticmethod
    def create_symlink(source: str, destination: str) -> None:
        os.symlink(os.readlink(source), destination)
        shutil.copystat(source, destination, follow_symlinks=False)

    def materialize_unit(
        self,
        shard_dir: str,
        unit: Unit,
        directory_metadata: dict[str, str],
        inode_map: dict[tuple[int, int], str],
    ) -> None:
        source_root = os.path.join(self.src, unit.path)
        destination_root = os.path.join(shard_dir, unit.path)
        self.ensure_parent_directories(shard_dir, unit.path, directory_metadata)

        if unit.kind == "file":
            self.create_regular_file(source_root, destination_root, os.lstat(source_root), inode_map)
            return
        if unit.kind == "symlink":
            self.create_symlink(source_root, destination_root)
            return

        os.mkdir(destination_root, 0o700)
        directory_metadata[destination_root] = source_root
        for relpath, kind, entry_stat, source in walk_tree(
            source_root,
            unit.path,
            self.excluded_directories,
        ):
            destination = os.path.join(destination_root, relpath)
            full_relpath = f"{unit.path}/{relpath}"
            if kind == "dir":
                os.mkdir(destination, 0o700)
                directory_metadata[destination] = source
            elif kind == "file":
                self.create_regular_file(source, destination, entry_stat, inode_map)
            elif kind == "symlink":
                self.create_symlink(source, destination)
            else:
                self.handle_special(full_relpath)

    def materialize_shard(self, shard_dir: str, units: list[Unit]) -> None:
        directory_metadata: dict[str, str] = {}
        inode_map: dict[tuple[int, int], str] = {}
        for unit in units:
            if self.verbose:
                tqdm.write(f"{TOOL}: linking unit {unit.path}", file=sys.stderr)
            self.materialize_unit(shard_dir, unit, directory_metadata, inode_map)
        self.copy_directory_metadata(directory_metadata)


def materialize(
    src: str,
    dest: str,
    prefix: str,
    digits: int,
    packing: PackingResult,
    link_mode: str,
    special_policy: str,
    special_paths: set[str],
    excluded_directories: set[str],
    total_files: int,
    verbose: bool,
    manifest_factory,
) -> dict[str, object]:
    parent = os.path.dirname(dest) or "."
    if not os.path.isdir(parent):
        raise DirshardError(f"DEST parent does not exist: {parent!r}")
    staging = tempfile.mkdtemp(prefix=f".{os.path.basename(dest)}.dirshard-staging-", dir=parent)
    try:
        with tqdm(
            total=total_files,
            desc="creating links",
            unit="file",
            dynamic_ncols=True,
            disable=None,
        ) as progress:
            worker = Materializer(
                src,
                link_mode,
                special_policy,
                special_paths,
                excluded_directories,
                verbose,
                progress,
            )
            for name, bin_units in zip(shard_names(len(packing.bins), prefix, digits), packing.bins):
                shard_dir = os.path.join(staging, name)
                os.mkdir(shard_dir, 0o755)
                worker.materialize_shard(shard_dir, bin_units)
            progress.set_postfix(size=human_size(worker.linked_bytes), refresh=False)

        manifest = manifest_factory()
        manifest_path = os.path.join(staging, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as output:
            json.dump(manifest, output, indent=2, ensure_ascii=False)
            output.write("\n")
        os.rename(staging, dest)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def validate_paths(src: str, dest: str, dry_run: bool) -> tuple[str, str]:
    source = os.path.abspath(src)
    destination = os.path.abspath(dest)
    if not os.path.isdir(source):
        raise DirshardError(f"SRC is not a directory: {src!r}")
    source_real = os.path.realpath(source)
    destination_real = os.path.realpath(destination)
    if destination_real == source_real or destination_real.startswith(f"{source_real}{os.sep}"):
        raise DirshardError("DEST cannot be SRC or a descendant of SRC")
    if not dry_run:
        parent = os.path.dirname(destination) or "."
        if not os.path.isdir(parent):
            raise DirshardError(f"DEST parent does not exist: {parent!r}")
    if not dry_run and os.path.lexists(destination):
        raise DirshardError(f"DEST already exists: {dest!r}")
    return source, destination


def run(args: argparse.Namespace) -> dict[str, object]:
    validate_prefix(args.prefix)
    src, dest = validate_paths(args.src, args.dest, args.dry_run)
    split_points = normalize_split_points(src, args.split_at)
    excluded_directories = normalize_exclusions(src, args.exclude)
    validate_split_exclusions(split_points, excluded_directories)
    scan = ScanState(args.special_files)
    units = discover_units(src, split_points, excluded_directories, scan)
    measure_units(src, units, excluded_directories, scan)
    unit_statistics = calculate_unit_statistics(units)
    report_unit_statistics(unit_statistics)
    report_special_files(scan.special_paths)
    packing = make_packing(units, args.capacity, args.packing)

    def make_manifest() -> dict[str, object]:
        return build_manifest(
            src,
            dest,
            args.capacity,
            unit_statistics,
            excluded_directories,
            args.prefix,
            args.digits,
            args.link,
            args.special_files,
            scan.special_paths,
            packing,
        )

    if args.dry_run:
        manifest = make_manifest()
        if args.json:
            print(json.dumps(manifest, indent=2, ensure_ascii=False))
        else:
            print_plan(manifest, args.verbose)
        return manifest

    manifest = materialize(
        src,
        dest,
        args.prefix,
        args.digits,
        packing,
        args.link,
        args.special_files,
        scan.special_paths,
        excluded_directories,
        scan.files,
        args.verbose,
        make_manifest,
    )
    if args.json:
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
    else:
        print_plan(manifest, args.verbose)
        print(
            f"{TOOL}: created {len(manifest['shards'])} shards in {dest!r}; "
            f"{human_size(manifest['total_size'])}, {manifest['total_files']} files, "
            f"link={args.link}, packing={packing.implementation}"
        )
    return manifest


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=TOOL,
        description="Partition a directory tree into capacity-bounded reflink/hardlink shards.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="examples:\n"
        "  dirshard SRC DEST -c 100GiB\n"
        "  dirshard SRC DEST -c 100GiB -s b2 --packing optimal\n"
        "  dirshard SRC DEST -c 100GiB -p volume- -d 3 -l hardlink",
    )
    parser.add_argument("src", help="source directory (never modified)")
    parser.add_argument("dest", help="new output root containing the shard directories")
    parser.add_argument(
        "-c",
        "--capacity",
        required=True,
        type=parse_size,
        metavar="SIZE",
        help="maximum apparent size per shard; K/M/G/T use powers of 1024 (10G = 10 * 1024^3 bytes)",
    )
    parser.add_argument(
        "-s",
        "--split-at",
        action="append",
        default=[],
        metavar="PATH",
        help="make the direct children of PATH (relative to SRC) separate units; repeatable",
    )
    parser.add_argument(
        "-e",
        "--exclude",
        action="append",
        default=[],
        metavar="PATH",
        help="exclude PATH (relative to SRC) and its entire directory tree; repeatable",
    )
    parser.add_argument(
        "--packing",
        choices=("ffd", "optimal"),
        default="ffd",
        help="packing algorithm (default: ffd; optimal uses exact subset DP)",
    )
    parser.add_argument(
        "-l",
        "--link",
        choices=("reflink", "hardlink"),
        default="reflink",
        help="file materialization method; failures are fatal (default: reflink)",
    )
    parser.add_argument(
        "--special-files",
        choices=("skip", "error"),
        default="skip",
        help="how to handle FIFO/device/socket entries (default: skip)",
    )
    parser.add_argument(
        "-p",
        "--prefix",
        default="shard-",
        help="shard directory name prefix (default: shard-)",
    )
    parser.add_argument(
        "-d",
        "--digits",
        type=positive_int,
        default=2,
        metavar="N",
        help="minimum zero-padded width of shard numbers (default: 2)",
    )
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="scan and print the plan without creating DEST",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="list the packing units assigned to each shard",
    )
    parser.add_argument("--json", action="store_true", help="print the plan/result as JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        run(args)
    except DirshardError as error:
        print(f"{TOOL}: error: {error}", file=sys.stderr)
        return 1
    except OSError as error:
        print(f"{TOOL}: error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(f"{TOOL}: interrupted", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
