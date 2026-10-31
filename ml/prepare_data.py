"""Check, download, and split selected Quick, Draw! bitmap categories.

By default this script only checks storage and remote file sizes. Pass
--download to fetch the configured category files and build reproducible splits.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import json
import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(__file__).with_name("dataset_config.json")
DATA_ROOT = ROOT / "data" / "quickdraw"
RAW_DIR = DATA_ROOT / "raw"
PROCESSED_DIR = DATA_ROOT / "processed"
REQUEST_TIMEOUT = 30
IMAGE_BYTES = 28 * 28
RANGE_HEADER_BYTES = 4096


def load_config(path: Path = CONFIG_PATH) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    classes = config.get("classes")
    if not isinstance(classes, list) or not classes or len(classes) != len(set(classes)):
        raise ValueError("Config must contain a non-empty list of unique classes.")
    if any(not isinstance(label, str) or not label.strip() for label in classes):
        raise ValueError("Each class label must be a non-empty string.")
    limit = config.get("max_examples_per_class")
    if not isinstance(limit, int) or limit < 3:
        raise ValueError("max_examples_per_class must be an integer of at least 3.")
    split = config.get("split", {})
    if set(split) != {"train", "validation", "test"}:
        raise ValueError("Split must define train, validation, and test.")
    if any(not isinstance(value, (int, float)) or value <= 0 for value in split.values()):
        raise ValueError("Split fractions must be positive numbers.")
    if not np.isclose(sum(split.values()), 1.0):
        raise ValueError("Split fractions must sum to 1.")
    return config


def remote_file_size(url: str) -> int | None:
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
            length = response.headers.get("Content-Length")
            return int(length) if length else None
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        raise RuntimeError(f"Could not inspect remote dataset file {url}: {exc}") from exc


def inspect_downloads(config: dict) -> tuple[list[tuple[str, Path, int | None]], int]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    files: list[tuple[str, Path, int | None]] = []
    bytes_to_download = 0
    base_url = config["source_url"].rstrip("/") + "/"
    for label in config["classes"]:
        destination = RAW_DIR / f"{label}.npy"
        if destination.exists():
            size = destination.stat().st_size
            try:
                with destination.open("rb") as handle:
                    version = np.lib.format.read_magic(handle)
                    shape, fortran_order, dtype = np.lib.format._read_array_header(handle, version)
                    expected_size = handle.tell() + int(np.prod(shape)) * dtype.itemsize
                valid = (
                    len(shape) == 2
                    and shape[1] == IMAGE_BYTES
                    and dtype == np.dtype(np.uint8)
                    and not fortran_order
                    and expected_size == size
                )
            except (OSError, ValueError, EOFError):
                valid = False
            if valid:
                files.append((label, destination, size))
                continue
            print(f"Removing incomplete local bitmap: {destination.name}", flush=True)
            destination.unlink()
        size = remote_file_size(base_url + urllib.parse.quote(label) + ".npy")
        files.append((label, destination, size))
        if size is not None:
            # Only the requested rows are fetched using HTTP range requests.
            bytes_to_download += min(size, config["max_examples_per_class"] * IMAGE_BYTES + RANGE_HEADER_BYTES)
    if any(size is None for _, path, size in files if not path.exists()):
        raise RuntimeError("A source file did not report its size; refusing an unbounded download.")
    return files, bytes_to_download


def format_bytes(value: int) -> str:
    amount = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if amount < 1024 or unit == "TB":
            return f"{amount:.1f} {unit}"
        amount /= 1024
    return f"{amount:.1f} TB"


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Stream to a temporary file and rename only after a complete response.
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".part", delete=False) as temp:
        temp_path = Path(temp.name)
        try:
            with urllib.request.urlopen(url, timeout=REQUEST_TIMEOUT) as response:
                shutil.copyfileobj(response, temp)
            temp.flush()
            os.fsync(temp.fileno())
        except BaseException:
            temp_path.unlink(missing_ok=True)
            raise
    temp_path.replace(destination)


def fetch_bitmap_sample(url: str, source_size: int, count: int, seed: int) -> np.ndarray:
    """Fetch a reproducible, spread-out sample from a remote NPY matrix via ranges."""
    header_request = urllib.request.Request(url, headers={"Range": f"bytes=0-{RANGE_HEADER_BYTES - 1}"})
    with urllib.request.urlopen(header_request, timeout=REQUEST_TIMEOUT) as response:
        if response.status != 206:
            raise RuntimeError("Dataset server did not honor the byte-range request; refusing a full-file transfer.")
        header_data = response.read(RANGE_HEADER_BYTES)
    header_stream = io.BytesIO(header_data)
    try:
        version = np.lib.format.read_magic(header_stream)
        shape, fortran_order, dtype = np.lib.format._read_array_header(header_stream, version)
    except (ValueError, EOFError) as exc:
        raise RuntimeError(f"Could not parse remote NumPy bitmap header: {exc}") from exc
    if len(shape) != 2 or shape[1] != IMAGE_BYTES or dtype != np.dtype(np.uint8) or fortran_order:
        raise RuntimeError(f"Unexpected remote bitmap layout: shape={shape}, dtype={dtype}, fortran={fortran_order}.")
    row_count = int(shape[0])
    data_offset = header_stream.tell()
    if data_offset + row_count * IMAGE_BYTES > source_size:
        raise RuntimeError("Remote bitmap header claims more data than the source file contains.")
    if row_count < count:
        raise RuntimeError(f"Remote class has {row_count:,} drawings, fewer than the requested {count:,}.")

    # Pick one random contiguous block from each of ten regions of the file.
    # This spreads the sample across the category's drawing history with few requests.
    block_count = min(10, count)
    block_sizes = np.full(block_count, count // block_count, dtype=np.int64)
    block_sizes[: count % block_count] += 1
    rng = np.random.default_rng(seed)
    samples = []
    region_start = 0
    for block_size in block_sizes.tolist():
        region_end = min(row_count, ((region_start // max(1, row_count // block_count)) + 1) * max(1, row_count // block_count))
        latest_start = region_end - block_size
        if latest_start < region_start:
            raise RuntimeError("Could not form non-overlapping sampled blocks from the remote class file.")
        row_start = int(rng.integers(region_start, latest_start + 1))
        first_byte = data_offset + row_start * IMAGE_BYTES
        last_byte = first_byte + block_size * IMAGE_BYTES - 1
        request = urllib.request.Request(url, headers={"Range": f"bytes={first_byte}-{last_byte}"})
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
            if response.status != 206:
                raise RuntimeError("Dataset server stopped honoring byte ranges; refusing a full-file transfer.")
            chunk = response.read(block_size * IMAGE_BYTES + 1)
        expected_bytes = block_size * IMAGE_BYTES
        if len(chunk) != expected_bytes:
            raise RuntimeError(f"Short ranged response: expected {expected_bytes} bytes, received {len(chunk)}.")
        samples.append(np.frombuffer(chunk, dtype=np.uint8).reshape(block_size, IMAGE_BYTES).copy())
        region_start = region_end
    return np.concatenate(samples, axis=0)


def save_bitmap_sample(destination: Path, images: np.ndarray) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".part", delete=False) as temp:
        temp_path = Path(temp.name)
        try:
            np.save(temp, images, allow_pickle=False)
            temp.flush()
            os.fsync(temp.fileno())
        except BaseException:
            temp_path.unlink(missing_ok=True)
            raise
    temp_path.replace(destination)


def load_bitmap(path: Path) -> np.ndarray:
    try:
        drawings = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Invalid NumPy bitmap file {path}: {exc}") from exc
    if drawings.ndim != 2 or drawings.shape[1] != IMAGE_BYTES:
        raise RuntimeError(
            f"Expected {path.name} to have shape (drawings, 784), got {drawings.shape}."
        )
    if drawings.dtype != np.uint8:
        raise RuntimeError(f"Expected uint8 pixels in {path.name}, got {drawings.dtype}.")
    return drawings


def split_indices(count: int, fractions: dict[str, float], rng: np.random.Generator) -> dict[str, np.ndarray]:
    indices = rng.permutation(count)
    train_end = int(count * fractions["train"])
    validation_end = train_end + int(count * fractions["validation"])
    return {
        "train": indices[:train_end],
        "validation": indices[train_end:validation_end],
        "test": indices[validation_end:],
    }


def prepare(config: dict, files: list[tuple[str, Path, int | None]]) -> dict:
    seed = int(config.get("seed", 42))
    limit = config["max_examples_per_class"]
    class_names = config["classes"]
    class_to_index = {name: index for index, name in enumerate(class_names)}
    rng = np.random.default_rng(seed)
    buckets: dict[str, list[np.ndarray]] = {name: [] for name in ("train", "validation", "test")}
    counts: dict[str, dict[str, int]] = {name: {} for name in buckets}

    for class_name, source_path, _ in files:
        drawings = load_bitmap(source_path)
        if len(drawings) < limit:
            raise RuntimeError(
                f"{class_name} contains only {len(drawings):,} drawings; "
                f"the config requests {limit:,}."
            )
        selected = rng.choice(len(drawings), size=limit, replace=False)
        class_splits = split_indices(limit, config["split"], rng)
        for split_name, indices in class_splits.items():
            # Copy selected rows out of the read-only memory map before stacking.
            buckets[split_name].append(np.asarray(drawings[selected[indices]], dtype=np.uint8))
            counts[split_name][class_name] = int(len(indices))

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for split_name, image_groups in buckets.items():
        images = np.concatenate(image_groups, axis=0)
        labels = np.concatenate([
            np.full(counts[split_name][class_name], class_to_index[class_name], dtype=np.int16)
            for class_name in class_names
        ])
        order = rng.permutation(len(labels))
        images = images[order].reshape((-1, 28, 28))
        labels = labels[order]
        output_path = PROCESSED_DIR / f"{split_name}.npz"
        np.savez_compressed(output_path, images=images, labels=labels)
        print(f"Wrote {output_path.relative_to(ROOT)}: {len(labels):,} examples")

    metadata = {
        "source": config["source"],
        "source_url": config["source_url"],
        "image_shape": [28, 28],
        "pixel_dtype": "uint8",
        "class_names": class_names,
        "class_to_index": class_to_index,
        "seed": seed,
        "max_examples_per_class": limit,
        "counts_by_split_and_class": counts,
        "preprocessing": "Quick, Draw! supplied 28x28 grayscale bitmaps; keep raw intensity values, reshape to 28x28, normalize to [0,1] in the model pipeline.",
    }
    metadata_path = PROCESSED_DIR / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {metadata_path.relative_to(ROOT)}")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download missing source files and create train/validation/test splits.",
    )
    parser.add_argument("--workers", type=int, default=6, help="Number of parallel category downloads (default: 6).")
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 16:
        parser.error("workers must be between 1 and 16.")
    try:
        config = load_config()
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        # Discard only incomplete temporary files left by an interrupted prior run.
        for partial in RAW_DIR.glob("*.part"):
            partial.unlink(missing_ok=True)
        files, bytes_to_download = inspect_downloads(config)
        disk = shutil.disk_usage(DATA_ROOT)
        sample_bytes = len(config["classes"]) * config["max_examples_per_class"] * IMAGE_BYTES
        required_with_margin = int(bytes_to_download + sample_bytes * 1.5 + 256 * 1024**2)
        print(f"Classes: {len(config['classes'])}; sample cap: {config['max_examples_per_class']:,}/class")
        print(f"Estimated ranged sample downloads: {format_bytes(bytes_to_download)}")
        print(f"Prepared pixel data upper bound: {format_bytes(sample_bytes)} (compressed output is usually smaller)")
        print(f"Free disk space: {format_bytes(disk.free)}")
        if disk.free < required_with_margin:
            raise RuntimeError(
                f"Not enough free space for the estimated downloads and preparation margin "
                f"({format_bytes(required_with_margin)} required)."
            )
        print(f"Estimated minimum free space for this operation: {format_bytes(required_with_margin)}")
        if not args.download:
            print("Check only; no dataset files were downloaded. Re-run with --download to continue.")
            return 0

        base_url = config["source_url"].rstrip("/") + "/"
        missing = [(label, path) for label, path, _ in files if not path.exists()]
        print(f"Fetching {len(missing)} category samples with {args.workers} workers...", flush=True)
        for label, path, _ in files:
            if path.exists():
                print(f"Using existing {path.name}", flush=True)

        def download_category(item: tuple[str, Path, int | None]) -> tuple[str, Path]:
            label, path, source_size = item
            url = base_url + urllib.parse.quote(label) + ".npy"
            if source_size is None:
                raise RuntimeError(f"No remote size was available for {label}.")
            category_index = config["classes"].index(label)
            sample = fetch_bitmap_sample(
                url,
                source_size,
                config["max_examples_per_class"],
                int(config.get("seed", 42)) + category_index,
            )
            save_bitmap_sample(path, sample)
            load_bitmap(path)
            return label, path

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(download_category, item) for item in files if not item[1].exists()]
            for future in as_completed(futures):
                label, path = future.result()
                print(f"Fetched and validated {label} sample ({format_bytes(path.stat().st_size)})", flush=True)
        prepare(config, files)
        return 0
    except (OSError, ValueError, RuntimeError, urllib.error.URLError) as exc:
        print(f"Data preparation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
