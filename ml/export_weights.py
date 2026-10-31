"""Export a trained .keras checkpoint to the portable NumPy format used by the game API.

    python ml/export_weights.py                                   # backend/model/doodle_classifier.keras
    python ml/export_weights.py --model path/to/model.keras --out-prefix backend/model/candidate

Writes ``<prefix>.weights.npz`` and ``<prefix>.arch.json``. The backend then
runs inference with NumPy only (backend/ml/engine.py), so TensorFlow is needed
for training but not for playing.

How weights are read, in order of preference:
1. TensorFlow/Keras, if installed: ``model.get_layer(name).get_weights()``.
2. h5py, if installed, reading ``model.weights.h5`` inside the .keras zip.
3. The system libhdf5 through ctypes (no Python HDF5 package needed).

Keras 3 stores weights under auto-generated, per-class names
(``layers/conv2d``, ``layers/conv2d_1``, ...). Methods 2 and 3 map them back to
config layer names by class and order of appearance, then check every shape
against the config, so a mismatch fails loudly instead of loading silently.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import io
import json
import re
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "backend" / "model" / "doodle_classifier.keras"


def snake_case(name: str) -> str:
    """Keras 3's `naming.to_snake_case`."""
    name = re.sub(r"\W+", "", name)
    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    name = re.sub("([a-z])([A-Z])", r"\1_\2", name).lower()
    return name


# --------------------------------------------------------------- HDF5 readers
def _read_h5_with_h5py(raw: bytes) -> dict[str, np.ndarray]:
    import h5py  # noqa: F401

    out: dict[str, np.ndarray] = {}
    with h5py.File(io.BytesIO(raw), "r") as handle:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset) and obj.shape != () and name.startswith("layers/"):
                out[name] = np.asarray(obj, dtype=np.float32)
        handle.visititems(visit)
    return out


def _find_libhdf5():
    candidates = [ctypes.util.find_library("hdf5"), ctypes.util.find_library("hdf5_serial"),
                  "libhdf5_serial.so.103", "libhdf5.so", "libhdf5.dylib",
                  "/opt/homebrew/lib/libhdf5.dylib", "/usr/local/lib/libhdf5.dylib"]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return ctypes.CDLL(candidate)
        except OSError:
            continue
    raise RuntimeError("Neither h5py, TensorFlow nor a system libhdf5 is available to read the weights.")


def _read_h5_with_ctypes(raw: bytes) -> dict[str, np.ndarray]:
    lib = _find_libhdf5()
    hid_t = ctypes.c_int64
    lib.H5open()
    lib.H5Eset_auto2(hid_t(0), None, None)
    lib.H5Fopen.restype = hid_t
    lib.H5Fopen.argtypes = [ctypes.c_char_p, ctypes.c_uint, hid_t]
    lib.H5Dopen2.restype = hid_t
    lib.H5Dopen2.argtypes = [hid_t, ctypes.c_char_p, hid_t]
    lib.H5Dget_space.restype = hid_t
    lib.H5Dget_space.argtypes = [hid_t]
    lib.H5Sget_simple_extent_ndims.argtypes = [hid_t]
    lib.H5Sget_simple_extent_dims.argtypes = [hid_t, ctypes.c_void_p, ctypes.c_void_p]
    lib.H5Dread.argtypes = [hid_t, hid_t, hid_t, hid_t, hid_t, ctypes.c_void_p]
    for fn in ("H5Sclose", "H5Dclose", "H5Fclose"):
        getattr(lib, fn).argtypes = [hid_t]
    native_float = hid_t.in_dll(lib, "H5T_NATIVE_FLOAT_g")
    callback_type = ctypes.CFUNCTYPE(ctypes.c_int, hid_t, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_void_p)
    lib.H5Lvisit.argtypes = [hid_t, ctypes.c_int, ctypes.c_int, callback_type, ctypes.c_void_p]

    with tempfile.NamedTemporaryFile(suffix=".h5") as temp:
        temp.write(raw)
        temp.flush()
        fid = lib.H5Fopen(temp.name.encode(), 0, hid_t(0))
        if fid < 0:
            raise RuntimeError("libhdf5 could not open the weights file.")
        names: list[str] = []
        keep_alive = callback_type(lambda gid, name, info, data: names.append(name.decode()) or 0)
        lib.H5Lvisit(fid, 0, 0, keep_alive, None)
        out: dict[str, np.ndarray] = {}
        for name in names:
            if not name.startswith("layers/"):
                continue
            did = lib.H5Dopen2(fid, name.encode(), hid_t(0))
            if did < 0:
                continue
            sid = lib.H5Dget_space(did)
            ndims = lib.H5Sget_simple_extent_ndims(sid)
            dims = (ctypes.c_uint64 * max(ndims, 1))()
            lib.H5Sget_simple_extent_dims(sid, dims, None)
            shape = tuple(int(d) for d in dims[:ndims])
            array = np.empty(shape, dtype=np.float32)
            if lib.H5Dread(did, native_float, hid_t(0), hid_t(0), hid_t(0), array.ctypes.data_as(ctypes.c_void_p)) >= 0 and shape:
                out[name] = array
            lib.H5Sclose(sid)
            lib.H5Dclose(did)
        lib.H5Fclose(fid)
    return out


# ---------------------------------------------------------------- mapping
def expected_shapes(cfg: dict, in_channels: int | None) -> list[tuple]:
    kind, c = cfg["class_name"], cfg["config"]
    if kind == "Conv2D":
        k = tuple(c["kernel_size"])
        shapes = [(*k, in_channels, c["filters"])]
        return shapes + ([(c["filters"],)] if c.get("use_bias", True) else [])
    if kind == "Dense":
        return [(in_channels, c["units"])] + ([(c["units"],)] if c.get("use_bias", True) else [])
    if kind == "BatchNormalization":
        count = int(c.get("scale", True)) + int(c.get("center", True)) + 2
        return [(in_channels,)] * count
    return []


def map_h5_to_layers(arch: dict, flat: dict[str, np.ndarray]) -> dict[str, list[np.ndarray]]:
    grouped: dict[str, list[tuple[int, np.ndarray]]] = {}
    for key, value in flat.items():
        parts = key.split("/")
        if len(parts) >= 4 and parts[0] == "layers" and parts[2] == "vars":
            grouped.setdefault(parts[1], []).append((int(parts[3]), value))
    counters: dict[str, int] = {}
    mapped: dict[str, list[np.ndarray]] = {}
    for layer in arch["layers"]:
        base = snake_case(layer["class_name"])
        count = counters.get(base, 0)
        counters[base] = count + 1
        h5_name = base if count == 0 else f"{base}_{count}"
        h5_name = layer["config"]["name"] if layer["config"]["name"] in grouped else h5_name
        if h5_name in grouped:
            mapped[layer["config"]["name"]] = [arr for _, arr in sorted(grouped[h5_name], key=lambda t: t[0])]
    return mapped


def validate_shapes(arch: dict, weights: dict[str, list[np.ndarray]]) -> None:
    """Walk the (chain-shaped) graph tracking channel counts and check each tensor."""
    channels: dict[str, int] = {}
    for layer in arch["layers"]:
        name = layer["config"]["name"]
        kind = layer["class_name"]
        inbound = _inbound(layer)
        in_ch = channels.get(inbound[0]) if inbound else None
        if kind == "InputLayer":
            shape = layer["config"].get("batch_shape") or layer["config"].get("batch_input_shape")
            channels[name] = shape[-1]
            continue
        expected = expected_shapes(layer, in_ch)
        got = [tuple(w.shape) for w in weights.get(name, [])]
        if expected and got != expected:
            raise RuntimeError(f"Weight shape mismatch for {name}: expected {expected}, got {got}")
        if kind in ("Conv2D",):
            channels[name] = layer["config"]["filters"]
        elif kind == "Dense":
            channels[name] = layer["config"]["units"]
        elif kind == "SeparableConv2D":
            channels[name] = layer["config"]["filters"]
        elif kind == "Flatten":
            channels[name] = None  # validated by the engine at run time
        else:
            channels[name] = in_ch


def _inbound(layer_cfg: dict) -> list[str]:
    sys.path.insert(0, str(ROOT))
    from backend.ml.engine import _inbound_names

    return _inbound_names(layer_cfg)


def read_keras_archive(model_path: Path) -> tuple[dict, dict[str, list[np.ndarray]], str]:
    with zipfile.ZipFile(model_path) as archive:
        config = json.loads(archive.read("config.json"))
        raw = archive.read("model.weights.h5")
    arch = {
        "name": config["config"].get("name", "model"),
        "layers": config["config"]["layers"],
        "input_layers": config["config"].get("input_layers"),
        "output_layers": config["config"].get("output_layers"),
    }
    try:
        import tensorflow as tf  # noqa: F401

        model = tf.keras.models.load_model(model_path, compile=False)
        weights = {layer.name: [np.asarray(w, np.float32) for w in layer.get_weights()] for layer in model.layers if layer.get_weights()}
        return arch, weights, "tensorflow"
    except ImportError:
        pass
    try:
        flat = _read_h5_with_h5py(raw)
        method = "h5py"
    except ImportError:
        flat = _read_h5_with_ctypes(raw)
        method = "libhdf5-ctypes"
    weights = map_h5_to_layers(arch, flat)
    return arch, weights, method


def export(model_path: Path, prefix: Path) -> dict:
    arch, weights, method = read_keras_archive(model_path)
    validate_shapes(arch, weights)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    flat = {f"{layer}/{i}": w for layer, tensors in weights.items() for i, w in enumerate(tensors)}
    np.savez_compressed(Path(str(prefix) + ".weights.npz"), **flat)
    Path(str(prefix) + ".arch.json").write_text(json.dumps(arch, indent=1) + "\n", encoding="utf-8")
    return {"method": method, "tensors": len(flat), "parameters": int(sum(w.size for w in flat.values()))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out-prefix", type=Path, default=None,
                        help="Output prefix (default: next to the model, same stem).")
    args = parser.parse_args()
    prefix = args.out_prefix or args.model.with_suffix("")
    try:
        info = export(args.model, prefix)
    except (OSError, KeyError, RuntimeError, ValueError) as exc:
        print(f"Export failed: {exc}", file=sys.stderr)
        return 1
    print(f"Exported {info['tensors']} tensors ({info['parameters']:,} parameters) via {info['method']} -> {prefix}.weights.npz / .arch.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
