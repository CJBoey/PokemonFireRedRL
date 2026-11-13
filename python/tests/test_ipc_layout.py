"""The NumPy view of the shared memory matches the worker's C structs."""
import pytest

from fireredsim import _ipc
from fireredsim.obs import resolve_obs_dtype


def test_layout_matches_worker(worker_path):
    info = _ipc.worker_info(worker_path)
    _ipc.check_layout(info)  # raises LayoutError on any difference
    assert info["shm_version"] == _ipc.SHM_VERSION
    assert info["pixel_bytes"] == _ipc.PIXEL_BYTES
    assert info["snapshot_size"] == 72 + 8 * len(info["regions"]) + info["state_size"]
    # pixel formats: PnShmHeader.pixel_stride in the formerly reserved area, listed apart
    # from "structs" (an older Python's check rejects fields it does not know)
    assert info["struct_ext"]["PnShmHeader"]["pixel_stride"] == [136, 4]
    assert "pixel_stride" not in info["structs"]["PnShmHeader"]["fields"]
    assert _ipc.HEADER_DTYPE.fields["pixel_stride"][1] == 136
    assert sorted(info["pixel_formats"]) == sorted(f.code for f in _ipc.PIXEL_FORMATS.values())


def test_layout_ext_mismatch_is_detected(worker_path):
    info = dict(_ipc.worker_info(worker_path))
    info["struct_ext"] = {"PnShmHeader": {"pixel_stride": [140, 4]}}
    with pytest.raises(_ipc.LayoutError):
        _ipc.check_layout(info)
    info["struct_ext"] = {"PnShmHeader": {"a_future_field": [144, 4]}}  # unknown extensions are ignored
    _ipc.check_layout(info)


def test_layout_mismatch_is_detected(worker_path):
    info = dict(_ipc.worker_info(worker_path))
    structs = {k: dict(v) for k, v in info["structs"].items()}
    fields = dict(structs["PnEnvCmd"]["fields"])
    fields["slot"] = [40, 4]
    structs["PnEnvCmd"] = dict(structs["PnEnvCmd"], fields=fields)
    info["structs"] = structs
    with pytest.raises(_ipc.LayoutError):
        _ipc.check_layout(info)


def test_observation_dtype_size(worker_path):
    info = _ipc.worker_info(worker_path)
    dt, kind = resolve_obs_dtype(info)
    assert dt.itemsize == info["obs_size"]
    assert kind in ("obs", "stub")


def test_shm_layout_is_page_aligned():
    L = _ipc.ShmLayout(num_envs=37, num_workers=5, obs_size=4256, pixels=True, snapshot_slots=3,
                       snapshot_size=564360)
    for off in (L.off_workers, L.off_cmds, L.off_results, L.off_info, L.off_obs, L.off_pixels, L.off_snapshots):
        assert off % _ipc.PAGE == 0
    assert L.off_pixels >= L.off_obs + 37 * 4256
    assert L.off_snapshots >= L.off_pixels + 37 * _ipc.PIXEL_BYTES
    assert L.total_size >= L.off_snapshots + 3 * L.snapshot_stride


def test_shm_layout_pixel_stride():
    common = dict(num_envs=37, num_workers=5, obs_size=4256, snapshot_slots=3, snapshot_size=564360)
    rgb = _ipc.ShmLayout(pixels=True, **common)
    assert rgb.pixel_stride == _ipc.PIXEL_BYTES  # 0 = one RGB frame (what an older Python's layout implies)
    g = _ipc.ShmLayout(pixels=True, pixel_stride=2400, **common)
    assert g.pixel_stride == 2400 and g.off_pixels == rgb.off_pixels
    assert g.off_snapshots >= g.off_pixels + 37 * 2400 and g.off_snapshots % _ipc.PAGE == 0
    assert g.off_snapshots < rgb.off_snapshots
    none = _ipc.ShmLayout(pixels=False, pixel_stride=2400, **common)
    assert none.pixel_stride == 0 and none.off_pixels == 0


def test_worker_set_pixel_views(worker_path):
    """The header carries the format code and the stride; the frame view is strided over the slots."""
    info = _ipc.worker_info(worker_path)
    ws = _ipc.WorkerSet(worker_path, info, 3, 1, pixels=True, pixel_code=0x13, pixel_stride=_ipc.PIXEL_BYTES)
    try:
        assert int(ws.header["pixels"]) == 0x13 and int(ws.header["pixel_stride"]) == _ipc.PIXEL_BYTES
        assert ws.pixel_slots.shape == (3, _ipc.PIXEL_BYTES)
        assert ws.pixels.shape == (3, 54, 80) and ws.pixels.strides == (_ipc.PIXEL_BYTES, 80, 1)
        ws.pixel_slots[1, :] = 0
        ws.pixel_slots[1, 80 * 2 + 5] = 7
        assert ws.pixels[1, 2, 5] == 7
    finally:
        ws.close()
    with pytest.raises(ValueError):
        _ipc.WorkerSet(worker_path, info, 1, 1, pixels=True, pixel_code=0x14, pixel_stride=100)
    with pytest.raises(ValueError):
        _ipc.WorkerSet(worker_path, info, 1, 1, pixels=True, pixel_code=0x15)
