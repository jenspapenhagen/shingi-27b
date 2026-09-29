import functools

import pytest
from shingi import gpu
from shingi.backend import NativeReadout

UUID = 'GPU-01234567-89ab-cdef-0123-456789abcdef'


@pytest.mark.parametrize('value', ['', '0', '0,1', 'GPU-short', UUID + ',' + UUID])
def test_requires_one_unambiguous_gpu(value, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', value)
    with pytest.raises(RuntimeError, match='exactly one full GPU UUID'):
        gpu.selected_gpu()


@pytest.mark.parametrize('total,expected', [(24564, (14*1024, 4*1024)), (97887, (30*1024, 10*1024))])
def test_profile_uses_capacity_without_machine_whitelist(total, expected, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', UUID)
    monkeypatch.setattr(gpu.subprocess, 'check_output', lambda *a, **kw: f'{UUID}, Test GPU, {total}, 18000\n')
    assert gpu.gpu_profile() == (UUID, *expected)
    assert gpu.gpu_free_mib() == 18000
    assert gpu.gpu_snapshot()['memory_source'] == 'nvidia-smi'


def test_rejects_wrong_device_response(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', UUID)
    monkeypatch.setattr(gpu.subprocess, 'check_output', lambda *a, **kw: 'GPU-other, Test GPU, 23028, 18000\n')
    with pytest.raises(RuntimeError, match='unexpected GPU identity'):
        gpu.gpu_snapshot()


def test_rejects_garbage_memory_values(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', UUID)
    monkeypatch.setattr(gpu.subprocess, 'check_output', lambda *a, **kw: f'{UUID}, Test GPU, lots, 18000\n')
    with pytest.raises(RuntimeError, match='invalid memory values'):
        gpu.gpu_snapshot()


MEMINFO = 'MemTotal:       128000000 kB\nMemFree:         9000000 kB\nMemAvailable:   100000000 kB\n'


@pytest.mark.parametrize('reported', ['[N/A], [N/A]', 'N/A, N/A', '[Not Supported], [Not Supported]'])
def test_unified_memory_falls_back_to_meminfo(reported, monkeypatch, tmp_path, capsys):
    # DGX Spark (GB10) reports [N/A] for memory; the GPU allocates from system memory.
    meminfo = tmp_path / 'meminfo'
    meminfo.write_text(MEMINFO)
    monkeypatch.setattr(gpu, 'system_memory_mib', functools.partial(gpu.system_memory_mib, meminfo))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', UUID)
    monkeypatch.setattr(gpu.subprocess, 'check_output', lambda *a, **kw: f'{UUID}, NVIDIA GB10, {reported}\n')
    snapshot = gpu.gpu_snapshot()
    assert snapshot['total_mib'] == 128000000 // 1024 and snapshot['free_mib'] == 100000000 // 1024
    assert snapshot['memory_source'] == '/proc/meminfo MemAvailable'
    assert gpu.gpu_profile() == (UUID, 30 * 1024, 10 * 1024)
    assert '/proc/meminfo' in capsys.readouterr().out


def test_unreadable_meminfo_is_a_clear_error(tmp_path):
    with pytest.raises(RuntimeError, match='does not report GPU memory'):
        gpu.system_memory_mib(tmp_path / 'missing')
    (tmp_path / 'partial').write_text('MemTotal: 1 kB\n')
    with pytest.raises(RuntimeError, match='does not report GPU memory'):
        gpu.system_memory_mib(tmp_path / 'partial')


def test_preload_gate_refuses_before_starting_the_runtime(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', UUID)
    monkeypatch.setattr(gpu.subprocess, 'check_output', lambda *a, **kw: f'{UUID}, Test GPU, 24564, 8000\n')
    started = []
    monkeypatch.setattr('shingi.backend.subprocess.Popen', lambda *a, **kw: started.append(a))
    with pytest.raises(RuntimeError, match='14336 MiB free'):
        NativeReadout('unused', 'unused')
    assert started == []


def test_driver_failure_is_backend_unavailability(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', UUID)
    def fail(*args, **kwargs):
        raise FileNotFoundError('nvidia-smi')
    monkeypatch.setattr(gpu.subprocess, 'check_output', fail)
    with pytest.raises(RuntimeError, match='could not query'):
        gpu.gpu_snapshot()
