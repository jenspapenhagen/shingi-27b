"""Portable CUDA and Apple Metal memory gates. This module never manages system services."""
import os
import platform
import re
import subprocess

UNREPORTED = re.compile(r"\[?(N/A|Not Supported)\]?")
VM_STAT_PAGE_SIZE = re.compile(r"page size of (\d+) bytes")
# vm_stat counters counted as available: pages the system can hand to a new allocation without
# swapping. Active and wired pages are excluded. Purgeable pages can overlap the other queues,
# so this is an optimistic estimate; the floors below leave room for that.
VM_STAT_AVAILABLE = ("Pages free", "Pages inactive", "Pages purgeable", "Pages speculative")


def is_macos():
    return platform.system() == "Darwin"


def darwin_memory_mib():
    """Total unified memory (sysctl hw.memsize) and available memory (vm_stat) on macOS."""
    try:
        total = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True, timeout=10))
        report = subprocess.check_output(["vm_stat"], text=True, timeout=10)
        page_size = int(VM_STAT_PAGE_SIZE.search(report).group(1))
        pages = {}
        for line in report.splitlines()[1:]:
            key, _, value = line.partition(":")
            pages[key.strip().strip('"')] = value.strip().rstrip(".")
        available = sum(int(pages[key]) for key in VM_STAT_AVAILABLE) * page_size
    except (OSError, subprocess.SubprocessError, AttributeError, KeyError, ValueError) as exc:
        raise RuntimeError("could not read unified memory from sysctl and vm_stat") from exc
    return total // (1024 * 1024), available // (1024 * 1024)


def metal_snapshot():
    try:
        name = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True, timeout=10).strip()
    except (OSError, subprocess.SubprocessError):
        name = ""
    total, free = darwin_memory_mib()
    return {"uuid": None, "name": name or "Apple Silicon", "total_mib": total, "free_mib": free,
            "memory_source": "vm_stat free + inactive + purgeable + speculative"}


def selected_gpu():
    uuid = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", uuid):
        raise RuntimeError("set CUDA_VISIBLE_DEVICES to exactly one full GPU UUID from nvidia-smi -L")
    return uuid


def system_memory_mib(path="/proc/meminfo"):
    """Total and available system memory, used for unified-memory GPUs such as the DGX Spark GB10."""
    values = {}
    try:
        with open(path) as stream:
            for line in stream:
                key, rest = line.split(":", 1)
                values[key] = int(rest.split()[0]) // 1024
        return values["MemTotal"], values["MemAvailable"]
    except (OSError, ValueError, IndexError, KeyError) as exc:
        raise RuntimeError("nvidia-smi does not report GPU memory and /proc/meminfo is unreadable") from exc


def gpu_snapshot():
    if is_macos():
        return metal_snapshot()
    uuid = selected_gpu()
    try:
        row = subprocess.check_output(
            ["nvidia-smi", "--id=" + uuid, "--query-gpu=uuid,name,memory.total,memory.free",
             "--format=csv,noheader,nounits"], text=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("could not query the selected CUDA GPU") from exc
    fields = [s.strip() for s in row.strip().split(",")]
    if len(fields) != 4 or fields[0] != uuid:
        raise RuntimeError("nvidia-smi returned an unexpected GPU identity")
    if UNREPORTED.fullmatch(fields[2]) or UNREPORTED.fullmatch(fields[3]):
        # Unified-memory GPUs report [N/A]; the GPU allocates from system memory.
        total, free = system_memory_mib()
        return {"uuid": fields[0], "name": fields[1], "total_mib": total, "free_mib": free,
                "memory_source": "/proc/meminfo MemAvailable"}
    try:
        return {"uuid": fields[0], "name": fields[1], "total_mib": int(fields[2]), "free_mib": int(fields[3]),
                "memory_source": "nvidia-smi"}
    except ValueError as exc:
        raise RuntimeError("nvidia-smi returned invalid memory values") from exc


def gpu_profile():
    """Return the GPU UUID (None on macOS) and the free-memory floors (MiB) before loading and during inference."""
    gpu = gpu_snapshot()
    print(f"GPU {gpu['name']} ({gpu['uuid'] or 'Metal, unified memory'}): {gpu['free_mib']} of {gpu['total_mib']} MiB free "
          f"according to {gpu['memory_source']}", flush=True)
    if gpu["uuid"] is None:
        # Apple Silicon: the model needs about 8 GB at the 16K context, and macOS itself needs
        # several GB. 16 GiB is the minimum; the target is 24 GB M-series Macs, where 12 GiB free
        # before loading leaves the model plus about 4 GiB, and 2 GiB must stay free while serving.
        if gpu["total_mib"] < 16 * 1024:
            raise RuntimeError("Shingi 27B requires a Mac with at least 16 GiB of unified memory")
        return None, 12 * 1024, 2 * 1024
    if gpu["total_mib"] < 20 * 1024:
        raise RuntimeError("Shingi 27B requires a GPU with at least 20 GiB usable VRAM")
    preload, headroom = (14, 4) if gpu["total_mib"] <= 32 * 1024 else (30, 10)
    return gpu["uuid"], preload * 1024, headroom * 1024


def gpu_free_mib():
    return gpu_snapshot()["free_mib"]
