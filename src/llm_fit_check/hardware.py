from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass

import psutil


@dataclass
class Hardware:
    os: str
    cpu_cores: int
    ram_gb: float
    gpu_vendor: str | None
    gpu_name: str | None
    vram_gb: float
    unified_memory: bool

    @property
    def usable_memory_gb(self) -> float:
        # On unified-memory systems (Apple Silicon), the GPU draws from RAM.
        # Reserve ~25% for the OS and the inference runtime overhead.
        if self.unified_memory:
            return round(self.ram_gb * 0.75, 1)
        if self.vram_gb > 0:
            return self.vram_gb
        # CPU-only: leave 4GB headroom.
        return max(0.0, round(self.ram_gb - 4.0, 1))

    @property
    def runtime(self) -> str:
        if self.unified_memory:
            return "metal"
        if self.gpu_vendor == "nvidia":
            return "cuda"
        if self.gpu_vendor == "amd":
            return "rocm"
        return "cpu"


def _detect_nvidia() -> tuple[str | None, float]:
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode()
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        vram = mem.total / (1024**3)
        pynvml.nvmlShutdown()
        return name, round(vram, 1)
    except Exception:
        pass

    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode()
        first = out.strip().splitlines()[0]
        name, mem_mib = [p.strip() for p in first.split(",")]
        return name, round(float(mem_mib) / 1024, 1)
    except Exception:
        return None, 0.0


def _detect_apple_silicon() -> bool:
    if platform.system() != "Darwin":
        return False
    return platform.machine() in ("arm64", "aarch64")


def detect() -> Hardware:
    ram_gb = round(psutil.virtual_memory().total / (1024**3), 1)
    cpu_cores = psutil.cpu_count(logical=False) or psutil.cpu_count() or 1
    os_name = platform.system()

    unified = _detect_apple_silicon()
    gpu_vendor: str | None = None
    gpu_name: str | None = None
    vram_gb = 0.0

    if unified:
        gpu_vendor = "apple"
        gpu_name = "Apple Silicon (unified memory)"
    else:
        name, vram = _detect_nvidia()
        if name:
            gpu_vendor = "nvidia"
            gpu_name = name
            vram_gb = vram

    return Hardware(
        os=os_name,
        cpu_cores=cpu_cores,
        ram_gb=ram_gb,
        gpu_vendor=gpu_vendor,
        gpu_name=gpu_name,
        vram_gb=vram_gb,
        unified_memory=unified,
    )
