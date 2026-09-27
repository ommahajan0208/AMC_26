"""
diagnose_cuda.py
================
Run this inside your notebook or terminal to identify exactly why
CUDA is not being detected by PyTorch.

Usage (in notebook cell):
    import subprocess, sys
    subprocess.run([sys.executable, "diagnose_cuda.py"])

Or from Anaconda Prompt with your env activated:
    python diagnose_cuda.py
"""

import sys
import subprocess

print("=" * 60)
print("CUDA / PyTorch Diagnostics")
print("=" * 60)

# 1. Python location
print(f"\n[Python] {sys.executable}")
print(f"[Python] version: {sys.version.split()[0]}")

# 2. Torch version and CUDA
try:
    import torch
    print(f"\n[PyTorch] version          : {torch.__version__}")
    print(f"[PyTorch] CUDA compiled for: {torch.version.cuda}")
    print(f"[PyTorch] cuda.is_available: {torch.cuda.is_available()}")
    print(f"[PyTorch] device count     : {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"[GPU {i}]  {p.name}  |  VRAM: {p.total_memory/1e9:.1f} GB")
    else:
        print("\n[DIAGNOSIS] CUDA not available. Common reasons:")
        # Check if CPU-only build
        if torch.version.cuda is None:
            print("  CAUSE: PyTorch is a CPU-only build (torch.version.cuda is None).")
            print("  FIX  : Reinstall PyTorch with the correct CUDA build.")
        else:
            print(f"  PyTorch was compiled for CUDA {torch.version.cuda}")
            print("  Possible causes:")
            print("    1. NVIDIA driver is outdated (driver must support CUDA",
                  f"{torch.version.cuda} or newer)")
            print("    2. The GPU driver is installed but not visible (reboot needed?)")
            print("    3. You are running in a sandboxed terminal without GPU access")
except ImportError:
    print("[ERROR] PyTorch is not installed in this environment.")

# 3. NVIDIA SMI (checks driver independently of PyTorch)
print("\n[NVIDIA-SMI] Checking GPU driver ...")
try:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
         "--format=csv,noheader"],
        capture_output=True, text=True, timeout=10
    )
    if result.returncode == 0:
        for line in result.stdout.strip().splitlines():
            print(f"  GPU: {line.strip()}")
        print("  -> nvidia-smi sees your GPU. Driver is installed.")
    else:
        print("  nvidia-smi failed (returncode != 0). Driver may not be installed.")
        print(f"  stderr: {result.stderr.strip()[:200]}")
except FileNotFoundError:
    print("  nvidia-smi not found in PATH.")
    print("  This means the NVIDIA display driver is either not installed")
    print("  or its bin dir is not in your PATH.")
except Exception as e:
    print(f"  Error running nvidia-smi: {e}")

# 4. Detect the mismatch: right GPU, wrong PyTorch build
print("\n[SUMMARY]")
try:
    import torch
    cuda_in_torch = torch.version.cuda
    smi_ok = subprocess.run(["nvidia-smi"], capture_output=True, timeout=5).returncode == 0

    if smi_ok and not torch.cuda.is_available():
        if cuda_in_torch is None:
            print("  GPU detected by nvidia-smi but PyTorch is CPU-only.")
            print("  -> You need to reinstall PyTorch with CUDA support.")
            print()
            print("  Run this in your Anaconda Prompt (with isro env activated):")
            print()
            print("    pip uninstall torch torchvision torchaudio -y")
            print()
            print("  Then install the CUDA 12.1 build (matches RTX 4050):")
            print()
            print("    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        else:
            print(f"  GPU detected by nvidia-smi, PyTorch compiled for CUDA {cuda_in_torch},")
            print("  but torch.cuda.is_available() is False.")
            print("  -> Your NVIDIA driver may be too old for this CUDA version.")
            print("  -> Or try rebooting if you recently installed the driver.")
    elif not smi_ok and not torch.cuda.is_available():
        print("  nvidia-smi also fails -> GPU driver not properly installed.")
        print("  -> Install NVIDIA Game Ready / Studio Driver from https://www.nvidia.com/Download/")
    elif torch.cuda.is_available():
        print("  Everything looks good - CUDA is available!")
except Exception:
    pass
