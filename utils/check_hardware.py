import torch
import psutil
import platform
import shutil
import subprocess

def check_hardware(verbose=True):
    """
    Check if the hardware supports CUDA and if a GPU is available.
    Returns:
        - device: The device to be used for computation (CPU or GPU).
        - is_cuda: Boolean indicating if CUDA is available.
    """
    if verbose:
        # CPU
        print("================================================================")
        print("🧠 CPU Info:")
        print(f"  - Processor: {platform.processor()}")
        print(f"  - Physical cores: {psutil.cpu_count(logical=False)}")
        print(f"  - Logical cores: {psutil.cpu_count(logical=True)}\n")

        # RAM
        ram = psutil.virtual_memory()
        print("💾 RAM Info:")
        print(f"  - Total RAM: {ram.total / (1024**3):.2f} GB\n")

        # Disk
        total, used, free = shutil.disk_usage("/")
        print("💽 Disk Info:")
        print(f"  - Total Disk: {total / (1024**3):.2f} GB")
        print(f"  - Free Space: {free / (1024**3):.2f} GB\n")

        # GPU - PyTorch
        print("🖥️ GPU (via PyTorch):")
        if torch.cuda.is_available():
            print(f"  - CUDA available: ✅ Yes")
            print(f"  - GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"    - GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("  - CUDA available: ❌ No")
        print()

        # CUDA Toolkit
        print("⚙️ CUDA Toolkit:")
        try:
            result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                print(result.stdout.strip())
            else:
                print("  - nvcc not found (CUDA Toolkit may not be installed)")
        except FileNotFoundError:
            print("  - nvcc not found (CUDA Toolkit may not be installed)")
        print()

        # cuDNN
        print("📦 cuDNN Info:")
        try:
            cudnn_version = torch.backends.cudnn.version()
            print(f"  - cuDNN version: {cudnn_version}")
        except Exception:
            print("  - cuDNN not found or not available")

        # Python & Library versions
        print("\n📦 Python & Library Versions:")
        print(f"  - Python: {platform.python_version()}")
        print(f"  - PyTorch: {torch.__version__}")
        print()

        print("✅ Done")
        print("================================================================")


    is_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_cuda else "cpu")
    return device, is_cuda

if __name__ == "__main__":
    check_hardware()