"""
Install PyTorch and project dependencies
"""
import subprocess
import sys

output_log = []

def run_command(cmd, description):
    output_log.append(f"\n{'='*60}")
    output_log.append(f"{description}")
    output_log.append(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)
        output_log.append(result.stdout)
        if result.stderr:
            output_log.append(f"STDERR: {result.stderr}")
        output_log.append(f"Return code: {result.returncode}")
    except Exception as e:
        output_log.append(f"ERROR: {str(e)}")

# Install PyTorch
run_command("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
            "Installing PyTorch with CUDA 12.1")

# Install project dependencies (excluding torch packages already installed)
run_command("pip install numpy pandas matplotlib seaborn scikit-learn albumentations gradio Pillow tqdm jupyter ipykernel",
            "Installing project dependencies")

# Verify installations
run_command("pip list | findstr torch || pip list | grep torch",
            "Verifying PyTorch installation")

run_command("python -c \"import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')\"",
            "Testing PyTorch and CUDA")

# Register Jupyter kernel
run_command("python -m ipykernel install --user --name plant_disease_dl --display-name \"Python (plant_disease_dl)\"",
            "Registering Jupyter kernel")

# List Jupyter kernels
run_command("jupyter kernelspec list",
            "Listing Jupyter kernels")

# Write log to file
with open("setup_log.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(output_log))

print("\nSetup log written to setup_log.txt")
print("\n".join(output_log[-500:]))  # Print last 500 lines
