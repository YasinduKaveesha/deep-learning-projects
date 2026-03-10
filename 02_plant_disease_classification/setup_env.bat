@echo off
REM Setup CUDA-enabled environment for plant disease classification

echo ======================================
echo Setting up plant_disease_dl environment
echo ======================================

REM Activate the environment
call conda activate plant_disease_dl

echo.
echo Installing PyTorch with CUDA...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo.
echo Installing project dependencies...
pip install -r requirements.txt

echo.
echo Installing JupyterLab...
pip install jupyterlab ipykernel

echo.
echo Registering Jupyter kernel...
python -m ipykernel install --user --name plant_disease_dl --display-name "Python (plant_disease_dl)"

echo.
echo Testing PyTorch and CUDA...
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"

echo.
echo Environment setup complete!
echo.
echo Conda environments:
conda env list

echo.
echo Jupyter kernels:
jupyter kernelspec list

pause
