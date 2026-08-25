# Reproduction

## Validated environment

- Python 3.12.10
- PyTorch 2.12.1 + CUDA 13.0
- NVIDIA GeForce RTX 5060 Laptop GPU
- Seed 42 and fixed `cuda:0`

## Setup

```powershell
git clone https://github.com/PengjinHao-cell/MiNi-DeepID.git
Set-Location MiNi-DeepID
python -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install --upgrade pip
& '.\.venv\Scripts\python.exe' -m pip install -r requirements.txt
```

## Gated workflow

```powershell
& '.\.venv\Scripts\python.exe' verify_environment.py
& '.\.venv\Scripts\python.exe' prepare_lfw.py
& '.\.venv\Scripts\python.exe' verify_data.py
& '.\.venv\Scripts\python.exe' -m pytest -q
& '.\.venv\Scripts\python.exe' tiny_overfit.py
& '.\.venv\Scripts\python.exe' train.py --epochs 2
```

Open `run_in_pycharm.py` with the project interpreter for formal training. Only after freezing the best checkpoint should `evaluate.py` be invoked. It creates a receipt and refuses a second final evaluation. PCA is fitted on train embeddings and only transforms saved test embeddings.
