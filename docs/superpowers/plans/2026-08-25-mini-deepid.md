# Mini-DeepID LFW Reproduction Implementation Plan

> **For agentic workers:** Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible Windows/CUDA teaching implementation that selects the 10 most frequent LFW identities, uses exactly 50 grayscale images per identity, trains a four-convolution Mini-DeepID network, and exports metrics plus a 160D embedding visualization.

**Architecture:** One immutable CSV manifest is the boundary between LFW acquisition and all later stages. A compact PyTorch model returns both a 160D embedding and 10-class logits; separate command-line entry points verify the environment, prepare data, train, evaluate, visualize, and predict while sharing one configuration module.

**Tech Stack:** Windows PowerShell, Python 3.12, PyTorch 2.12.1 with CUDA 13.0, torchvision 0.27.1, scikit-learn, NumPy, pandas, Pillow, Matplotlib, Seaborn, tqdm, pytest.

## Global Constraints

- Use `C:\Users\pengjian\AppData\Local\Programs\Python\Python312\python.exe` only to create `E:\Self Experiment\MINI DEEP ID\.venv`.
- Pin `torch==2.12.1` and `torchvision==0.27.1` from `https://download.pytorch.org/whl/cu130`.
- Run training on `cuda:0`; abort if CUDA execution fails and never fall back silently to CPU.
- Use LFW funneled grayscale images with `min_faces_per_person=20` and loader resize `0.5`.
- Select the 10 most frequent identities and exactly 50 samples per identity with random seed `42`.
- Split each identity into 35 train, 7 validation, and 8 test samples.
- Freeze the split in `data/manifests/split_manifest.csv`; every later stage must read it and must never call `random_split`.
- Apply random augmentation only after splitting and only to manifest rows whose split is `train`.
- Resize model inputs to `1×64×64`; return a 160D embedding and 10 logits.
- Use AdamW, learning rate `1e-3`, weight decay `1e-4`, batch size `32`, maximum 80 epochs, and early-stopping patience `12`.
- Before smoke or formal training, overfit 32 deterministic train-manifest images with augmentation, dropout, and weight decay disabled; require at least 95% train accuracy and target 100% within 300 epochs.
- Do not use the test split for model selection or hyperparameter changes.
- Run model inference over the frozen test split exactly once, write `final_test_receipt.json`, and refuse a second final evaluation under the same protocol.
- Record G0–G14 attempts, status, timestamp, metrics, and evidence paths in append-only `outputs/gate_status.json`; a command must refuse to start when its prerequisite gate has not passed.
- Keep this a closed-set teaching experiment; do not implement `UNKNOWN`, verification, private-photo training, a GUI, or production authentication.

---

## File Map

- `README.md`: complete setup, execution, interpretation, and limitations guide.
- `.gitignore`: excludes environments, downloaded data, checkpoints, and generated outputs while retaining manifests only when intentionally added.
- `requirements.txt`: direct runtime and test dependencies.
- `requirements-lock.txt`: fully resolved versions from the verified environment.
- `environment.txt`: human-readable Python, package, CUDA, GPU, driver, capability, and seed record.
- `config.py`: immutable paths and experiment hyperparameters.
- `dataset.py`: top-identity selection, deterministic manifest construction, image export, transforms, and dataset loading.
- `model.py`: four convolution blocks, multi-scale fusion, 160D DeepID layer, and classifier.
- `training.py`: reproducibility, one-epoch execution, evaluation loop, early stopping, atomic checkpoints, and history output.
- `gate_status.py`: validated append-only G0–G14 evidence ledger and prerequisite checks.
- `verify_environment.py`: JSON environment report plus real CUDA forward/backward smoke test.
- `prepare_lfw.py`: LFW fetch, balanced subset export, manifest validation, and data summary.
- `train.py`: full training command.
- `tiny_overfit.py`: mandatory 32-image memorization sanity check and evidence report.
- `run_in_pycharm.py`: no-argument, green-button launcher for the user-owned formal training run.
- `docs/PYCHARM_RUN.md`: exact interpreter selection and visual run instructions.
- `evaluate.py`: one-shot test metrics and required evaluation figures.
- `visualize_embeddings.py`: fit PCA on train embeddings and transform saved test embeddings without a second test-loader pass.
- `predict.py`: closed-set prediction for one processed LFW image.
- `tests/test_config.py`: stable path and hyperparameter contract.
- `tests/test_dataset.py`: selection, split, validation, transform, and dataset behavior.
- `tests/test_model.py`: output shapes, gradients, and deterministic reload.
- `tests/test_training.py`: parameter updates, finite metrics, and checkpoint round trip.

---

## Task 1: Establish the Project Boundary and Archive Primary References

**Files:**

- Create: `.gitignore`
- Create: `requirements.txt`
- Create: `docs/references/README.md`
- Download during execution: `docs/references/Sun_Deep_Learning_Face_2014_CVPR_paper.pdf`

- [ ] **Step 1: Add the ignore rules**

Write `.gitignore` with these exact entries:

```gitignore
.venv/
__pycache__/
.pytest_cache/
*.py[cod]
data/cache/
data/processed/
checkpoints/
outputs/
```

- [ ] **Step 2: Declare direct dependencies**

Write `requirements.txt`:

```text
--index-url https://download.pytorch.org/whl/cu130
--extra-index-url https://pypi.org/simple
torch==2.12.1
torchvision==0.27.1
numpy
pandas
Pillow
scikit-learn
matplotlib
seaborn
tqdm
pytest
```

- [ ] **Step 3: Record the primary references**

Write `docs/references/README.md` with the paper title, authors, CVPR 2014 citation, official HTML/PDF links, scikit-learn LFW loader link, and a section explaining that the project retains identity classification, a compact 160D hidden feature, and multi-scale fusion while omitting the original 10,000-class scale, multiple patches, multiple ConvNets, Joint Bayesian, and verification protocol.

Create `README.md` with this exact first line and keep it unchanged through later tasks:

```markdown
> **Mini-DeepID educational reproduction inspired by CVPR 2014 DeepID, not a reproduction of the paper's reported LFW 97.45% result.**
```

- [ ] **Step 4: Download the official PDF**

Run in PowerShell:

```powershell
Invoke-WebRequest -Uri 'https://openaccess.thecvf.com/content_cvpr_2014/papers/Sun_Deep_Learning_Face_2014_CVPR_paper.pdf' -OutFile 'docs\references\Sun_Deep_Learning_Face_2014_CVPR_paper.pdf'
Get-Item 'docs\references\Sun_Deep_Learning_Face_2014_CVPR_paper.pdf' | Select-Object Name,Length
```

Expected: the file exists and has non-zero length. Open or render the first page during verification and confirm the title and authors.

- [ ] **Step 5: Commit the boundary**

```powershell
git add .gitignore README.md requirements.txt docs/references
git commit -m "docs: archive DeepID references and project boundary"
```

---

## Task 2: Create and Prove the Dedicated CUDA Environment

**Files:**

- Create at runtime: `.venv/`
- Create at runtime: `requirements-lock.txt`
- Create at runtime: `environment.txt`
- Create at runtime: `outputs/environment_report.json`
- Create later in this task: `verify_environment.py`

- [ ] **Step 1: Create the environment from the verified base interpreter**

```powershell
& 'C:\Users\pengjian\AppData\Local\Programs\Python\Python312\python.exe' -m venv '.venv'
& '.\.venv\Scripts\python.exe' --version
```

Expected: Python 3.12.x and `sys.prefix` points inside `E:\Self Experiment\MINI DEEP ID\.venv`.

- [ ] **Step 2: Install the locked CUDA 13.0 build**

Run:

```powershell
& '.\.venv\Scripts\python.exe' -m pip install --upgrade pip
& '.\.venv\Scripts\python.exe' -m pip install -r requirements.txt
& '.\.venv\Scripts\python.exe' -m pip check
```

Expected: `pip check` prints `No broken requirements found.` and importing Torch reports version 2.12.1 with CUDA runtime 13.0.

- [ ] **Step 3: Write the environment verifier**

Implement `verify_environment.py` so `main()`:

1. imports Python/platform, Torch, torchvision, NumPy, pandas, scikit-learn, Pillow, Matplotlib, and Seaborn versions;
2. raises `RuntimeError` when `torch.cuda.is_available()` is false;
3. asserts `torch.cuda.get_device_name(0)` contains `RTX 5060`;
4. creates a random `2000×2000` tensor on `cuda:0`, multiplies it by itself, sums the result, and calls `backward()`;
5. writes Python version, random seed, package versions, device name, driver, capability, CUDA runtime, and tensor result finiteness to both `environment.txt` and `outputs/environment_report.json`;
6. prints `MINI_DEEPID_CUDA_TENSOR_OK device=cuda:0` only after all tensor checks pass.

- [ ] **Step 4: Run the real CUDA check**

```powershell
& '.\.venv\Scripts\python.exe' verify_environment.py
```

Expected final line at this stage: `MINI_DEEPID_CUDA_TENSOR_OK device=cuda:0`.

- [ ] **Step 5: Lock the resolved environment**

```powershell
& '.\.venv\Scripts\python.exe' -m pip freeze | Set-Content -Encoding utf8 'requirements-lock.txt'
& '.\.venv\Scripts\python.exe' -m pip check
```

Expected: the lock file contains exact `torch`, `torchvision`, `scikit-learn`, and plotting-library versions.

- [ ] **Step 6: Commit the verifier**

```powershell
git add verify_environment.py requirements-lock.txt environment.txt
git commit -m "feat: add strict CUDA environment verification"
```

---

## Task 3: Lock Configuration and the Gate Evidence Ledger

**Files:**

- Create: `config.py`
- Create: `gate_status.py`
- Create: `tests/test_config.py`
- Create: `tests/test_gate_status.py`

- [ ] **Step 1: Write the failing configuration test**

The test must assert these exact defaults:

```python
from config import CONFIG


def test_default_config_contract():
    assert CONFIG.seed == 42
    assert CONFIG.image_size == 64
    assert CONFIG.num_classes == 10
    assert CONFIG.samples_per_class == 50
    assert (CONFIG.train_per_class, CONFIG.val_per_class, CONFIG.test_per_class) == (35, 7, 8)
    assert CONFIG.embedding_dim == 160
    assert CONFIG.batch_size == 32
    assert CONFIG.max_epochs == 80
    assert CONFIG.early_stopping_patience == 12
    assert CONFIG.learning_rate == 1e-3
    assert CONFIG.weight_decay == 1e-4
    assert CONFIG.device == "cuda:0"
```

- [ ] **Step 2: Confirm the test fails**

```powershell
& '.\.venv\Scripts\python.exe' -m pytest tests/test_config.py -q
```

Expected: failure because `config.py` does not exist.

- [ ] **Step 3: Implement the immutable configuration**

Create a frozen `ExperimentConfig` dataclass. Derive `project_root` from `Path(__file__).resolve().parent`; derive cache, processed, manifest, checkpoint, and output paths from that root. Include all asserted fields plus `min_faces_per_person=20`, `lfw_resize=0.5`, `dropout=0.4`, `horizontal_flip_probability=0.5`, `rotation_degrees=8`, `translation_fraction=0.05`, and `scale_range=(0.95, 1.05)`. Export exactly one `CONFIG = ExperimentConfig()`.

- [ ] **Step 4: Write the failing gate-ledger tests**

Test that `record_gate("G0", "passed", metrics, evidence_paths)` appends a timestamped attempt to a temporary JSON file, preserves an earlier failed attempt, rejects gate identifiers outside G0–G14, and that `require_gate_passed("G0")` raises before a pass but succeeds after one. The JSON file must remain parseable after every write.

- [ ] **Step 5: Implement the gate ledger**

`gate_status.py` must write through a sibling temporary file and `Path.replace`, preserve all attempts, validate status as `passed` or `failed`, store metrics and project-relative evidence paths, and expose `record_gate` plus `require_gate_passed`. Never delete or rewrite an earlier attempt.

- [ ] **Step 6: Run the tests and commit**

```powershell
& '.\.venv\Scripts\python.exe' -m pytest tests/test_config.py tests/test_gate_status.py -q
git add config.py gate_status.py tests/test_config.py tests/test_gate_status.py
git commit -m "feat: lock experiment configuration and gate evidence"
```

Expected: all configuration and append-only gate-ledger tests pass.

---

## Task 4: Build a Deterministic Balanced LFW Manifest

**Files:**

- Create: `dataset.py`
- Create: `tests/test_dataset.py`
- Create: `prepare_lfw.py`

- [ ] **Step 1: Write failing pure-function tests**

Use synthetic labels so tests do not download LFW. Test these public functions:

```python
select_top_labels(targets: np.ndarray, num_classes: int, samples_per_class: int) -> list[int]
sample_and_split_indices(targets: np.ndarray, selected_labels: list[int], seed: int) -> list[dict]
validate_manifest(frame: pd.DataFrame, num_classes: int) -> None
```

The synthetic counts must include labels with 70, 65, 60, 58, 57, 56, 55, 54, 53, 52, and 51 samples. Assert that only the ten largest are selected, each selected label contributes exactly 50 rows, repeated calls with seed 42 are identical, each label has 35/7/8 rows, and no `source_index` occurs twice.

- [ ] **Step 2: Confirm failures**

```powershell
& '.\.venv\Scripts\python.exe' -m pytest tests/test_dataset.py -q
```

Expected: import or missing-function failure.

- [ ] **Step 3: Implement selection and splitting**

In `dataset.py`:

- Count targets with `collections.Counter`.
- Sort by descending count and then ascending numeric label to make ties deterministic.
- Reject a candidate with fewer than 50 images.
- Use one `np.random.default_rng(seed)` and shuffle each identity's source indices.
- Take the first 50 shuffled indices; assign the first 35 to train, next 7 to val, final 8 to test.
- Remap selected source labels to contiguous model labels 0–9 in selected-rank order.
- Return rows containing `source_index`, `source_label`, `model_label`, and `split`.
- In `validate_manifest`, raise `ValueError` for duplicate source indices, non-contiguous model labels, incorrect per-class counts, incorrect split counts, or an unexpected split name.

- [ ] **Step 4: Add image export and manifest construction**

Implement `prepare_lfw_dataset(config=CONFIG)`:

1. call `fetch_lfw_people` with the exact global constraints and project-local `data_home`;
2. select and split the identities;
3. clip each selected NumPy image to 0–255, convert it to unsigned 8-bit grayscale, and resize it to `64×64` with Pillow bilinear interpolation;
4. normalize only for model input later, not while exporting;
5. sanitize identity folder names to underscores;
6. export lossless grayscale PNGs to `data/processed/<identity>/<source_index>.png`;
7. add `identity_name` and project-relative `image_path` columns;
8. validate before atomically writing `data/manifests/split_manifest.csv` and `identities.json`;
9. write `outputs/data_summary.json` and a balanced class-distribution figure.

If the final manifest already exists and validates, return it without resampling unless the caller passes `--force`. `--force` may rebuild generated data but must not delete the scikit-learn download cache.

- [ ] **Step 5: Implement the CLI**

`prepare_lfw.py` accepts `--force`, calls the preparation function, and prints:

```text
LFW_PREPARE_OK classes=10 samples=500 train=350 val=70 test=80
```

- [ ] **Step 6: Run unit tests, then the real preparation**

```powershell
& '.\.venv\Scripts\python.exe' -m pytest tests/test_dataset.py -q
& '.\.venv\Scripts\python.exe' prepare_lfw.py
```

Expected: tests pass and the exact success marker appears. Manually inspect the class-distribution image and a ten-identity sample grid before accepting the task.

- [ ] **Step 7: Commit**

```powershell
git add dataset.py prepare_lfw.py tests/test_dataset.py data/manifests
git commit -m "feat: prepare deterministic balanced LFW subset"
```

---

## Task 5: Implement Dataset Transforms and the Mini-DeepID Model with TDD

**Files:**

- Modify: `dataset.py`
- Create: `model.py`
- Modify: `tests/test_dataset.py`
- Create: `tests/test_model.py`

- [ ] **Step 1: Add failing dataset transform tests**

Create a temporary grayscale `64×64` PNG and one-row manifest. Assert:

- train transform returns a finite `torch.float32` tensor shaped `[1,64,64]`;
- validation and test transforms are identical and deterministic;
- values after normalization lie within `[-1,1]` for valid 8-bit images;
- dataset returns `(image, model_label, source_index, image_path)`.

- [ ] **Step 2: Implement transforms and `LFWManifestDataset`**

Training transform order: Pillow grayscale, resize 64, random horizontal flip 0.5, random affine with 8 degrees/0.05 translation/0.95–1.05 scale, tensor conversion, normalize mean 0.5 and standard deviation 0.5. Evaluation transform: grayscale, resize, tensor, the same normalization. The dataset filters the manifest by split and never mutates the frame.

- [ ] **Step 3: Write failing model tests**

Tests must assert:

```python
model = MiniDeepID(num_classes=10, embedding_dim=160)
embedding, logits = model(torch.randn(4, 1, 64, 64))
assert embedding.shape == (4, 160)
assert logits.shape == (4, 10)
assert torch.isfinite(embedding).all()
assert torch.isfinite(logits).all()
```

Also compute cross entropy, call `backward()`, and assert every trainable layer receives at least one finite gradient. Save `state_dict`, load it into a new model in evaluation mode, and assert identical outputs for the same input.

- [ ] **Step 4: Implement `MiniDeepID`**

Each convolution uses kernel 3, stride 1, padding 1, followed by ReLU and 2×2 max pooling. Channels are 1→32→64→128→128. Adaptive-average-pool the third pooled output to 4×4, concatenate it with the fourth pooled output, flatten 4096 values, project to 160, apply ReLU and dropout 0.4, then classify 160→10. Do not apply Softmax in the model.

- [ ] **Step 5: Run tests and complete the environment verifier**

```powershell
& '.\.venv\Scripts\python.exe' -m pytest tests/test_dataset.py tests/test_model.py -q
& '.\.venv\Scripts\python.exe' verify_environment.py
```

Before running the command, extend `verify_environment.py` to instantiate `MiniDeepID(num_classes=10)`, run one `2×1×64×64` batch on `cuda:0`, compute cross entropy, call `backward()`, record `model_smoke=true`, and replace the tensor-only success line with `MINI_DEEPID_CUDA_SMOKE_OK device=cuda:0`. Expected: all tests pass and the final verifier marker appears after the model forward/backward pass.

- [ ] **Step 6: Commit**

```powershell
git add dataset.py model.py tests/test_dataset.py tests/test_model.py verify_environment.py
git commit -m "feat: implement 160D multi-scale Mini-DeepID model"
```

---

## Task 6: Implement Training, Tiny-Set Overfit, Atomic Checkpoints, and Smoke Test

**Files:**

- Create: `training.py`
- Create: `tests/test_training.py`
- Create: `tiny_overfit.py`
- Create: `train.py`

- [ ] **Step 1: Write failing training tests**

Using a fixed synthetic dataset of `32×1×64×64` tensors and balanced labels, test:

- `set_reproducible(42)` repeats Torch and NumPy random values;
- `run_epoch(..., optimizer=optimizer)` returns finite loss/accuracy and changes at least one parameter;
- `run_epoch(..., optimizer=None)` leaves all parameters unchanged;
- `save_checkpoint_atomic` followed by `load_checkpoint` restores model outputs and identity mapping;
- loading a checkpoint with a different identity mapping raises `ValueError`.

- [ ] **Step 2: Implement the training utilities**

`training.py` must provide:

- `set_reproducible(seed)` for Python, NumPy, CPU Torch, and all CUDA generators;
- `run_epoch(model, loader, criterion, device, optimizer=None)` with explicit train/eval modes and `torch.no_grad()` for evaluation;
- finite-loss checking before backward;
- accumulated sample-weighted loss and exact accuracy;
- `save_checkpoint_atomic(path, payload)` that writes a sibling `.tmp` file and uses `Path.replace` only after `torch.save` succeeds;
- `load_checkpoint(path, model, optimizer, expected_identities)` using `map_location` and strict state loading;
- JSON history writing with one record per epoch.

- [ ] **Step 3: Run the tests**

```powershell
& '.\.venv\Scripts\python.exe' -m pytest tests/test_training.py -q
```

Expected: all training tests pass.

- [ ] **Step 4: Implement and run the mandatory tiny-set overfit gate**

`tiny_overfit.py` must read only `train` rows from `split_manifest.csv`, select 32 images deterministically and as evenly across ten identities as possible, disable random augmentation, set model dropout to 0, use AdamW with learning rate `1e-3` and weight decay 0, and train for at most 300 epochs. Save `outputs/tiny_overfit.json` and `outputs/tiny_overfit_curve.png`. Exit non-zero below 95% training accuracy; print `MINI_DEEPID_TINY_OVERFIT_OK` at 95% or higher and report 100% as the target.

```powershell
& '.\.venv\Scripts\python.exe' tiny_overfit.py
```

Expected: the success marker appears. If it does not, stop this plan before smoke or formal training and diagnose labels, preprocessing, loss, gradients, optimizer steps, and learning rate.

- [ ] **Step 5: Implement the training CLI**

`train.py` must:

1. require a valid manifest and environment check;
2. create train/val datasets and seeded DataLoaders;
3. instantiate `MiniDeepID`, CrossEntropyLoss, and AdamW with the fixed settings;
4. accept `--epochs` only to reduce the maximum for smoke testing, never to exceed 80;
5. save the best validation-accuracy checkpoint to `checkpoints/mini_deepid_best.pth`;
6. stop after 12 epochs without improvement;
7. write history and loss/accuracy curves under a timestamped run directory;
8. print `MINI_DEEPID_TRAIN_OK` with best epoch and validation accuracy.

- [ ] **Step 6: Run the two-epoch real-data smoke test**

```powershell
& '.\.venv\Scripts\python.exe' train.py --epochs 2
```

Expected: CUDA is used, both epochs complete with finite loss, a checkpoint is saved, and the success marker appears.

- [ ] **Step 7: Commit**

```powershell
git add training.py tiny_overfit.py train.py tests/test_training.py
git commit -m "feat: add reproducible Mini-DeepID training pipeline"
```

---

## Task 7: Implement Final Evaluation and Required Figures

**Files:**

- Create: `evaluate.py`
- Create: `visualize_embeddings.py`
- Create or extend: `tests/test_training.py`

- [ ] **Step 1: Add failing metric and artifact tests**

Test a pure `collect_predictions` helper using a tiny fixed model/loader. Assert returned labels, predictions, probabilities, embeddings, source indices, and paths have matching sample counts. Test that metrics JSON contains accuracy, per-class precision/recall/F1, confusion matrix, checkpoint path, and identity mapping.

- [ ] **Step 2: Implement `evaluate.py`**

Load the immutable test split and best checkpoint. Refuse to start if `final_test_receipt.json` already exists. Run exactly one test inference pass and generate:

- `metrics.json` with `random_guess_accuracy=0.10`, Mini-DeepID accuracy, macro precision/recall/F1, per-class metrics, and confusion matrix;
- `confusion_matrix.png` with identity names on both axes;
- `predictions.png` with at least 12 test faces, truth, prediction, and Softmax confidence;
- `test_embeddings.npz` containing the embeddings and labels from that one pass;
- `final_test_receipt.json` containing the checkpoint SHA-256, manifest SHA-256, protocol identifier, timestamp, and final metrics;
- an explicit closed-set warning in console output and metrics metadata.

Do not expose any tuning controls in this command. Print `MINI_DEEPID_EVAL_OK samples=80` after all outputs exist.

- [ ] **Step 3: Implement `visualize_embeddings.py`**

Load the same checkpoint and collect embeddings only from the frozen training split. Fit `PCA(n_components=2, random_state=42)` on train embeddings, then call `transform` on `test_embeddings.npz` saved by the one final test pass. Save `embeddings_pca.png`, include explained-variance ratios and a legend for all ten identities, and print `MINI_DEEPID_PCA_OK fit=train transform=test dimensions=160->2 samples=80`. Never instantiate or iterate a test DataLoader in this script.

- [ ] **Step 4: Run focused and full tests**

```powershell
& '.\.venv\Scripts\python.exe' -m pytest tests/test_training.py -q
& '.\.venv\Scripts\python.exe' -m pytest -q
```

Expected: the full suite passes with no skipped tests that hide missing core behavior.

- [ ] **Step 5: Commit**

```powershell
git add evaluate.py visualize_embeddings.py tests/test_training.py
git commit -m "feat: add closed-set evaluation and embedding figures"
```

---

## Task 8: Implement Single-Image Closed-Set Prediction

**Files:**

- Create: `predict.py`
- Create: `tests/test_predict.py`

- [ ] **Step 1: Write a failing prediction test**

Use a temporary processed grayscale image and a fixed checkpoint. Assert prediction returns exactly these fields: `predicted_label`, `predicted_identity`, `confidence`, `embedding`, and `closed_set_warning`. Assert embedding length 160, confidence is between zero and one, and the warning states that unknown identities are unsupported.

- [ ] **Step 2: Implement prediction**

`predict.py` must require an `--image` option containing an existing PNG path, then use the evaluation transform, checkpoint identity mapping, `torch.softmax(logits, dim=1)`, and `torch.inference_mode()`. It prints the predicted identity and confidence, writes a JSON result beside the run outputs, and always prints:

```text
WARNING: closed-set result; unknown identities are not supported.
```

Reject missing files, non-image files, and checkpoint/identity-map mismatches with actionable errors.

- [ ] **Step 3: Run tests and one real validation-split prediction**

```powershell
& '.\.venv\Scripts\python.exe' -m pytest tests/test_predict.py -q
$miniDeepIdRelativeImage = (Import-Csv 'data\manifests\split_manifest.csv' | Where-Object { $_.split -eq 'val' } | Select-Object -First 1).image_path
$miniDeepIdTestImage = Join-Path (Get-Location) $miniDeepIdRelativeImage
& '.\.venv\Scripts\python.exe' predict.py --image $miniDeepIdTestImage
```

Expected: a valid identity, confidence, warning, and 160-value embedding metadata are produced.

- [ ] **Step 4: Commit**

```powershell
git add predict.py tests/test_predict.py
git commit -m "feat: add closed-set single-image prediction"
```

---

## Task 9: Prepare the PyCharm Handoff, Let the User Train, Then Evaluate Once

**Files:**

- Modify: `README.md`
- Create: `run_in_pycharm.py`
- Create: `docs/PYCHARM_RUN.md`
- Generate: `checkpoints/mini_deepid_best.pth`
- Generate: timestamped `outputs/<run_id>/` artifacts

- [ ] **Step 1: Run all preflight checks**

```powershell
& '.\.venv\Scripts\python.exe' -m pip check
& '.\.venv\Scripts\python.exe' -m pytest -q
& '.\.venv\Scripts\python.exe' verify_environment.py
& '.\.venv\Scripts\python.exe' prepare_lfw.py
```

Expected: no dependency conflict, all tests pass, strict CUDA marker appears, and the dataset reports exactly 10/500/350/70/80.

- [ ] **Step 2: Implement the no-argument PyCharm launcher**

`run_in_pycharm.py` must print `sys.executable`, require it to resolve to `E:\Self Experiment\MINI DEEP ID\.venv\Scripts\python.exe`, verify CUDA and the frozen manifest, require successful Tiny Overfit and Smoke Train evidence, confirm that no final-test receipt exists, print `PYCHARM_TRAIN_READY`, and then call the same formal training function used by `train.py`. Immediately before epoch 1 it prints `PYCHARM_FORMAL_TRAIN_START`. It trains and selects the best checkpoint but never imports or calls final evaluation code.

- [ ] **Step 3: Write the PyCharm guide and verify readiness without formal training**

`docs/PYCHARM_RUN.md` must instruct the user to open the project, select `E:\Self Experiment\MINI DEEP ID\.venv\Scripts\python.exe` as the Project Interpreter, open `run_in_pycharm.py`, and click the green Run triangle. It must show the expected first console lines and explain epoch progress, early stopping, and checkpoint location.

Add `--check-only` support to `run_in_pycharm.py`; it executes every preflight check, prints `PYCHARM_TRAIN_READY`, and exits before training.

```powershell
& '.\.venv\Scripts\python.exe' run_in_pycharm.py --check-only
```

Expected: `PYCHARM_TRAIN_READY` appears. Stop automated work here and tell the user the program is ready; do not run formal training on their behalf.

- [ ] **Step 4: Commit the verified PyCharm handoff before the user runs it**

```powershell
git add README.md run_in_pycharm.py docs/PYCHARM_RUN.md
git commit -m "feat: prepare verified PyCharm training handoff"
```

Expected: the exact launcher tested with `--check-only` is committed. Tell the user that the program is ready only after this commit succeeds.

- [ ] **Step 5: User-owned formal training action**

The user selects the project `.venv` interpreter in PyCharm, opens `run_in_pycharm.py`, and clicks Run. The console must show the exact interpreter path, RTX 5060, the gate summary, `PYCHARM_FORMAL_TRAIN_START`, and epoch metrics. Expected: training finishes by epoch 80 or early stopping with an intact `checkpoints/mini_deepid_best.pth`. Wait for the user to report completion before continuing.

- [ ] **Step 6: Freeze model selection and evaluate once after user confirmation**

```powershell
& '.\.venv\Scripts\python.exe' evaluate.py
& '.\.venv\Scripts\python.exe' visualize_embeddings.py
```

Expected: the final-test receipt and all required JSON/PNG artifacts exist. Record the real test accuracy; do not rerun after changing hyperparameters based on test observations.

- [ ] **Step 7: Visually inspect every figure**

Open the loss curve, accuracy curve, confusion matrix, predictions grid, and PCA plot. Confirm labels are legible, images are not clipped, identity colors are distinguishable, and figure titles state LFW closed-set evaluation.

- [ ] **Step 8: Complete `README.md`**

Document prerequisites, exact interpreter path, environment creation, install commands, the fixed dataset rule, all execution commands in order, expected success markers, output locations, actual metrics, known failure modes, DeepID-vs-Mini-DeepID differences, and the closed-set limitation. State results exactly as observed and do not claim paper reproduction accuracy.

- [ ] **Step 9: Perform the final verification**

```powershell
& '.\.venv\Scripts\python.exe' -m pytest -q
git status --short
```

Expected: tests pass. Only intended source/docs changes appear; generated ignored data, environments, outputs, and checkpoints do not appear in Git status.

- [ ] **Step 10: Commit the verified result record**

```powershell
git add README.md
git commit -m "docs: record verified Mini-DeepID reproduction"
```

---

## Plan Self-Review Checklist

- [ ] No private photos or external identities are introduced.
- [ ] The dataset is exactly 10 identities × 50 samples and 35/7/8 per class.
- [ ] Every model consumer uses grayscale 64×64 normalization from the shared dataset module.
- [ ] Every checkpoint carries the identity mapping and refuses incompatible prediction.
- [ ] Test data is used only in Task 9 after model selection is frozen.
- [ ] Formal training is not run by the implementation agent; the user starts it from PyCharm after `PYCHARM_TRAIN_READY`.
- [ ] CUDA failures stop execution without CPU fallback.
- [ ] All required figures and JSON reports are created and visually checked.
- [ ] README distinguishes core DeepID ideas from Mini simplifications.
- [ ] No placeholder text, unfinished branch, or unverified success claim remains.
