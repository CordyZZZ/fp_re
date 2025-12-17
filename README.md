# FoldPath (built from MaskPlanner codebase)

This repository is a **runnable** implementation of **FoldPath** ("End-to-End Object-Centric Motion Generation via Modulated Implicit Paths") built by **reusing** the local MaskPlanner project files you provided.

FoldPath differs from MaskPlanner in one key way: instead of predicting *unordered discrete waypoints / segments* and relying on post-processing, it learns each path as a **continuous function** of a scalar parameter, enabling **ordered, smooth** path sampling directly (no concatenation stage required). This is the central shift described in the FoldPath paper: representing each path as a neural field conditioned on object features and per-path embeddings.

## What is included / not included

Included:

* PointNet++ utilities and basic geometry/dataset I/O reused from MaskPlanner.
* A FoldPath model implementation:
  * PointNet++ encoder producing visual tokens `z` (length 256)
  * transformer decoder with `N` learned path queries
  * modulated MLP path head mapping `s \in [-1,1] -> (x,y,z, vx,vy,vz)`
  * Hungarian matching on path positions and the losses described in the paper
* A minimal training script: `train_foldpath.py`
* A dataset wrapper that repackages PaintNet trajectories into **sets of paths**:
  `utils/dataset/foldpath_dataset.py`

Not included:

* **Any pretrained weights** (per your requirement).
* Proprietary simulation code and paint coverage evaluation.
* Official FINER activation (we provide a runnable approximation; see TODO in code).

## 1) Environment setup

Tested with Python 3.10+ and PyTorch 2.x.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Requirements / dependencies

The original MaskPlanner repo includes extra dependencies for many baselines. This FoldPath subset only needs:

* torch
* numpy
* scipy
* omegaconf (used by reused dataset code)
* tqdm

If your environment already runs MaskPlanner, you are fine.

## 2) Dataset layout (PaintNet)

This code expects the **same on-disk layout** as MaskPlanner:

```
<DATA_ROOT>/
  train_split.json
  test_split.json
  <SAMPLE_ID_0>/
    <SAMPLE_ID_0>.obj
    trajectory.txt
  <SAMPLE_ID_1>/
    <SAMPLE_ID_1>.obj
    trajectory.txt
  ...
```

Notes:

* `trajectory.txt` should store 6D poses (position + orientation vector) in the PaintNet format.
* The FoldPath dataset wrapper splits trajectories into paths using `stroke_ids` (as in MaskPlanner), then linearly resamples each path to `T` points.

If your local dataset uses a different convention (e.g., multiple strokes per path, or post-processed paths only), you will need to adapt:

* `utils/dataset/foldpath_dataset.py::_split_by_stroke_ids()`

## 3) Training

Example (windows):

```bash
python train_foldpath.py \
  --dataset windows-v2 \
  --data_root /fileStore/windows-v2 \
  --out_dir runs/foldpath_windows \
  --epochs 200 --batch_size 24 --lr 3e-4 \
  --activation relu
```

Common switches:

* `--activation {relu,siren,finer}`
  * `finer` is a runnable approximation (see TODO in `models/foldpath.py`).
* `--T_train 64 --T_test 384` (paper defaults)
* `--num_queries 40` (paper default)

Outputs:

* `runs/<...>/config.json`
* `runs/<...>/checkpoints/last.pth`

## 4) Evaluation(new)
```bash
python quick_check.py
    --dataset windows-v2
    --data_root /fileStore/windows-v2
    --ckpt runs/foldpath_windows/checkpoint.pth
    --split test
    --out_dir ./eval_results
    --save_predictions
```
Results are stored in JSON files under the eval_results folder.

Example Results:
|windows+ReLU| code | paper |
|--------|------|------|
| AP_DTW^50 | 88 | 91.4 |
| AP_DTW | 51.8 | 71.8 | 

## 4) Inference / sampling

FoldPath is designed to generate a whole path by sampling scalars `s` in `[-1,1]`.

Minimal code snippet:

```python
import torch
from models.foldpath import FoldPath, FoldPathConfig

model = FoldPath(FoldPathConfig()).eval().cuda()
pc = torch.randn(1, 5120, 3, device='cuda')
s = torch.linspace(-1, 1, 384, device='cuda').view(1, 384, 1)
y_hat, f_hat = model(pc, s)  # y_hat: [1, 40, 384, 6]
```

To select the most relevant paths, threshold the confidences:

```python
keep = f_hat[0] > 0.5
paths = y_hat[0, keep]  # [K, 384, 6]
```

## 5) TODO markers (expected missing pieces)

This repo is runnable as-is, but the following are explicitly left as TODOs to avoid inventing assumptions beyond the paper:

1. **FINER activation exact reproduction**: see `models/foldpath.py::VariablePeriodic`.
2. **Full AP-DTW metric and paint-coverage evaluation**: FoldPath paper introduces AP metrics using DTW; MaskPlanner has evaluation utilities, but integrating them end-to-end is task-specific.
3. **Exact category-specific hyperparameters**: you can add config files per category if you want parity with the paper.

## 6) Troubleshooting

* If you see missing files such as `train_split.json` / `.obj` / `trajectory.txt`, your dataset root does not match the expected layout.
* If orientations are not 3D unit vectors, normalize or convert them in the dataset loader.

## License / provenance

This implementation reuses substantial components from the MaskPlanner project you provided locally (PointNet++ utilities, data readers, and general project structure). No pretrained weights are distributed in this repository.


## Evaluation (AP-DTW)

This repo includes a runnable AP-DTW implementation (DTW-based detection AP as described in the FoldPath paper).

```bash
python eval_foldpath.py \
  --dataset cuboids-v2 \
  --data_root /path/to/PaintNet/cuboids-v2 \
  --split test \
  --ckpt runs/foldpath_cuboids/checkpoints/last.pth \
  --T 384 --conf_thresh 0.35 --max_paths 40 \
  --normalize_dtw
```

### TODO: DTW thresholds

You **must** calibrate `--dtw_taus` to your trajectory units / normalization. The defaults are placeholders and may not correspond to your dataset scale.

### Avoid重复推理：dump preds

```bash
python eval_foldpath.py \
  --dataset cuboids-v2 --data_root /path/to/PaintNet/cuboids-v2 --split test \
  --ckpt runs/foldpath_cuboids/checkpoints/last.pth \
  --dump_preds --pred_out runs/foldpath_cuboids/eval_preds_foldpath.npy
```

Then evaluate only:

```bash
python eval_foldpath.py \
  --dataset cuboids-v2 --data_root /path/to/PaintNet/cuboids-v2 --split test \
  --ckpt DUMMY --pred_file runs/foldpath_cuboids/eval_preds_foldpath.npy
```

## Inference dump (FoldPath → .npy)

```bash
python predict_foldpath.py \
  --dataset cuboids-v2 \
  --data_root /path/to/PaintNet/cuboids-v2 \
  --split test \
  --ckpt runs/foldpath_cuboids/checkpoints/last.pth \
  --T 384 --conf_thresh 0.35 --max_paths 40
```

Outputs a `.npy` file under `<run>/predictions_foldpath/`.

## Visualization

Use the existing MaskPlanner `render_results.py` with the new `--pred_dir` option:

```bash
python render_results.py --run runs/foldpath_cuboids --pred_dir predictions_foldpath --split test \
  --top_k_paths 20 --conf_thresh 0.35 --sidebyside --display
```
