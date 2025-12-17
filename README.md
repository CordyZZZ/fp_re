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
  --epochs 200 \
  --batch_size 24 \
  --lr 3e-4 \
  --activation relu
```

Outputs:

* `runs/foldpath_windows/config.json`
* `runs/foldpath_windows/checkpoints/last.pth`

## 4) Generate predictions
```bash
python generate_predictions.py \
  --checkpoint /workspace/tjl/foldpath_project/runs/foldpath_windows/checkpoints/last.pth \
  --config /workspace/tjl/foldpath_project/runs/foldpath_windows/config.json \
  --dataset windows-v2  \
  --data_root /fileStore/windows-v2 \
  --output_dir /workspace/tjl/foldpath_project/runs/foldpath_windows/ \
  --split test \
  --batch_size 4
```

Outputs:
* `runs/foldpath_windows/all_predictions.npy`

## 5) Evaluation
```bash
python quick_check.py \
  --dataset windows-v2 \
  --data_root /fileStore/windows-v2 \
  --ckpt runs/foldpath_windows/checkpoints/last.pth \
  --split test \
  --out_dir ./foldpath_windows/ \
  --save_predictions
```
Outputs:
* `runs/foldpath_windows/metrics.json`

Example Results:
|windows+ReLU| code | paper |
|--------|------|------|
| AP_DTW^50 | 88 | 91.4 |
| AP_DTW | 51.8 | 71.8 | 

|windows+Siren| code | paper |
|--------|------|------|
| AP_DTW^50 | 90 | 91.3 |
| AP_DTW | 52.10 | 71.9 | 

## 6) Visualization
```bash
python render_results_plt.py \
  --pred_dir /workspace/tjl/foldpath_project/runs/foldpath_windows \
  --normalized_root /fileStore/windows-v2-normalized \
  --sample_dirs 1_wr1fr_1 \
  --output_dir /workspace/tjl/foldpath_project/runs/foldpath_windows/ \
  --top_k_paths 4
```
Outputs:
* `runs/foldpath_windows/1_wr1fr_1_pred_vs_gt.png`
