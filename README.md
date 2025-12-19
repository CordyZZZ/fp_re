# FoldPath (built from MaskPlanner codebase)

This repository is a **runnable** implementation of **FoldPath** ("End-to-End Object-Centric Motion Generation via Modulated Implicit Paths") built by **reusing** the local MaskPlanner project files you provided.

FoldPath differs from MaskPlanner in one key way: instead of predicting *unordered discrete waypoints / segments* and relying on post-processing, it learns each path as a **continuous function** of a scalar parameter, enabling **ordered, smooth** path sampling directly (no concatenation stage required). This is the central shift described in the FoldPath paper: representing each path as a neural field conditioned on object features and per-path embeddings.

## 1) Environment setup

### Clone the Repository
```bash
git clone https://github.com/CordyZZZ/fp_re.git
cd fp_re
```

### Environment Setup

```bash
conda create -n fp python=3.10
conda activate fp
pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 torchaudio==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## 2) Dataset layout (PaintNet)

This code expects the **same on-disk layout** as MaskPlanner:
In the command lines below, <DATA_ROOT> is taken as "/fileStore/windows-v2"

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
python generate_predictions_foldpath.py \
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
python eval_foldpath.py \
  --dataset windows-v2 \
  --data_root /fileStore/windows-v2 \
  --ckpt runs/foldpath_windows/checkpoints/last.pth \
  --split test \
  --out_dir ./foldpath_windows/ \
  --save_predictions
```
Outputs:
* `runs/foldpath_windows/metrics.json`


Warning: Forget to add LayerNorm for ReLU activation, so this result is expected to be replaced later.

Warning: I may misunderstand the word 'mean' in calculating AP_DTW …

Example Results:
#### reproduction of TABLE 1: cuboids, windows, and shelves
|windows+ReLU| code | paper |
|--------|------|------|
| AP_DTW^50 | 88 | 91.4 |
| AP_DTW | 51.8 | 71.8 | 

|windows+Siren| code | paper |
|--------|------|------|
| AP_DTW^50 | 90 | 91.3 |
| AP_DTW | 52.10 | 71.9 | 

| dataset | activation | AP_DTW | AP_DTW (paper) | AP_DTW^50 | AP_DTW^50 (paper)
|--------|--------|------|------| ------ | ------ |
| cuboids | relu | x | `35.2` | 90.5 | `59.8` |
| | siren | x | `60.3` | x | `97.5`|
| | firen | x | `91.1` | x | `99.2`|
| windows | relu | x | `71.8` |  | `91.4` |
| | siren | x | `71.9` | 90 | `91.3`|
| | firen | x | `75.0` | x | `91.9`|
| shelves | relu | x | `75.4` | x | `88.4` |
| | siren | x | `78.0` | x | `89.53`|
| | firen | x | `84.3` | x | `91.3`|

#### reproduction of TABLE 2: containers
| dataset | activation| AP_DTW^easy| AP_DTW^easy (paper) | Paint Cov. | Paint Cov. (paper)
|--------|--------|------|------| ------ | ------ |
| containers | finer | x | `13.7` | x | `91.1` |

PS: As noted in the paper, PCD metrics are dependent on the sampling rate and exhibit high sensitivity to outliers, rendering them unreliable and less informative in real-world scenarios. For this reason, we omit this metric from our reproduction and comparative analysis.

## 6) Visualization
To enhance the elegance and maintainability of this repository, I propose integrating this normalization operation into the data preprocessing pipeline. 
```bash
python normalize_dataset.py \
  --data_root /fileStore/windows-v2 \
  --output_root /fileStore/windows-v2-normalized \
  --normalization per-mesh
```
The resulting normalized dataset is then leveraged for visualization purposes.
```bash
python render_results_foldpath.py \
  --pred_dir /workspace/tjl/foldpath_project/runs/foldpath_windows \
  --normalized_root /fileStore/windows-v2-normalized \
  --sample_dirs 1_wr1fr_1 \
  --output_dir /workspace/tjl/foldpath_project/runs/foldpath_windows/ \
  --top_k_paths 4
```
Outputs:
* `runs/foldpath_windows/1_wr1fr_1_pred_vs_gt.png`
