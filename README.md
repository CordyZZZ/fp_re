# FoldPath

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
In the command lines below, <DATA_ROOT> is taken as "/fileStore/cuboids-v2"

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
  --dataset cuboids-v2 \
  --data_root /fileStore/cuboids-v2 \
  --out_dir runs/foldpath_cuboids_finer \
  --epochs 200 \
  --batch_size 24 \
  --lr 3e-4 \
  --activation finer
```

Outputs:

* `runs/foldpath_cuboids_finer/config.json`
* `runs/foldpath_cuboids_finer/checkpoints/last.pth`

## 4) Inference
```bash
python inference.py \
  --checkpoint runs/foldpath_cuboids_finer/checkpoints/last.pth \
  --data_root ./../dataset/cuboids-v2 \
  --sample_dir 167_cube_1001_1459_988 \
  --output ./runs/foldpath_cuboids_finer/foldpath_cuboids_predictions_167_finer.npy 
```
Outputs:

* `runs/foldpath_cuboids_finer/foldpath_cuboids_predictions_167_finer.json`
* `runs/foldpath_cuboids_finer/foldpath_cuboids_predictions_167_finer.npy`

## 5) Evaluation
```bash
python eval_foldpath.py \
  --dataset windows-v2 \
  --data_root /fileStore/windows-v2 \
  --ckpt runs/foldpath_windows_relu/checkpoints/last.pth \
  --activation relu \
  --split test \
  --out_dir ./foldpath_windows_relu/ \
  --save_predictions
```
  
Outputs:
* `runs/foldpath_windows_relu/metrics.json`


#### reproduction of TABLE 1: cuboids, windows, and shelves

| dataset | activation | AP_DTW | AP_DTW (paper) | AP_DTW^50 | AP_DTW^50 (paper)
|--------|--------|------|------| ------ | ------ |
| cuboids | relu | x | `35.2` | 90.5 | `59.8` |
| | siren | x | `60.3` | x | `97.5`|
| | firen | x | `91.1` | x | `99.2`|
| windows | relu | x | `71.8` |  | `91.4` |
| | siren | x | `71.9` | 90 | `91.3`|
| | firen | x | `75.0` | x | `91.9`|
| shelves | relu | x | `75.4` | x | `88.4` |
| | siren | x | `78.0` | x | `89.5`|
| | firen | x | `84.3` | x | `91.3`|

#### reproduction of TABLE 2: containers
| dataset | activation| AP_DTW^easy| AP_DTW^easy (paper) |
|--------|--------|------|------|
| containers | finer | x | `13.7` |

PS: As noted in the paper, PCD metrics are dependent on the sampling rate and exhibit high sensitivity to outliers, rendering them unreliable and less informative in real-world scenarios. For this reason, we omit this metric from our reproduction and comparative analysis.

## 6) Normalization
I have already integrated the normalization step into the visualization module, so there is no longer a need to perform normalization separately. That said, I have retained the normalization code for verification and auditing purposes.
```bash
python normalize_dataset.py \
  --data_root /fileStore/cuboids-v2 \
  --output_root /fileStore/cuboids-v2-normalized \
  --normalization per-mesh
```

## 7) Visualization
To accommodate different server environments, I propose the following three implementation options:
1. Utilize the VTK library to generate a static visualization;
2. Utilize the VTK library to generate an interactively rotatable visualization;
3. Utilize the Matplotlib library for scenarios where the server lacks X11 display capabilities.

```bash
python visualize_foldpath.py \
  --inference_file ./runs/foldpath_cuboids_finer/foldpath_cuboids_predictions_167_finer.npy \
  --data_root ./../dataset/cuboids-v2 \
  --sample_dir 167_cube_1001_1459_988 \
  --max_trajectories 6 \
  --output_dir ./runs/foldpath_cuboids_finer
```

```bash
## interactively rotatable version
python viz_foldpath.py \
  --inference_file ./runs/foldpath_cuboids_finer/foldpath_cuboids_predictions_167_finer.npy \
  --data_root ./../dataset/cuboids-v2 \
  --sample_dir 167_cube_1001_1459_988 \
  --max_trajectories 6
```

Outputs:
* `runs/foldpath_cuboids_finer/167_cube_1001_1459_988_foldpath.png`

Example Outcome:
| visualize_foldpath | viz_foldpath |
| --- | --- |
| <img width="500" height="500" alt="167_cube_1001_1459_988_foldpath" src="https://github.com/user-attachments/assets/4c219343-ae8e-478e-8f4a-f9064affa44e" /> | <img width="350" height="350" alt="167_cube_1001_1459_988_foldpath_scroll" src="https://github.com/user-attachments/assets/9cb22089-a45a-4022-8779-c308112ccc75" /> |

