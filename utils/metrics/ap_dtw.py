import numpy as np
from .dtw import dtw_distance

def _extract_paths(points: np.ndarray, stroke_ids: np.ndarray):
    """
    points: (P, D), stroke_ids: (P,)
    returns list of (path_id, path_points)
    """
    points = np.asarray(points)
    stroke_ids = np.asarray(stroke_ids).astype(np.int64)
    out = []
    for pid in np.unique(stroke_ids):
        mask = stroke_ids == pid
        out.append((int(pid), points[mask]))
    return out

def _best_dtw(pred_path: np.ndarray, gt_path: np.ndarray, normalize: bool):
    d1 = dtw_distance(pred_path, gt_path, normalize=normalize)
    d2 = dtw_distance(pred_path, gt_path[::-1].copy(), normalize=normalize)
    return min(d1, d2)

def ap_dtw_single(pred_points, pred_stroke_ids, pred_confs,
                  gt_points, gt_stroke_ids,
                  taus, *, normalize_dtw: bool = True):
    """
    detection-style AP for one sample.
    - Predictions sorted by confidence (desc).
    - Each GT can match at most one prediction.
    - TP if best-direction DTW <= tau.
    Returns dict: tau -> AP
    """
    pred_paths = _extract_paths(pred_points, pred_stroke_ids)
    gt_paths = _extract_paths(gt_points, gt_stroke_ids)

    # map confidences
    conf_map = {}
    if pred_confs is None:
        for pid, _ in pred_paths:
            conf_map[pid] = 1.0
    else:
        pred_confs = np.asarray(pred_confs).reshape(-1)
        for pid, _ in pred_paths:
            if pid < len(pred_confs):
                conf_map[pid] = float(pred_confs[pid])
            else:
                conf_map[pid] = 1.0

    # sort preds by confidence
    pred_paths = sorted(pred_paths, key=lambda x: conf_map.get(x[0], 0.0), reverse=True)

    # precompute DTW matrix (pred x gt)
    dtw_mat = np.zeros((len(pred_paths), len(gt_paths)), dtype=np.float32)
    for i, (_, pp) in enumerate(pred_paths):
        for j, (_, gp) in enumerate(gt_paths):
            dtw_mat[i, j] = _best_dtw(pp, gp, normalize=normalize_dtw)

    results = {}
    for tau in taus:
        matched_gt = set()
        tps = []
        fps = []
        for i, (pid, _) in enumerate(pred_paths):
            # find best unmatched gt within tau
            best_j = None
            best_d = None
            for j, (gid, _) in enumerate(gt_paths):
                if j in matched_gt:
                    continue
                d = float(dtw_mat[i, j])
                if d <= tau and (best_d is None or d < best_d):
                    best_d = d
                    best_j = j
            if best_j is not None:
                matched_gt.add(best_j)
                tps.append(1)
                fps.append(0)
            else:
                tps.append(0)
                fps.append(1)

        tps = np.array(tps, dtype=np.float32)
        fps = np.array(fps, dtype=np.float32)
        if len(tps) == 0:
            results[float(tau)] = 0.0
            continue

        tp_cum = np.cumsum(tps)
        fp_cum = np.cumsum(fps)
        recalls = tp_cum / max(len(gt_paths), 1)
        precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-8)

        # standard AP: integrate precision envelope over recall
        mrec = np.concatenate(([0.0], recalls, [1.0]))
        mpre = np.concatenate(([0.0], precisions, [0.0]))
        for k in range(len(mpre)-2, -1, -1):
            mpre[k] = max(mpre[k], mpre[k+1])
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
        results[float(tau)] = ap
    return results

def ap_dtw_dataset(pred_list, gt_list, taus, *, normalize_dtw: bool = True):
    """
    pred_list: list of dict with keys traj_pred, stroke_ids_pred, conf_pred(optional)
    gt_list: list of dict with keys traj_as_pc, stroke_ids_as_pc
    Returns: dict with AP@tau and mAP
    """
    assert len(pred_list) == len(gt_list)
    per_tau = {float(t): [] for t in taus}
    for pred, gt in zip(pred_list, gt_list):
        ap_tau = ap_dtw_single(
            pred_points=pred['traj_pred'],
            pred_stroke_ids=pred['stroke_ids_pred'],
            pred_confs=pred.get('conf_pred'),
            gt_points=gt['traj_as_pc'],
            gt_stroke_ids=gt['stroke_ids_as_pc'],
            taus=taus,
            normalize_dtw=normalize_dtw,
        )
        for t, ap in ap_tau.items():
            per_tau[float(t)].append(ap)

    out = {f"AP@{t:.2f}": float(np.mean(per_tau[t])) if len(per_tau[t]) else 0.0 for t in per_tau}
    out["mAP"] = float(np.mean(list(out.values()))) if len(out) else 0.0
    return out
