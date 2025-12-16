import os, json, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from models.foldpath import FoldPath
from utils.dataset.foldpath_dataset import FoldPathDataset, FoldPathDatasetConfig
from utils.metrics.ap_dtw import ap_dtw_dataset

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--data_root', required=True)
    p.add_argument('--split', default='test', choices=['train','test'])
    p.add_argument('--ckpt', required=True)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--num_workers', type=int, default=4)

    p.add_argument('--T', type=int, default=384)
    p.add_argument('--max_paths', type=int, default=40)
    p.add_argument('--conf_thresh', type=float, default=0.35)

    p.add_argument('--dtw_taus', type=float, nargs='+',
                   default=[0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95])
    p.add_argument('--normalize_dtw', action='store_true')

    p.add_argument('--out', default='eval_results.json')
    p.add_argument('--dump_preds', action='store_true',
                   help='If set, also dump per-sample predictions to --pred_out (same schema as predict_foldpath.py).')
    p.add_argument('--pred_out', default='eval_preds_foldpath.npy')
    p.add_argument('--pred_file', default=None,
                   help='If provided, skip inference and evaluate using a dumped predictions file.')
    return p.parse_args()

@torch.no_grad()
def run_inference(args):
    # ds = FoldPathDataset( 
    #     dataset=args.dataset,
    #     data_root=args.data_root,
    #     split=args.split,
    #     T=args.T,
    #     sampling='equispaced',
    # )
    
    cfg = FoldPathDatasetConfig(
        dataset=args.dataset,
        roots=[args.data_root],
        split=args.split,
        pc_points=5120,
        normalization="per-mesh",
        num_queries=args.max_paths,
        T=args.T,
        sampling="equispaced",
    )

    ds = FoldPathDataset(cfg)
    
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FoldPath()
    ckpt = torch.load(args.ckpt, map_location='cpu')
    state = ckpt.get('model', ckpt)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()

    pred_list, gt_list, dirnames = [], [], []
    
    for batch in dl:
        # FoldPathDataset returns dict-like
        pc = batch['pc'].to(device)          # (B, N, 3)
        # other conditioning (normals/features) are handled inside dataset/model if needed
        y_gt = batch['y']  # Ground truth paths (B, Q, T, 6)
        f_gt = batch['f']  # Path existence mask (B, Q)
        
        out = model.infer(pc, T=args.T, max_paths=args.max_paths, conf_thresh=args.conf_thresh)

        # out is list of per-sample dicts
        for i in range(len(out)):
            pred_list.append({
                'traj_pred': out[i]['traj_pred'].astype(np.float32),
                'stroke_ids_pred': out[i]['stroke_ids_pred'].astype(np.int64),
                'conf_pred': out[i].get('conf_pred', None),
            })

            # DEBUG
            real_mask = f_gt[i] > 0.5
            if real_mask.sum() > 0:
                real_paths = y_gt[i][real_mask]  # (num_real_paths, T, 6)
                traj_list = []
                stroke_ids_list = []
                for path_idx, path in enumerate(real_paths):
                    traj_list.append(path.numpy())
                    stroke_ids_list.append(np.full(len(path), path_idx))
                
                if traj_list:
                    traj_gt = np.concatenate(traj_list, axis=0)
                    stroke_ids_gt = np.concatenate(stroke_ids_list, axis=0)
                else:
                    traj_gt = np.zeros((0, 6), dtype=np.float32)
                    stroke_ids_gt = np.zeros((0,), dtype=np.int64)
            else:
                traj_gt = np.zeros((0, 6), dtype=np.float32)
                stroke_ids_gt = np.zeros((0,), dtype=np.int64)

            gt_list.append({
                'traj_as_pc': traj_gt,
                'stroke_ids_as_pc': stroke_ids_gt,
            })
            dirnames.append(batch['name'][i])

            # gt_list.append({
            #     'traj_as_pc': batch['traj_as_pc'][i].numpy().astype(np.float32),
            #     'stroke_ids_as_pc': batch['stroke_ids_as_pc'][i].numpy().astype(np.int64),
            # })
            # dirnames.append(batch['dirname'][i])

    return pred_list, gt_list, dirnames

def main():
    args = parse_args()

    if args.pred_file is not None:
        data = np.load(args.pred_file, allow_pickle=True).item()
        pred_list = data['pred_list']
        gt_list = data['gt_list']
        dirnames = data.get('dirnames', [str(i) for i in range(len(pred_list))])
    else:
        pred_list, gt_list, dirnames = run_inference(args)

    metrics = ap_dtw_dataset(pred_list, gt_list, taus=args.dtw_taus, normalize_dtw=args.normalize_dtw)

    out_obj = {
        'dataset': args.dataset,
        'split': args.split,
        'ckpt': args.ckpt,
        'T': args.T,
        'conf_thresh': args.conf_thresh,
        'max_paths': args.max_paths,
        'dtw_taus': args.dtw_taus,
        'normalize_dtw': bool(args.normalize_dtw),
        'metrics': metrics,
    }
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out_obj, f, indent=2)

    print(json.dumps(metrics, indent=2))

    if args.dump_preds and args.pred_file is None:
        np.save(args.pred_out, {
            'dirnames': dirnames,
            'pred_list': pred_list,
            'gt_list': gt_list,
        }, allow_pickle=True)
        print(f"[Saved] {args.pred_out}")

if __name__ == '__main__':
    main()
