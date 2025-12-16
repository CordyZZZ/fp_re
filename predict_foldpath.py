import os, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from models.foldpath import FoldPath
from utils.dataset.foldpath_dataset import FoldPathDataset

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--data_root', required=True)
    p.add_argument('--split', default='test', choices=['train','test'])
    p.add_argument('--ckpt', required=True)
    p.add_argument('--out_dir', default=None, help='If None, writes to <ckpt_dir>/predictions_foldpath')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--num_workers', type=int, default=4)

    p.add_argument('--T', type=int, default=384)
    p.add_argument('--max_paths', type=int, default=40)
    p.add_argument('--conf_thresh', type=float, default=0.35)

    return p.parse_args()

@torch.no_grad()
def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FoldPath()
    ckpt = torch.load(args.ckpt, map_location='cpu')
    state = ckpt.get('model', ckpt)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()

    ds = FoldPathDataset(dataset=args.dataset, data_root=args.data_root, split=args.split,
                         T=args.T, sampling='equispaced')
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    if args.out_dir is None:
        ckpt_dir = os.path.dirname(os.path.abspath(args.ckpt))
        run_dir = os.path.dirname(ckpt_dir)
        out_dir = os.path.join(run_dir, 'predictions_foldpath')
    else:
        out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    all_dirnames = []
    all_traj_gt, all_sid_gt = [], []
    all_traj_pred, all_sid_pred, all_conf_pred = [], [], []

    for batch in dl:
        pc = batch['pc'].to(device)
        out = model.infer(pc, T=args.T, max_paths=args.max_paths, conf_thresh=args.conf_thresh)

        for i in range(len(out)):
            all_dirnames.append(batch['dirname'][i])
            all_traj_gt.append(batch['traj_as_pc'][i].numpy())
            all_sid_gt.append(batch['stroke_ids_as_pc'][i].numpy())

            all_traj_pred.append(out[i]['traj_pred'].astype(np.float32))
            all_sid_pred.append(out[i]['stroke_ids_pred'].astype(np.int64))
            all_conf_pred.append(out[i].get('conf_pred', None))

    fname = f"foldpath_{args.split}_T{args.T}_th{args.conf_thresh:.2f}.npy"
    np.save(os.path.join(out_dir, fname), {
        'dirnames': np.array(all_dirnames),
        'traj_as_pc': np.array(all_traj_gt, dtype=object),
        'stroke_ids_as_pc': np.array(all_sid_gt, dtype=object),
        'traj_pred': np.array(all_traj_pred, dtype=object),
        'stroke_ids_pred': np.array(all_sid_pred, dtype=object),
        'conf_pred': np.array(all_conf_pred, dtype=object),
    }, allow_pickle=True)
    print(f"[Saved] {os.path.join(out_dir, fname)}")

if __name__ == '__main__':
    main()
