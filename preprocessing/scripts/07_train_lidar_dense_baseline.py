from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from uavssc.dataset import LocalGridDataset
from uavssc.models import DenseLidarSSCUNet



def compute_losses(batch: dict[str, torch.Tensor], pred: dict[str, torch.Tensor], num_classes: int) -> tuple[torch.Tensor, dict[str, float]]:
    occ_gt = batch['occ']
    known = batch['known']
    sem_gt = batch['sem']

    occ_logits = pred['occ_logits']
    sem_logits = pred['sem_logits']

    known_mask = known > 0.5
    occ_loss_map = F.binary_cross_entropy_with_logits(occ_logits, occ_gt, reduction='none')
    occ_loss = occ_loss_map[known_mask].mean() if known_mask.any() else occ_loss_map.mean() * 0.0

    occ_vox = occ_gt[:, 0] > 0.5
    sem_valid = occ_vox & known_mask[:, 0]
    if sem_valid.any():
        sem_loss = F.cross_entropy(sem_logits.permute(0, 2, 3, 4, 1)[sem_valid], sem_gt[sem_valid] - 1)
    else:
        sem_loss = occ_loss * 0.0

    loss = occ_loss + sem_loss
    return loss, {
        'loss': float(loss.detach().cpu()),
        'occ_loss': float(occ_loss.detach().cpu()),
        'sem_loss': float(sem_loss.detach().cpu()),
    }



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', type=str, required=True, help='Directory created by script 06')
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch-size', type=int, default=2)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--num-classes', type=int, default=18)
    ap.add_argument('--num-workers', type=int, default=0)
    ap.add_argument('--save-dir', type=str, default='artifacts/checkpoints_dense_baseline')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = LocalGridDataset(args.data_root, input_key='input_occ_lidar')
    n_val = max(1, int(0.1 * len(ds)))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = DenseLidarSSCUNet(in_channels=1, num_classes=args.num_classes, base_ch=16).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_val = float('inf')

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f'train epoch {epoch}')
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            pred = model(batch['x'])
            loss, logs = compute_losses(batch, pred, num_classes=args.num_classes)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            train_loss += logs['loss']
            pbar.set_postfix(loss=f"{logs['loss']:.4f}", occ=f"{logs['occ_loss']:.4f}", sem=f"{logs['sem_loss']:.4f}")
        train_loss /= max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'val epoch {epoch}'):
                batch = {k: v.to(device) for k, v in batch.items()}
                pred = model(batch['x'])
                loss, logs = compute_losses(batch, pred, num_classes=args.num_classes)
                val_loss += logs['loss']
        val_loss /= max(1, len(val_loader))
        print(f'Epoch {epoch}: train={train_loss:.4f} val={val_loss:.4f}')

        ckpt = {
            'model': model.state_dict(),
            'optimizer': opt.state_dict(),
            'epoch': epoch,
            'val_loss': val_loss,
        }
        torch.save(ckpt, save_dir / 'last.pt')
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, save_dir / 'best.pt')


if __name__ == '__main__':
    main()
