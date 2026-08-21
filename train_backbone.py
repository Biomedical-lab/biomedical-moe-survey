import argparse, warnings, random, time
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")

from config import SEEDS, LR, WD, EPOCHS, PATIENCE, HIDDEN_DIM
from models import BaselineConcat
from utils.datasets import get_loaders
from utils.training import fit

def parse_args():
    p = argparse.ArgumentParser(description="MoE Gating Survey — Backbone Training")
    p.add_argument("--vqarad_root", type=str, required=True)
    p.add_argument("--slake_root", type=str, required=True)
    p.add_argument("--derm7pt_root", type=str, required=True)
    p.add_argument("--dataset", type=str, default="vqarad", choices=["vqarad", "slake", "derm7pt"])
    p.add_argument("--output_dir", type=str, default="./results")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()

def main():
    args = parse_args()
    assert torch.cuda.is_available(), "CUDA not available!"

    DEVICE = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    data_roots = {
        "vqarad": args.vqarad_root,
        "slake": args.slake_root,
        "derm7pt": args.derm7pt_root,
    }

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    print("=" * 65)
    print(f"BACKBONE TRAINING — {args.dataset.upper()} (seed={args.seed})")
    print("=" * 65)

    tr_ld, va_ld, te_ld, vocab_size, _ = get_loaders(args.dataset, data_roots, args.seed)

    model = BaselineConcat(vocab_size, HIDDEN_DIM)
    t0 = time.time()
    metrics = fit(model, tr_ld, va_ld, te_ld, device=DEVICE)
    elapsed = (time.time() - t0) / 60

    print(f"\nBackbone results: Acc={metrics['acc']:.4f} F1={metrics['f1']:.4f} "
          f"AUC={metrics['auc']:.4f} ({elapsed:.1f}min)")

    ckpt_path = output_dir / f"backbone_{args.dataset}_s{args.seed}.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved to {ckpt_path}")

if __name__ == "__main__":
    main()
