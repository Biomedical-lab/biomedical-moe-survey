import argparse, warnings
from pathlib import Path

import torch
import pandas as pd

warnings.filterwarnings("ignore")

from config import DATASETS, HIDDEN_DIM
from models import get_model
from utils.datasets import get_loaders
from utils.training import evaluate

try:
    from torch.amp import autocast as _amp_autocast
    autocast = lambda: _amp_autocast("cuda")
except ImportError:
    from torch.cuda.amp import autocast

def parse_args():
    p = argparse.ArgumentParser(description="MoE Gating Survey — Experiment B")
    p.add_argument("--vqarad_root", type=str, required=True)
    p.add_argument("--slake_root", type=str, required=True)
    p.add_argument("--derm7pt_root", type=str, required=True)
    p.add_argument("--checkpoint_dir", type=str, default="./results")
    p.add_argument("--output_dir", type=str, default="./results")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()

@torch.no_grad()
def evaluate_missing(model, loader, condition, device):

    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    model.eval()
    preds, probs, labs = [], [], []
    for imgs, q, y in loader:
        imgs = imgs.to(device, non_blocking=True)
        q = q.to(device, non_blocking=True)

        if condition == "no_image":
            imgs = torch.zeros_like(imgs)
        elif condition == "no_text":
            q = torch.zeros_like(q)
        elif condition == "half_mask":
            mask = torch.ones_like(imgs)
            mask[:, :, :, imgs.shape[3] // 2:] = 0
            imgs = imgs * mask

        with autocast():
            logits, _ = model(imgs, q)
        probs += torch.softmax(logits, -1)[:, 1].cpu().tolist()
        preds += logits.argmax(-1).cpu().tolist()
        labs += y.tolist()

    acc = accuracy_score(labs, preds)
    f1 = f1_score(labs, preds, average="macro", zero_division=0)
    try:
        auc = roc_auc_score(labs, probs)
    except Exception:
        auc = 0.0
    return dict(acc=acc, f1=f1, auc=auc)

def main():
    args = parse_args()
    DEVICE = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    ckpt_dir = Path(args.checkpoint_dir)

    data_roots = {
        "vqarad": args.vqarad_root,
        "slake": args.slake_root,
        "derm7pt": args.derm7pt_root,
    }

    CONDITIONS = [
        ("full", "Full (image and text)"),
        ("no_image", "No image (zeros)"),
        ("no_text", "No text (zeros)"),
        ("half_mask", "Half image masked"),
    ]

    print("=" * 65)
    print("EXPERIMENT B — Missing Modality Robustness (A1.3 Soft Gate)")
    print("=" * 65)

    all_rows = []
    for ds in DATASETS:
        print(f"\n>>> Dataset: {ds.upper()}")
        _, _, te_ld, vocab_size, _ = get_loaders(ds, data_roots, args.seed)

        model = get_model("soft_gate", vocab_size).to(DEVICE)
        ckpt = ckpt_dir / f"ckpt_{ds}_soft_gate_s{args.seed}.pt"
        if ckpt.exists():
            model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
            print(f"  Loaded checkpoint: {ckpt.name}")
        else:
            print(f"  WARNING: No checkpoint found at {ckpt}, using random weights!")

        full_acc = None
        for cond_key, cond_label in CONDITIONS:
            metrics = evaluate_missing(model, te_ld, cond_key, DEVICE)
            if cond_key == "full":
                full_acc = metrics["acc"]
                delta = 0.0
            else:
                delta = metrics["acc"] - full_acc

            row = dict(dataset=ds, condition=cond_label, acc=metrics["acc"], delta=delta)
            all_rows.append(row)
            print(f"  {cond_label:25s}  Acc={metrics['acc']:.3f}  Δ={delta:+.3f}")

    df = pd.DataFrame(all_rows)
    df.to_csv(output_dir / "experiment_b_results.csv", index=False)
    print(f"\nResults saved to {output_dir / 'experiment_b_results.csv'}")

if __name__ == "__main__":
    main()
