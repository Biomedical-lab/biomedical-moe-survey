import os, sys, gc, time, argparse, random, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

from config import SEEDS, DATASETS, MODEL_NAMES, MODEL_LABELS
from models import get_model
from utils.datasets import get_loaders
from utils.training import fit

def parse_args():
    p = argparse.ArgumentParser(description="MoE Gating Survey — Experiment A")
    p.add_argument("--vqarad_root", type=str, required=True, help="Path to VQA_RAD dataset root")
    p.add_argument("--slake_root", type=str, required=True, help="Path to SLAKE dataset root")
    p.add_argument("--derm7pt_root", type=str, required=True, help="Path to derm7pt/release_v0 root")
    p.add_argument("--output_dir", type=str, default="./results", help="Directory to save results")
    p.add_argument("--device", type=str, default="cuda", help="Device to use")
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

    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}  |  VRAM: {props.total_mem / 1024**3:.1f} GB")
    print(f"PyTorch {torch.__version__}  |  CUDA {torch.version.cuda}")
    torch.backends.cudnn.benchmark = True

    print("=" * 65)
    print("EXPERIMENT A — Multi-seed (5 seeds x 6 models x 3 datasets)")
    print("=" * 65)

    all_rows = []

    for ds in DATASETS:
        print(f"\n>>> Dataset: {ds.upper()}")
        for seed in SEEDS:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            tr_ld, va_ld, te_ld, vocab_size, _ = get_loaders(ds, data_roots, seed)

            for mname in MODEL_NAMES:
                print(f"  [{MODEL_LABELS[mname]}] seed={seed}")
                t0 = time.time()
                model = get_model(mname, vocab_size)
                metrics = fit(model, tr_ld, va_ld, te_ld, device=DEVICE)
                elapsed = (time.time() - t0) / 60

                row = dict(
                    dataset=ds, model=mname, label=MODEL_LABELS[mname],
                    seed=seed, **metrics, time_min=round(elapsed, 1),
                )
                all_rows.append(row)
                print(f"    Acc={metrics['acc']:.4f} F1={metrics['f1']:.4f} "
                      f"AUC={metrics['auc']:.4f} ({elapsed:.1f}min)")

                ckpt_path = output_dir / f"ckpt_{ds}_{mname}_s{seed}.pt"
                torch.save(model.state_dict(), ckpt_path)

                del model
                gc.collect()
                torch.cuda.empty_cache()

    df = pd.DataFrame(all_rows)
    df.to_csv(output_dir / "experiment_a_results.csv", index=False)
    print(f"\nResults saved to {output_dir / 'experiment_a_results.csv'}")

    print("\n" + "=" * 65)
    print("SUMMARY (mean ± std over 5 seeds)")
    print("=" * 65)
    for ds in DATASETS:
        print(f"\n--- {ds.upper()} ---")
        sub = df[df["dataset"] == ds]
        for mname in MODEL_NAMES:
            ms = sub[sub["model"] == mname]
            print(f"  {MODEL_LABELS[mname]:35s}  "
                  f"Acc={ms['acc'].mean():.3f}±{ms['acc'].std():.3f}  "
                  f"F1={ms['f1'].mean():.3f}±{ms['f1'].std():.3f}  "
                  f"AUC={ms['auc'].mean():.3f}±{ms['auc'].std():.3f}")

if __name__ == "__main__":
    main()
