import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from config import AUX_WEIGHT, EPOCHS, PATIENCE, LR, WD

try:
    from torch.amp import autocast as _amp_autocast
    autocast = lambda: _amp_autocast("cuda")
except ImportError:
    from torch.cuda.amp import autocast

from torch.cuda.amp import GradScaler

def train_epoch(model, loader, opt, sched, scaler):
    model.train()
    crit = nn.CrossEntropyLoss()
    total = 0.0
    device = next(model.parameters()).device
    for imgs, q, y in loader:
        imgs = imgs.to(device, non_blocking=True)
        q = q.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        opt.zero_grad()
        with autocast():
            logits, aux = model(imgs, q)
            loss = crit(logits, y) + AUX_WEIGHT * aux
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        total += loss.item()
    return total / len(loader)

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    preds, probs, labs = [], [], []
    device = next(model.parameters()).device
    for imgs, q, y in loader:
        imgs = imgs.to(device, non_blocking=True)
        q = q.to(device, non_blocking=True)
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

def fit(model, tr_ld, va_ld, te_ld, device="cuda"):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, total_steps=EPOCHS * len(tr_ld), pct_start=0.1,
    )
    try:
        scaler = torch.amp.GradScaler("cuda")
    except Exception:
        scaler = GradScaler()

    best_acc = 0.0
    best_state = None
    wait = 0
    for ep in range(1, EPOCHS + 1):
        loss = train_epoch(model, tr_ld, opt, sched, scaler)
        vm = evaluate(model, va_ld)
        if vm["acc"] > best_acc:
            best_acc = vm["acc"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if ep % 5 == 0 or ep == 1:
            print(f"  ep{ep:3d}: loss={loss:.4f} | val_acc={vm['acc']:.4f} f1={vm['f1']:.4f}")
        if wait >= PATIENCE:
            print(f"  Early stop ep{ep}")
            break
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return evaluate(model, te_ld)
