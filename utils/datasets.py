import json, re
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
from sklearn.model_selection import train_test_split

from config import (
    IMG_SIZE, MAX_QUESTION_LEN, BATCH_SIZE, NUM_WORKERS,
    MELANOMA_LABELS,
)

def simple_tokenize(text):
    return re.sub(r"[^\w\s]", " ", text.lower().strip()).split()

def build_vocab(questions, min_freq=2):
    cnt = Counter(t for q in questions for t in simple_tokenize(q))
    vocab = ["<PAD>", "<UNK>"] + [w for w, c in cnt.most_common() if c >= min_freq]
    return vocab, {w: i for i, w in enumerate(vocab)}

def encode_question(text, tok2idx, max_len=MAX_QUESTION_LEN):
    ids = [tok2idx.get(t, 1) for t in simple_tokenize(text)[:max_len]]
    return ids + [0] * (max_len - len(ids))

TRAIN_TF = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.2, 0.2),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
VAL_TF = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

class VQARADDataset(Dataset):

    def __init__(self, df, tok2idx, tf=VAL_TF):
        self.df = df.reset_index(drop=True)
        self.tok2idx = tok2idx
        self.tf = tf
        print(f"    Caching {len(df)} images into RAM...", end=" ", flush=True)
        self.imgs = []
        for _, r in df.iterrows():
            try:
                self.imgs.append(self.tf(Image.open(r["img_path"]).convert("RGB")))
            except Exception:
                self.imgs.append(torch.zeros(3, IMG_SIZE, IMG_SIZE))
        print("done")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        q = torch.tensor(encode_question(r["question"], self.tok2idx), dtype=torch.long)
        return self.imgs[i], q, torch.tensor(r["y"], dtype=torch.long)

class SLAKEDataset(Dataset):

    def __init__(self, records, tok2idx, img_dir, tf=VAL_TF):
        self.records = records
        self.tok2idx = tok2idx
        self.img_dir = Path(img_dir)
        self.tf = tf
        print(f"    Caching {len(records)} images into RAM...", end=" ", flush=True)
        self.imgs = []
        for r in records:
            try:
                self.imgs.append(self.tf(Image.open(self.img_dir / r["img_name"]).convert("RGB")))
            except Exception:
                self.imgs.append(torch.zeros(3, IMG_SIZE, IMG_SIZE))
        print("done")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        q = torch.tensor(encode_question(r["question"], self.tok2idx), dtype=torch.long)
        return self.imgs[i], q, torch.tensor(r["y"], dtype=torch.long)

class Derm7ptDataset(Dataset):

    ATTR_COLS = [
        "pigment_network", "streaks", "pigmentation", "regression_structures",
        "dots_and_globules", "blue_whitish_veil", "vascular_structures",
    ]

    def __init__(self, df, tok2idx, img_dir, tf=VAL_TF):
        self.df = df.reset_index(drop=True)
        self.tok2idx = tok2idx
        self.img_dir = Path(img_dir)
        self.tf = tf
        print(f"    Caching {len(df)} derm7pt images...", end=" ", flush=True)
        self.imgs = []
        for _, r in df.iterrows():
            p = self.img_dir / r["clinic"]
            try:
                self.imgs.append(self.tf(Image.open(p).convert("RGB")))
            except Exception:
                self.imgs.append(torch.zeros(3, IMG_SIZE, IMG_SIZE))
        print("done")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        txt = ", ".join(f"{c}: {r[c]}" for c in self.ATTR_COLS)
        q = torch.tensor(encode_question(txt, self.tok2idx), dtype=torch.long)
        return self.imgs[idx], q, torch.tensor(r["y"], dtype=torch.long)

def load_vqarad(data_root, seed=42):
    with open(Path(data_root) / "VQA_RAD Dataset Public.json", encoding="utf-8") as f:
        raw = json.load(f)
    rows = []
    for r in raw:
        ans = str(r.get("answer", "")).strip().lower()
        if str(r.get("answer_type", "")).upper() == "CLOSED" and ans in ["yes", "no"]:
            p = Path(data_root) / "VQA_RAD_Image_Folder" / r["image_name"]
            if p.exists():
                rows.append(dict(
                    img_path=str(p), image_name=r["image_name"],
                    question=r["question"], y=1 if ans == "yes" else 0,
                ))
    df = pd.DataFrame(rows)
    imgs = df["image_name"].unique()
    tr, tmp = train_test_split(imgs, test_size=0.3, random_state=seed)
    va, te = train_test_split(tmp, test_size=0.5, random_state=seed)
    tr_df = df[df["image_name"].isin(tr)].reset_index(drop=True)
    va_df = df[df["image_name"].isin(va)].reset_index(drop=True)
    te_df = df[df["image_name"].isin(te)].reset_index(drop=True)
    v, t = build_vocab(tr_df["question"].tolist())
    return tr_df, va_df, te_df, v, t

def load_slake(data_root):
    splits = {}
    for spl, fn in [("train", "train.json"), ("val", "validation.json"), ("test", "test.json")]:
        with open(Path(data_root) / fn, encoding="utf-8") as f:
            data = json.load(f)
        rows = []
        for r in data:
            lang = r.get("q_lang", r.get("language", "en"))
            ans = str(r.get("answer", "")).strip().lower()
            if lang == "en" and str(r.get("answer_type", "")).upper() == "CLOSED" and ans in ["yes", "no"]:
                rows.append(dict(
                    img_name=str(r.get("img_name", r.get("img_id", ""))),
                    question=r["question"], y=1 if ans == "yes" else 0,
                ))
        splits[spl] = rows
    v, t = build_vocab([r["question"] for r in splits["train"]])
    return splits["train"], splits["val"], splits["test"], v, t

def load_derm7pt(data_root):
    root = Path(data_root)
    meta = pd.read_csv(root / "meta" / "meta.csv")
    meta["y"] = meta["diagnosis"].str.lower().apply(lambda d: 1 if d in MELANOMA_LABELS else 0)
    tr_idx = pd.read_csv(root / "meta" / "train_indexes.csv")["indexes"].values
    va_idx = pd.read_csv(root / "meta" / "valid_indexes.csv")["indexes"].values
    te_idx = pd.read_csv(root / "meta" / "test_indexes.csv")["indexes"].values
    tr_df = meta.iloc[tr_idx].reset_index(drop=True)
    va_df = meta.iloc[va_idx].reset_index(drop=True)
    te_df = meta.iloc[te_idx].reset_index(drop=True)
    all_txt = [", ".join(f"{c}: {r[c]}" for c in Derm7ptDataset.ATTR_COLS) for _, r in tr_df.iterrows()]
    vocab, tok2idx = build_vocab(all_txt)
    return tr_df, va_df, te_df, vocab, tok2idx

def get_loaders(ds_name, data_roots, seed=42):

    kw = dict(num_workers=NUM_WORKERS, pin_memory=True)
    prefetch = 2 if NUM_WORKERS > 0 else None

    if ds_name == "vqarad":
        tr, va, te, vocab, tok2idx = load_vqarad(data_roots["vqarad"], seed)
        tr_ds = VQARADDataset(tr, tok2idx, TRAIN_TF)
        va_ds = VQARADDataset(va, tok2idx)
        te_ds = VQARADDataset(te, tok2idx)
    elif ds_name == "slake":
        tr, va, te, vocab, tok2idx = load_slake(data_roots["slake"])
        idir = Path(data_roots["slake"]) / "imgs"
        tr_ds = SLAKEDataset(tr, tok2idx, idir, TRAIN_TF)
        va_ds = SLAKEDataset(va, tok2idx, idir)
        te_ds = SLAKEDataset(te, tok2idx, idir)
    elif ds_name == "derm7pt":
        tr, va, te, vocab, tok2idx = load_derm7pt(data_roots["derm7pt"])
        idir = Path(data_roots["derm7pt"]) / "images"
        tr_ds = Derm7ptDataset(tr, tok2idx, idir, TRAIN_TF)
        va_ds = Derm7ptDataset(va, tok2idx, idir)
        te_ds = Derm7ptDataset(te, tok2idx, idir)
    else:
        raise ValueError(f"Unknown dataset: {ds_name}")

    tr_ld = DataLoader(tr_ds, BATCH_SIZE, shuffle=True, drop_last=True, prefetch_factor=prefetch, **kw)
    va_ld = DataLoader(va_ds, BATCH_SIZE * 2, shuffle=False, prefetch_factor=prefetch, **kw)
    te_ld = DataLoader(te_ds, BATCH_SIZE * 2, shuffle=False, prefetch_factor=prefetch, **kw)
    print(f"[{ds_name.upper()}] Train:{len(tr_ds)} Val:{len(va_ds)} Test:{len(te_ds)} Vocab:{len(vocab)}")
    return tr_ld, va_ld, te_ld, len(vocab), tok2idx
