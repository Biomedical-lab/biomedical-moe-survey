IMG_SIZE = 224
MAX_QUESTION_LEN = 32

BATCH_SIZE = 128
EPOCHS = 40
PATIENCE = 8
LR = 2e-4
WD = 1e-4
NUM_WORKERS = 0          
AUX_WEIGHT = 1e-3        

HIDDEN_DIM = 512

SEEDS = [42, 43, 44, 45, 46]

DATASETS = ["vqarad", "slake", "derm7pt"]

MODEL_NAMES = [
    "baseline", "gmu", "sparse_moe",
    "soft_gate", "cross_attn", "modality_specific",
]

MODEL_LABELS = {
    "baseline":          "A1.0 Baseline (Concat)",
    "gmu":               "A1.1 GMU (Dense Softmax)",
    "sparse_moe":        "A1.2 Sparse Top-k MoE",
    "soft_gate":         "A1.3 Soft Gate",
    "cross_attn":        "A1.4 Cross-Attention",
    "modality_specific": "A1.5 Modality-Specific",
}

MELANOMA_LABELS = [
    "melanoma", "melanoma (in situ)", "melanoma (less than 0.76 mm)",
    "melanoma (0.76 to 1.5 mm)", "melanoma (more than 1.5 mm)",
    "melanoma metastasis",
]
