import torch
import torch.nn as nn
import torchvision.models as tvm

from config import HIDDEN_DIM

class ImageEncoder(nn.Module):
    def __init__(self, hd=HIDDEN_DIM):
        super().__init__()
        net = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1)
        self.bb = nn.Sequential(*list(net.children())[:-1])
        self.proj = nn.Sequential(nn.Linear(2048, hd), nn.BatchNorm1d(hd), nn.Dropout(0.3))

    def forward(self, x):
        return self.proj(self.bb(x).squeeze(-1).squeeze(-1))

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, hd=HIDDEN_DIM):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, 128, padding_idx=0)
        self.lstm = nn.LSTM(128, hd // 2, 2, batch_first=True, bidirectional=True, dropout=0.3)
        self.proj = nn.Sequential(nn.Linear(hd, hd), nn.BatchNorm1d(hd), nn.Dropout(0.3))

    def forward(self, x):
        _, (h, _) = self.lstm(self.emb(x))
        return self.proj(torch.cat([h[-2], h[-1]], -1))

class Clf(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 2))

    def forward(self, x):
        return self.net(x)

class Base(nn.Module):
    def __init__(self, vs, hd=HIDDEN_DIM):
        super().__init__()
        self.ie = ImageEncoder(hd)
        self.te = TextEncoder(vs, hd)
        self.hd = hd

    def enc(self, img, q):
        return self.ie(img), self.te(q)

class BaselineConcat(Base):
    def __init__(self, vs, hd=HIDDEN_DIM):
        super().__init__(vs, hd)
        self.fuse = nn.Sequential(nn.Linear(hd * 2, hd), nn.ReLU(), nn.Dropout(0.3))
        self.clf = Clf(hd)

    def forward(self, img, q):
        hi, ht = self.enc(img, q)
        return self.clf(self.fuse(torch.cat([hi, ht], -1))), torch.tensor(0.0, device=img.device)

class GMUModel(Base):
    def __init__(self, vs, hd=HIDDEN_DIM):
        super().__init__(vs, hd)
        self.Wv = nn.Linear(hd, hd)
        self.Wu = nn.Linear(hd, hd)
        self.Wg = nn.Linear(hd * 2, hd)
        self.clf = Clf(hd)

    def forward(self, img, q):
        hi, ht = self.enc(img, q)
        g = torch.sigmoid(self.Wg(torch.cat([hi, ht], -1)))
        return self.clf(g * torch.tanh(self.Wv(hi)) + (1 - g) * torch.tanh(self.Wu(ht))), \
               torch.tensor(0.0, device=img.device)

class FFN(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(0.1), nn.Linear(d, d))

    def forward(self, x):
        return self.net(x)

class SparseMoEModel(Base):
    def __init__(self, vs, hd=HIDDEN_DIM, ne=4, k=2):
        super().__init__(vs, hd)
        D = hd * 2
        self.ne = ne
        self.k = k
        self.gate = nn.Linear(D, ne)
        self.experts = nn.ModuleList([FFN(D) for _ in range(ne)])
        self.proj = nn.Linear(D, hd)
        self.clf = Clf(hd)

    def forward(self, img, q):
        hi, ht = self.enc(img, q)
        x = torch.cat([hi, ht], -1)
        sc = torch.softmax(self.gate(x), -1)
        ts, ti = sc.topk(self.k, -1)
        ts = ts / ts.sum(-1, keepdim=True)
        eo = torch.stack([e(x) for e in self.experts], 1)
        sel = torch.gather(eo, 1, ti.unsqueeze(-1).expand(-1, -1, x.size(-1)))
        z = self.proj((ts.unsqueeze(-1) * sel).sum(1))
        f = torch.zeros(self.ne, device=img.device)
        for ki in range(self.k):
            for e in range(self.ne):
                f[e] += (ti[:, ki] == e).float().mean()
        aux = self.ne * ((f / self.k) * sc.mean(0)).sum()
        return self.clf(z), aux

class SoftGateModel(Base):
    def __init__(self, vs, hd=HIDDEN_DIM, ne=4):
        super().__init__(vs, hd)
        D = hd * 2
        self.ne = ne
        self.S = nn.Parameter(torch.randn(D, ne) * 0.01)
        self.experts = nn.ModuleList([FFN(D) for _ in range(ne)])
        self.proj = nn.Linear(D, hd)
        self.clf = Clf(hd)

    def forward(self, img, q):
        hi, ht = self.enc(img, q)
        x = torch.cat([hi, ht], -1)
        phi = torch.softmax(x @ self.S, -1)
        out = sum(phi[:, i:i+1] * self.experts[i](phi[:, i:i+1] * x) for i in range(self.ne))
        return self.clf(self.proj(out)), torch.tensor(0.0, device=img.device)

class CrossAttentionModel(Base):
    def __init__(self, vs, hd=HIDDEN_DIM, nh=8):
        super().__init__(vs, hd)
        self.i2t = nn.MultiheadAttention(hd, nh, dropout=0.1, batch_first=True)
        self.t2i = nn.MultiheadAttention(hd, nh, dropout=0.1, batch_first=True)
        self.n1 = nn.LayerNorm(hd)
        self.n2 = nn.LayerNorm(hd)
        self.fuse = nn.Sequential(nn.Linear(hd * 2, hd), nn.ReLU(), nn.Dropout(0.3))
        self.clf = Clf(hd)

    def forward(self, img, q):
        hi, ht = self.enc(img, q)
        qi, qt = hi.unsqueeze(1), ht.unsqueeze(1)
        ai, _ = self.i2t(qi, qt, qt)
        at, _ = self.t2i(qt, qi, qi)
        zi = self.n1(hi + ai.squeeze(1))
        zt = self.n2(ht + at.squeeze(1))
        return self.clf(self.fuse(torch.cat([zi, zt], -1))), torch.tensor(0.0, device=img.device)

class ModalitySpecificMoE(Base):
    def __init__(self, vs, hd=HIDDEN_DIM, ne=2):
        super().__init__(vs, hd)
        self.ne = ne

        def pool(d):
            experts = nn.ModuleList([
                nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(0.1), nn.Linear(d, d))
                for _ in range(ne)
            ])
            gate = nn.Linear(d, ne)
            return experts, gate

        self.iex, self.ig = pool(hd)
        self.tex, self.tg = pool(hd)
        self.fuse = nn.Sequential(nn.Linear(hd * 2, hd), nn.ReLU(), nn.Dropout(0.3))
        self.clf = Clf(hd)

    def _route(self, x, g, ex):
        w = torch.softmax(g(x), -1)
        return sum(w[:, i:i+1] * ex[i](x) for i in range(self.ne))

    def forward(self, img, q):
        hi, ht = self.enc(img, q)
        return self.clf(self.fuse(torch.cat([self._route(hi, self.ig, self.iex),
                                             self._route(ht, self.tg, self.tex)], -1))), \
               torch.tensor(0.0, device=img.device)

REGISTRY = {
    "baseline":          BaselineConcat,
    "gmu":               GMUModel,
    "sparse_moe":        SparseMoEModel,
    "soft_gate":         SoftGateModel,
    "cross_attn":        CrossAttentionModel,
    "modality_specific": ModalitySpecificMoE,
}

def get_model(name, vocab_size):
    return REGISTRY[name](vocab_size, HIDDEN_DIM)
