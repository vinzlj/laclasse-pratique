"""Mini-GPT réutilisable, inspiré de nanoGPT (Karpathy, https://github.com/karpathy/nanoGPT).

Transformer décodeur seul (Radford et al., 2018/2019), pré-normalisation (LayerNorm
avant l'attention et le MLP), attention causale multi-têtes, poids d'entrée et de
sortie partagés. Volontairement court et commenté : les chapitres 20, 27, 28 et 38
s'en servent (entraînement, loi d'échelle, inférence, interprétabilité).

Taille par défaut (4 couches, 4 têtes, 256 dimensions, contexte 128) : ~3,2 M de
paramètres, val loss ≈ 1,5 sur TinyShakespeare en ~2 min 20 sur M1 Pro (mesure
de la phase 1).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ConfigGPT:
    taille_vocab: int = 65  # nombre de caractères distincts de TinyShakespeare
    taille_bloc: int = 128  # longueur maximale du contexte (T)
    n_couches: int = 4
    n_tetes: int = 4
    n_embd: int = 256  # dimension des représentations (d_model)
    dropout: float = 0.1
    biais: bool = False  # biais dans les Linear et LayerNorm (nanoGPT : False, un peu mieux)


class AttentionCausale(nn.Module):
    """Attention multi-têtes masquée : chaque position ne voit que le passé."""

    def __init__(self, cfg: ConfigGPT):
        super().__init__()
        assert cfg.n_embd % cfg.n_tetes == 0, "n_embd doit être divisible par n_tetes"
        self.n_tetes = cfg.n_tetes
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.biais)  # Q, K, V d'un coup
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.biais)
        self.dropout = cfg.dropout
        self.drop_residuel = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) → (B, têtes, T, C / têtes)
        q, k, v = (t.view(B, T, self.n_tetes, C // self.n_tetes).transpose(1, 2) for t in (q, k, v))
        p_drop = self.dropout if self.training else 0.0
        if p_drop > 0 and x.device.type == "mps":
            # Le noyau fusionné ne gère pas le dropout sur MPS (NotImplementedError,
            # torch 2.14.1) : calcul explicite, identique en espérance.
            att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
            masque = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
            att = att.masked_fill(~masque, float("-inf")).softmax(dim=-1)
            y = F.dropout(att, p_drop, training=True) @ v
        else:
            # softmax(QKᵀ / √d) V avec masque causal, version fusionnée de PyTorch
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=p_drop)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.drop_residuel(self.proj(y))


class MLP(nn.Module):
    def __init__(self, cfg: ConfigGPT):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.biais)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.biais)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Bloc(nn.Module):
    """Bloc Transformer : x + Attention(LN(x)), puis x + MLP(LN(x))."""

    def __init__(self, cfg: ConfigGPT):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd, bias=cfg.biais)
        self.attn = AttentionCausale(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd, bias=cfg.biais)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class MiniGPT(nn.Module):
    def __init__(self, cfg: ConfigGPT):
        super().__init__()
        self.cfg = cfg
        self.emb_tokens = nn.Embedding(cfg.taille_vocab, cfg.n_embd)
        self.emb_positions = nn.Embedding(cfg.taille_bloc, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocs = nn.ModuleList([Bloc(cfg) for _ in range(cfg.n_couches)])
        self.ln_final = nn.LayerNorm(cfg.n_embd, bias=cfg.biais)
        self.tete = nn.Linear(cfg.n_embd, cfg.taille_vocab, bias=False)
        self.tete.weight = self.emb_tokens.weight  # partage des poids (weight tying)
        self.apply(self._init_poids)
        # Initialisation réduite des projections résiduelles (GPT-2)
        for nom, p in self.named_parameters():
            if nom.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_couches))

    @staticmethod
    def _init_poids(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def nb_parametres(self) -> int:
        """Nombre de paramètres, sans les plongements de position (convention nanoGPT)."""
        return sum(p.numel() for p in self.parameters()) - self.emb_positions.weight.numel()

    def forward(self, idx: torch.Tensor, cibles: torch.Tensor | None = None):
        """idx : (B, T) entiers. Renvoie (logits (B, T, V), loss ou None)."""
        _, T = idx.shape  # (taille du lot, longueur)
        if T > self.cfg.taille_bloc:
            raise ValueError(f"Séquence de {T} tokens > taille_bloc = {self.cfg.taille_bloc}")
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.emb_tokens(idx) + self.emb_positions(pos))
        for bloc in self.blocs:
            x = bloc(x)
        logits = self.tete(self.ln_final(x))
        loss = None
        if cibles is not None:
            # Entropie croisée sur le token suivant (ch. 01 et 20)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), cibles.view(-1))
        return logits, loss

    @torch.no_grad()
    def generer(self, idx: torch.Tensor, n_nouveaux: int, temperature: float = 1.0,
                top_k: int | None = None, generateur: torch.Generator | None = None) -> torch.Tensor:
        """Génération autorégressive : on échantillonne un token, on l'ajoute, on recommence.

        Le modèle passe en mode évaluation (pas de dropout) le temps de la génération.
        """
        etait_en_train = self.training
        self.eval()
        for _ in range(n_nouveaux):
            contexte = idx[:, -self.cfg.taille_bloc :]
            logits, _ = self(contexte)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                seuil = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                logits = logits.masked_fill(logits < seuil, float("-inf"))
            probas = F.softmax(logits, dim=-1)
            # multinomial sur CPU : reproductible avec un générateur CPU, quel que soit le device
            suivant = torch.multinomial(probas.cpu(), 1, generator=generateur).to(idx.device)
            idx = torch.cat([idx, suivant], dim=1)
        self.train(etait_en_train)
        return idx

    def sauvegarder(self, chemin: str | Path, **extra) -> None:
        """Enregistre config + poids (+ métadonnées libres : vocabulaire, pas, loss…)."""
        chemin = Path(chemin)
        chemin.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": asdict(self.cfg), "etat": self.state_dict(), **extra}, chemin)

    @classmethod
    def charger(cls, chemin: str | Path, device: str | torch.device = "cpu"):
        """Recharge un modèle sauvegardé. Renvoie (modèle, dictionnaire complet)."""
        donnees = torch.load(chemin, map_location=device, weights_only=False)
        modele = cls(ConfigGPT(**donnees["config"]))
        modele.load_state_dict(donnees["etat"])
        return modele.to(device), donnees


class TokeniseurCaracteres:
    """Tokeniseur au niveau du caractère : un identifiant par caractère distinct."""

    def __init__(self, texte: str):
        self.caracteres = sorted(set(texte))
        self.vers_id = {c: i for i, c in enumerate(self.caracteres)}

    @property
    def taille_vocab(self) -> int:
        return len(self.caracteres)

    def encoder(self, texte: str) -> list[int]:
        return [self.vers_id[c] for c in texte]

    def decoder(self, ids) -> str:
        return "".join(self.caracteres[int(i)] for i in ids)


def lot_aleatoire(donnees: torch.Tensor, taille_bloc: int, taille_lot: int,
                  device: torch.device | str, generateur: torch.Generator | None = None):
    """Tire `taille_lot` fenêtres au hasard dans `donnees` (1D, int64). Renvoie (x, y) sur device."""
    debuts = torch.randint(len(donnees) - taille_bloc - 1, (taille_lot,), generator=generateur)
    x = torch.stack([donnees[i : i + taille_bloc] for i in debuts])
    y = torch.stack([donnees[i + 1 : i + 1 + taille_bloc] for i in debuts])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimer_loss(modele: MiniGPT, donnees: torch.Tensor, taille_bloc: int, taille_lot: int,
                 device, n_lots: int = 20, generateur: torch.Generator | None = None) -> float:
    """Loss moyenne sur `n_lots` lots aléatoires, modèle en mode évaluation."""
    etait_en_train = modele.training
    modele.eval()
    pertes = [modele(*lot_aleatoire(donnees, taille_bloc, taille_lot, device, generateur))[1].item()
              for _ in range(n_lots)]
    modele.train(etait_en_train)
    return sum(pertes) / len(pertes)


def entrainer(modele: MiniGPT, donnees: torch.Tensor, n_pas: int, device,
              taille_lot: int = 64, lr: float = 1e-3, log_tous: int = 0,
              generateur: torch.Generator | None = None) -> list[float]:
    """Boucle d'entraînement minimale (AdamW, lr constant). Renvoie la loss à chaque pas."""
    optim = torch.optim.AdamW(modele.parameters(), lr=lr, weight_decay=0.1)
    modele.train()
    historique = []
    for pas in range(n_pas):
        x, y = lot_aleatoire(donnees, modele.cfg.taille_bloc, taille_lot, device, generateur)
        _, loss = modele(x, y)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(modele.parameters(), 1.0)
        optim.step()
        historique.append(loss.item())
        if log_tous and (pas % log_tous == 0 or pas == n_pas - 1):
            print(f"pas {pas:5d}  loss {historique[-1]:.4f}")
    return historique
