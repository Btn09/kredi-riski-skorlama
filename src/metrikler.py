import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve


def ks(y, p):
    """Kolmogorov-Smirnov: iyi ve kötü müşterilerin puan dağılımları arasındaki en büyük fark (bankacılıkta standart)."""
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def ozet(y, p):
    auc = roc_auc_score(y, p)
    return {"auc": auc, "gini": 2 * auc - 1, "ks": ks(y, p), "brier": brier_score_loss(y, p)}


def kar(y, onay, iyi_kazanc=1.0, kotu_kayip=5.0):
    """Onaylanan iyi müşteri +1 birim kazandırıyor, onaylanan kötü müşteri -5 birim kaybettiriyor
    (temerrütte kaybedilen anapara, bir müşteriden kazanılan faizden kat kat büyük)."""
    y, onay = np.asarray(y), np.asarray(onay)
    return float(((y == 0) & onay).sum() * iyi_kazanc - ((y == 1) & onay).sum() * kotu_kayip)
