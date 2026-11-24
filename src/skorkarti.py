"""Klasik kredi skor kartı: WoE (Weight of Evidence) dönüşümü + lojistik regresyon + puan ölçekleme.

Bankalarda hâlâ en yaygın yöntem, çünkü her özelliğin her aralığının kaç puan getirdiği tek bir tabloda
görülebiliyor ve bir başvurunun neden reddedildiği doğrudan bu tablodan söylenebiliyor.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# 50:1 iyi/kötü oranı 600 puan, oran her ikiye katlandığında +20 puan (PDO)
TABAN_PUAN, TABAN_ORAN, PDO = 600, 50, 20
FAKTOR = PDO / np.log(2)
OFSET = TABAN_PUAN - FAKTOR * np.log(TABAN_ORAN)


def ilk_sinirlar(x, en_fazla=10):
    """Değer sayısı azsa her değer ayrı aralık, değilse eşit frekanslı aralıklar."""
    tekil = np.unique(x)
    if len(tekil) <= en_fazla:
        return list((tekil[:-1] + tekil[1:]) / 2)
    return list(np.unique(np.quantile(x, np.linspace(0, 1, en_fazla + 1)[1:-1])))


def woe_tablosu(x, y, sinirlar, yumusatma=0.5):
    kova = np.searchsorted(sinirlar, x, side="right")
    iyi_top, kotu_top = (y == 0).sum(), (y == 1).sum()
    satirlar = []
    for b in range(len(sinirlar) + 1):
        m = kova == b
        iyi, kotu = ((y == 0) & m).sum(), ((y == 1) & m).sum()
        p_iyi, p_kotu = (iyi + yumusatma) / iyi_top, (kotu + yumusatma) / kotu_top
        satirlar.append({"adet": int(m.sum()), "kotu_orani": kotu / max(m.sum(), 1),
                         "woe": np.log(p_iyi / p_kotu), "iv": (p_iyi - p_kotu) * np.log(p_iyi / p_kotu)})
    return pd.DataFrame(satirlar)


def monoton_birlestir(x, y, sinirlar, en_az_oran=0.05):
    """WoE aralıklar boyunca tek yönde değişene ve her aralık yeterince büyük olana kadar komşu aralıkları birleştirir.
    Monotonluk hem aşırı öğrenmeyi azaltıyor hem de puanların mantıklı olmasını sağlıyor
    (ör. gecikme arttıkça puan hiç artmamalı)."""
    sinirlar = list(sinirlar)
    while sinirlar:
        t = woe_tablosu(x, y, sinirlar)
        kucuk = np.flatnonzero(t["adet"].values < en_az_oran * len(x))
        fark = np.diff(t["woe"].values)
        yon = np.sign(np.sum(fark))
        ters = np.flatnonzero(np.sign(fark) == -yon) if yon != 0 else np.arange(len(fark))
        if len(kucuk) == 0 and len(ters) == 0:
            break
        # Önce küçük aralıklar, sonra yönü bozan aralıklar: iki komşudan WoE'si en yakın olanla birleştir
        i = kucuk[0] if len(kucuk) else ters[0] + 1
        if i == 0:
            sinirlar.pop(0)
        elif i >= len(sinirlar):
            sinirlar.pop(-1)
        else:
            w = t["woe"].values
            sinirlar.pop(i - 1 if abs(w[i] - w[i - 1]) <= abs(w[i] - w[i + 1]) else i)
    return sinirlar


class SkorKarti:
    def __init__(self, iv_esigi=0.02, korelasyon_esigi=0.8, C=1.0):
        self.iv_esigi = iv_esigi
        self.korelasyon_esigi = korelasyon_esigi
        self.C = C

    def fit(self, X, y):
        y = np.asarray(y)
        self.sinirlar, self.tablolar, self.iv = {}, {}, {}
        for o in X.columns:
            s = monoton_birlestir(X[o].values, y, ilk_sinirlar(X[o].values))
            self.sinirlar[o] = s
            self.tablolar[o] = woe_tablosu(X[o].values, y, s)
            self.iv[o] = float(self.tablolar[o]["iv"].sum())

        # Bilgi değeri düşük olanları at, birbirine çok benzeyenlerden IV'si yüksek olanı tut
        adaylar = sorted([o for o in X.columns if self.iv[o] >= self.iv_esigi], key=lambda o: -self.iv[o])
        W = self.woe(X, adaylar)
        self.secilen = []
        for o in adaylar:
            if all(abs(np.corrcoef(W[o], W[s])[0, 1]) < self.korelasyon_esigi for s in self.secilen):
                self.secilen.append(o)

        # WoE yüksek = daha güvenli, yani tüm katsayılar negatif olmalı. Pozitif çıkan (birbirine çok benzeyen
        # değişkenlerin yarattığı) ters işaretli değişkeni çıkarıp yeniden eğitiyorum, puanlar mantıksız olmasın.
        while True:
            self.lr = LogisticRegression(C=self.C, max_iter=1000).fit(self.woe(X, self.secilen), y)
            ters = [o for o, b in zip(self.secilen, self.lr.coef_[0]) if b >= 0]
            if not ters:
                break
            self.secilen.remove(min(ters, key=lambda o: self.iv[o]))
        self.puan_tablosu = self._puanlar()
        return self

    def woe(self, X, ozellikler=None):
        ozellikler = ozellikler or self.secilen
        return pd.DataFrame({o: self.tablolar[o]["woe"].values[np.searchsorted(self.sinirlar[o], X[o].values,
                                                                                   side="right")]
                             for o in ozellikler}, index=X.index)

    def predict_proba(self, X):
        p = self.lr.predict_proba(self.woe(X))[:, 1]
        return np.column_stack([1 - p, p])

    def _puanlar(self):
        """Her özelliğin her aralığına puan. Toplam puan = OFSET - FAKTOR * logit(temerrüt olasılığı)."""
        n = len(self.secilen)
        satirlar = []
        for beta, o in zip(self.lr.coef_[0], self.secilen):
            s = self.sinirlar[o]
            for b, woe in enumerate(self.tablolar[o]["woe"]):
                alt = -np.inf if b == 0 else s[b - 1]
                ust = np.inf if b == len(s) else s[b]
                puan = OFSET / n - FAKTOR * (beta * woe + self.lr.intercept_[0] / n)
                satirlar.append({"ozellik": o, "aralik": b, "alt": alt, "ust": ust,
                                 "kotu_orani": self.tablolar[o]["kotu_orani"][b], "puan": puan})
        return pd.DataFrame(satirlar)

    def skor(self, X):
        return OFSET - FAKTOR * self.lr.decision_function(self.woe(X))

    def ozellik_puanlari(self, X):
        """(n, seçilen özellik) puan matrisi."""
        return pd.DataFrame({o: self.puan_tablosu[self.puan_tablosu.ozellik == o]["puan"].values[
            np.searchsorted(self.sinirlar[o], X[o].values, side="right")] for o in self.secilen}, index=X.index)

    def red_nedenleri(self, X, adet=3):
        """Her başvuru için, alabileceği en yüksek puana göre en çok puan kaybettiği özellikler."""
        puan = self.ozellik_puanlari(X)
        en_yuksek = self.puan_tablosu.groupby("ozellik")["puan"].max()[self.secilen]
        kayip = en_yuksek.values - puan.values
        sonuc = []
        for satir in kayip:
            sira = np.argsort(-satir)[:adet]
            sonuc.append([(self.secilen[j], round(float(satir[j]), 1)) for j in sira if satir[j] > 0.5])
        return sonuc
