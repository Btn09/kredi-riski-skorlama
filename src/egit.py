"""Skor kartı ve karşılaştırma modellerini eğitir; eşik, strateji tablosu ve adillik analizini üretir.

Çalıştırma:  python -m src.egit
"""
import json

import joblib
import matplotlib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .metrikler import kar, ozet
from .skorkarti import SkorKarti
from .veri import KOK, KORUNAN, OZELLIKLER, bol, ozellik_ekle, veri_oku

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RAPOR = KOK / "reports"
GRAFIK = RAPOR / "figures"
MODELLER = KOK / "models"


def lgbm():
    return LGBMClassifier(n_estimators=500, learning_rate=0.02, num_leaves=15, min_child_samples=100,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5,
                          random_state=42, verbose=-1)


def en_iyi_esik(y, olasilik):
    """Kârı en yüksek yapan temerrüt olasılığı eşiği: bunun altındakiler onaylanır."""
    esikler = np.quantile(olasilik, np.linspace(0.05, 1.0, 96))
    karlar = [kar(y, olasilik <= e) for e in esikler]
    return float(esikler[int(np.argmax(karlar))])


def adillik(df, onay):
    """Korunan gruplara göre onay oranları. 'Fırsat eşitliği': gerçekten ödeyecek (iyi) müşterilerin onaylanma oranı."""
    gruplar = {
        "Cinsiyet": df["SEX"].map({1: "Erkek", 2: "Kadın"}),
        "Yaş": pd.cut(df["AGE"], [0, 29, 44, 200], labels=["<30", "30-44", "45+"]).astype(str),
        "Medeni hal": df["MARRIAGE"].map({1: "Evli", 2: "Bekar", 3: "Diğer"}),
    }
    y = df["temerrut"].values
    satirlar = []
    for boyut, g in gruplar.items():
        for ad in sorted(g.unique()):
            m = (g == ad).values
            satirlar.append({"boyut": boyut, "grup": ad, "adet": int(m.sum()),
                             "gercek_temerrut": round(float(y[m].mean()), 3),
                             "onay_orani": round(float(onay[m].mean()), 3),
                             "iyilerde_onay": round(float(onay[m & (y == 0)].mean()), 3),
                             "onaylananda_temerrut": round(float(y[m & onay].mean()), 3)})
    t = pd.DataFrame(satirlar)
    # En düşük / en yüksek onay oranı ("4/5 kuralı": 0,8'in altı dikkat ister). 100 kişiden küçük gruplar
    # çok gürültülü olduğu için orana katılmıyor.
    buyuk = t[t.adet >= 100]
    oranlar = {b: round(float(g.onay_orani.min() / g.onay_orani.max()), 3) for b, g in buyuk.groupby("boyut")}
    return t, oranlar


def main():
    for k in (GRAFIK, MODELLER):
        k.mkdir(parents=True, exist_ok=True)
    df = ozellik_ekle(veri_oku())
    egitim, dogrulama, test = bol(df)
    y_e, y_d, y_t = egitim.temerrut.values, dogrulama.temerrut.values, test.temerrut.values
    print(f"Eğitim {len(y_e)} | Doğrulama {len(y_d)} | Test {len(y_t)} | temerrüt %{y_e.mean() * 100:.1f}")

    modeller = {
        "Skor kartı (WoE + lojistik regresyon)": (SkorKarti(), OZELLIKLER),
        "Lojistik regresyon (ham özellikler)": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
                                                OZELLIKLER),
        "LightGBM": (lgbm(), OZELLIKLER),
        "LightGBM + korunan özellikler": (lgbm(), OZELLIKLER + KORUNAN),
    }
    satirlar, olasiliklar = [], {}
    for ad, (model, ozellikler) in modeller.items():
        model.fit(egitim[ozellikler], y_e)
        p_d = model.predict_proba(dogrulama[ozellikler])[:, 1]
        p_t = model.predict_proba(test[ozellikler])[:, 1]
        olasiliklar[ad] = (p_d, p_t)
        esik = en_iyi_esik(y_d, p_d)
        onay = p_t <= esik
        satir = {"model": ad, **{f"dogrulama_{k}": v for k, v in ozet(y_d, p_d).items()},
                 **{f"test_{k}": v for k, v in ozet(y_t, p_t).items()},
                 "test_onay_orani": onay.mean(), "test_onaylananda_temerrut": y_t[onay].mean(),
                 "test_kar": kar(y_t, onay)}
        satirlar.append(satir)
        print(f"{ad:40s} test AUC {satir['test_auc']:.3f} | Gini {satir['test_gini']:.3f} | KS {satir['test_ks']:.3f}"
              f" | onay %{satir['test_onay_orani'] * 100:.0f} | onaylananda temerrüt "
              f"%{satir['test_onaylananda_temerrut'] * 100:.1f} | kâr {satir['test_kar']:.0f}")
    pd.DataFrame(satirlar).to_csv(RAPOR / "model_karsilastirma.csv", index=False, float_format="%.4f")

    # Servis edilen model: skor kartı
    sk = modeller["Skor kartı (WoE + lojistik regresyon)"][0]
    p_d, p_t = olasiliklar["Skor kartı (WoE + lojistik regresyon)"]
    esik = en_iyi_esik(y_d, p_d)
    # Olasılık eşiğini puan eşiğine çevir (puan, olasılığın tek yönlü bir fonksiyonu)
    puan_esigi = float(np.round(600 - 20 / np.log(2) * (np.log(esik / (1 - esik)) + np.log(50)), 1))
    skor_t = sk.skor(test[OZELLIKLER])
    onay = skor_t >= puan_esigi
    print(f"\nPuan eşiği: {puan_esigi} | herkesi onaylasaydık kâr {kar(y_t, np.ones(len(y_t), bool)):.0f}")

    sk.puan_tablosu.round(2).to_csv(RAPOR / "puan_tablosu.csv", index=False)
    joblib.dump(sk, MODELLER / "skorkarti.joblib")
    with open(MODELLER / "model_bilgisi.json", "w", encoding="utf-8") as f:
        json.dump({"puan_esigi": puan_esigi, "olasilik_esigi": round(esik, 4), "ozellikler": sk.secilen,
                   "kar_varsayimi": "iyi müşteri +1, temerrüt -5"}, f, ensure_ascii=False, indent=2)

    # Puan bantları
    bantlar = pd.cut(skor_t, [-np.inf, 480, 500, 520, 540, 560, np.inf])
    bant_tablosu = (pd.DataFrame({"bant": bantlar.astype(str), "temerrut": y_t})
                    .groupby("bant", sort=False).agg(adet=("temerrut", "size"), temerrut_orani=("temerrut", "mean"))
                    .reindex([str(b) for b in bantlar.categories]).reset_index())
    bant_tablosu.to_csv(RAPOR / "puan_bantlari.csv", index=False, float_format="%.3f")

    # Adillik: korunan özellikleri kullanmayan skor kartı ile kullanan LightGBM
    adil_sk, oran_sk = adillik(test, onay)
    p_lk_d, p_lk_t = olasiliklar["LightGBM + korunan özellikler"]
    adil_lk, oran_lk = adillik(test, p_lk_t <= en_iyi_esik(y_d, p_lk_d))
    adil_sk.to_csv(RAPOR / "adillik_skorkarti.csv", index=False)
    adil_lk.to_csv(RAPOR / "adillik_lgbm_korunan.csv", index=False)
    print(adil_sk.to_string(index=False))
    print("Onay oranı en düşük / en yüksek grup -> skor kartı:", oran_sk, "| LightGBM+korunan:", oran_lk)

    sonuc = {"satirlar": satirlar, "puan_esigi": puan_esigi, "skorkarti_ozellikler": sk.secilen,
             "iv": {o: round(v, 4) for o, v in sorted(sk.iv.items(), key=lambda x: -x[1])},
             "onay_orani_dengesi": {"skor_karti": oran_sk, "lgbm_korunan": oran_lk}}
    with open(RAPOR / "sonuclar.json", "w", encoding="utf-8") as f:
        json.dump(sonuc, f, ensure_ascii=False, indent=2, default=float)

    grafikler(sk, skor_t, y_t, puan_esigi, olasiliklar, y_t)


def grafikler(sk, skor_t, y_t, puan_esigi, olasiliklar, y):
    iv = pd.Series(sk.iv).sort_values()
    fig, eksen = plt.subplots(figsize=(7, 6))
    eksen.barh(iv.index, iv.values, color=["tab:blue" if o in sk.secilen else "lightgray" for o in iv.index])
    eksen.axvline(0.02, color="tab:red", ls="--", lw=1)
    eksen.set_xlabel("Bilgi değeri (IV)")
    eksen.set_title("Özelliklerin bilgi değeri (mavi: skor kartına giren)")
    plt.tight_layout()
    plt.savefig(GRAFIK / "bilgi_degeri.png", dpi=140)
    plt.close(fig)

    fig, eksen = plt.subplots(figsize=(8, 4.2))
    aralik = np.linspace(skor_t.min(), skor_t.max(), 40)
    eksen.hist(skor_t[y_t == 0], aralik, alpha=0.6, density=True, label="Ödeyen müşteriler")
    eksen.hist(skor_t[y_t == 1], aralik, alpha=0.6, density=True, label="Temerrüde düşenler")
    eksen.axvline(puan_esigi, color="black", ls="--", label=f"Onay eşiği ({puan_esigi:.0f})")
    eksen.set_xlabel("Kredi puanı")
    eksen.set_ylabel("Yoğunluk")
    eksen.set_title("Test setinde puan dağılımı")
    eksen.legend()
    plt.tight_layout()
    plt.savefig(GRAFIK / "puan_dagilimi.png", dpi=140)
    plt.close(fig)

    fig, eksen = plt.subplots(figsize=(7, 4.5))
    for ad, (_, p) in olasiliklar.items():
        sira = np.argsort(p)
        onay_orani = np.arange(1, len(p) + 1) / len(p)
        kotu = np.cumsum(y[sira]) / np.arange(1, len(p) + 1)
        eksen.plot(onay_orani * 100, kotu * 100, label=ad)
    eksen.set_xlabel("Onay oranı (%) (en güvenliden başlayarak)")
    eksen.set_ylabel("Onaylananlarda temerrüt (%)")
    eksen.set_title("Onay oranı ile risk arasındaki denge (test)")
    eksen.grid(alpha=0.3)
    eksen.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(GRAFIK / "onay_risk.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
