"""Kayıtlı skor kartıyla başvuru değerlendirme. API bunu kullanıyor."""
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .veri import KOK, ozellik_ekle

MODEL_KLASORU = Path(os.environ.get("MODEL_KLASORU", KOK / "models"))

AY = {1: "son ay", 2: "2 ay önceki", 3: "3 ay önceki", 4: "4 ay önceki", 5: "5 ay önceki", 6: "6 ay önceki"}
TURKCE = {
    "LIMIT_BAL": "Kredi limiti", "EDUCATION": "Eğitim düzeyi",
    "gecikme_ay_sayisi": "Son 6 ayda gecikmeli ay sayısı", "en_uzun_gecikme": "En uzun gecikme (ay)",
    "odeme_orani": "Faturaların ödenme oranı", "kullanim_orani": "Limit kullanım oranı",
    "fatura_egilimi": "Borcun 6 aylık artışı",
    **{f"PAY_{k}": f"Ödeme durumu ({AY[k]})" for k in AY},
    **{f"PAY_AMT{k}": f"Ödenen tutar ({AY[k]})" for k in AY},
    **{f"BILL_AMT{k}": f"Fatura tutarı ({AY[k]})" for k in AY},
}


class Degerlendirici:
    def __init__(self):
        self.sk = joblib.load(MODEL_KLASORU / "skorkarti.joblib")
        self.bilgi = json.loads((MODEL_KLASORU / "model_bilgisi.json").read_text(encoding="utf-8"))

    def degerlendir(self, basvurular):
        X = ozellik_ekle(pd.DataFrame(basvurular))
        skor = self.sk.skor(X)
        p = self.sk.predict_proba(X)[:, 1]
        nedenler = self.sk.red_nedenleri(X)
        esik = self.bilgi["puan_esigi"]
        sonuc = []
        for s, pi, n in zip(skor, p, nedenler):
            onay = bool(s >= esik)
            sonuc.append({"puan": int(np.round(s)), "temerrut_olasiligi": round(float(pi), 4),
                          "karar": "onay" if onay else "red", "esik": esik,
                          # Onaylanan başvuruda da puanı en çok düşüren özellikler bilgi için dönüyor
                          "puani_dusurenler": [{"ozellik": TURKCE[o], "kaybedilen_puan": k} for o, k in n]})
        return sonuc
