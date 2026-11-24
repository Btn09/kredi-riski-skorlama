"""UCI 'Default of Credit Card Clients' (Tayvan, 2005) verisi.

Hedef: müşterinin bir sonraki ay kredi kartı borcunu ödememesi (temerrüt).
PAY_k: k ay önceki ödeme durumu (-2: kullanım yok, -1: tam ödendi, 0: asgari ödendi, 1..8: o kadar ay gecikme).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

KOK = Path(__file__).resolve().parent.parent
VERI_DOSYASI = KOK / "data" / "indirme" / "default of credit card clients.xls"

ODEME = [f"PAY_{k}" for k in range(1, 7)]
FATURA = [f"BILL_AMT{k}" for k in range(1, 7)]
ODENEN = [f"PAY_AMT{k}" for k in range(1, 7)]
HAM = ["LIMIT_BAL", "EDUCATION"] + ODEME + FATURA + ODENEN
TURETILEN = ["kullanim_orani", "odeme_orani", "gecikme_ay_sayisi", "en_uzun_gecikme", "fatura_egilimi"]
OZELLIKLER = HAM + TURETILEN

# Modelde kullanılmayan, sadece adillik kontrolünde kullanılan korunan özellikler
KORUNAN = ["SEX", "MARRIAGE", "AGE"]


def veri_oku(yol=VERI_DOSYASI):
    df = pd.read_excel(yol, header=1).rename(columns={"PAY_0": "PAY_1", "default payment next month": "temerrut"})
    # Dokümanda tanımlı olmayan kodlar "diğer" grubuna
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
    return df


def ozellik_ekle(df):
    df = df.copy()
    df["kullanim_orani"] = (df["BILL_AMT1"] / df["LIMIT_BAL"]).clip(-1, 3)
    # PAY_AMTk, bir önceki ayın faturası BILL_AMT(k+1) için yapılan ödeme
    odenen = df[ODENEN[:5]].sum(axis=1)
    fatura = df[FATURA[1:]].clip(lower=0).sum(axis=1)
    df["odeme_orani"] = np.where(fatura > 0, odenen / fatura.replace(0, 1), 1.0).clip(0, 2)
    df["gecikme_ay_sayisi"] = (df[ODEME] > 0).sum(axis=1)
    df["en_uzun_gecikme"] = df[ODEME].max(axis=1).clip(lower=0)
    df["fatura_egilimi"] = ((df["BILL_AMT1"] - df["BILL_AMT6"]) / df["LIMIT_BAL"]).clip(-2, 2)
    return df


def bol(df, seed=42):
    egitim, kalan = train_test_split(df, test_size=0.4, stratify=df["temerrut"], random_state=seed)
    dogrulama, test = train_test_split(kalan, test_size=0.5, stratify=kalan["temerrut"], random_state=seed)
    return egitim, dogrulama, test
