"""Kredi başvurusu değerlendirme servisi.

Çalıştırma:  uvicorn api.app:app
Dokümantasyon: http://127.0.0.1:8000/docs

Cinsiyet, yaş ve medeni hal bilinçli olarak girdi değil: model bunları kullanmıyor.
"""
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field, create_model

from src.cikarim import Degerlendirici

OdemeDurumu = int
Basvuru = create_model(
    "Basvuru",
    LIMIT_BAL=(float, Field(gt=0, description="Kredi limiti")),
    EDUCATION=(Literal[1, 2, 3, 4], Field(description="1: lisansüstü, 2: üniversite, 3: lise, 4: diğer")),
    **{f"PAY_{k}": (OdemeDurumu, Field(ge=-2, le=9, description="-1 tam ödendi, 0 asgari, 1-9 ay gecikme"))
       for k in range(1, 7)},
    **{f"BILL_AMT{k}": (float, ...) for k in range(1, 7)},
    **{f"PAY_AMT{k}": (float, Field(ge=0)) for k in range(1, 7)},
)


class Neden(BaseModel):
    ozellik: str
    kaybedilen_puan: float


class Karar(BaseModel):
    puan: int
    temerrut_olasiligi: float
    karar: Literal["onay", "red"]
    esik: float
    puani_dusurenler: list[Neden]


durum = {}


@asynccontextmanager
async def lifespan(app):
    durum["d"] = Degerlendirici()
    yield
    durum.clear()


app = FastAPI(title="Kredi Riski Skorlama", lifespan=lifespan)


@app.get("/saglik")
def saglik():
    return {"durum": "calisiyor", "puan_esigi": durum["d"].bilgi["puan_esigi"],
            "ozellik_sayisi": len(durum["d"].bilgi["ozellikler"])}


@app.post("/basvuru", response_model=Karar)
def basvuru(b: Basvuru):
    return durum["d"].degerlendir([b.model_dump()])[0]


@app.get("/skor-karti")
def skor_karti():
    """Her özelliğin her aralığının puanı; kararların şeffaf olması için."""
    t = durum["d"].sk.puan_tablosu.round(2).replace([float("inf"), float("-inf")], None)
    return t.to_dict(orient="records")
