import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.app import app

ORNEKLER = json.loads((Path(__file__).parent / "ornek_basvurular.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def istemci():
    with TestClient(app) as c:
        yield c


def test_saglik(istemci):
    assert istemci.get("/saglik").json()["durum"] == "calisiyor"


def test_guvenli_basvuru_onay(istemci):
    s = istemci.post("/basvuru", json=ORNEKLER["guvenli"]).json()
    assert s["karar"] == "onay"
    assert s["puan"] >= s["esik"]


def test_riskli_basvuru_red_ve_neden(istemci):
    s = istemci.post("/basvuru", json=ORNEKLER["riskli"]).json()
    assert s["karar"] == "red"
    assert s["puan"] < s["esik"]
    assert s["puani_dusurenler"][0]["ozellik"] == "Ödeme durumu (son ay)"


def test_gecikme_puani_dusurur(istemci):
    # Diğer her şey aynıyken son ay 3 ay gecikme puanı düşürmeli
    temiz = istemci.post("/basvuru", json=ORNEKLER["guvenli"]).json()["puan"]
    gecikmeli = istemci.post("/basvuru", json=dict(ORNEKLER["guvenli"], PAY_1=3)).json()["puan"]
    assert gecikmeli < temiz


def test_skor_karti(istemci):
    tablo = istemci.get("/skor-karti").json()
    assert {"ozellik", "aralik", "puan"} <= set(tablo[0])


def test_gecersiz_egitim(istemci):
    assert istemci.post("/basvuru", json=dict(ORNEKLER["guvenli"], EDUCATION=7)).status_code == 422


def test_cinsiyet_girdi_degil(istemci):
    # Fazladan gönderilen korunan özellik sonucu değiştirmemeli
    a = istemci.post("/basvuru", json=ORNEKLER["guvenli"]).json()
    b = istemci.post("/basvuru", json=dict(ORNEKLER["guvenli"], SEX=2, AGE=60)).json()
    assert a == b
