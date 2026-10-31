"""FastAPI 시연 서버 시험 (캐시 모드). 로컬 데이터(SFT held-out 기록·이미지 캐시)가 있어야 하므로 없으면 건너뛴다."""

from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NEED = [ROOT / "data" / "sft" / "m2_salad_f0" / "heldout.jsonl", ROOT / "data" / "index.parquet",
        ROOT / "runs" / "m2_4b_f0_s0" / "final" / "heldout_predictions.jsonl"]
pytestmark = pytest.mark.skipif(not all(p.exists() for p in NEED), reason="로컬 데이터 없음(공개 사본만으로는 SFT 기록이 없다)")


@pytest.fixture(scope="module")
def client():
    from app.api import State, create_app

    return TestClient(create_app(State(live=None)))


def test_meta_and_images(client):
    m = client.get("/api/meta").json()
    assert m["n_images"] == 1568 and m["live"] is False
    assert {c["code"] for c in m["categories"]} == {"breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"}
    assert 0.5 < m["summary"]["ours_group"] < 0.6 and 0.6 < m["summary"]["b1_group"] < 0.7
    rows = client.get("/api/images", params={"category": "pushpins", "label": "good"}).json()
    assert rows and all(r["label"] == "good" and r["id"].startswith("pushpins/") for r in rows)
    assert {r["ours_verdict"] for r in rows} <= {"정답 (무출력)", "오경보"}


def test_predict_and_images(client):
    first = client.get("/api/images", params={"category": "breakfast_box", "label": "logical"}).json()[0]["id"]
    p = client.get(f"/api/predict/{first}").json()
    assert p["id"] == first and p["label"] == "logical" and p["stage"] in ("pass", "judge", "describe")
    assert p["gt"].startswith("이상:") and p["ours"]["verdict"] in ("놓침 (무출력)", "그룹 정답", "그룹 오답")
    assert p["b1"] is not None and len(p["images"]) == 3 and p["live"] is False
    for kind in ("overlay", "inspect", "crop", "ref"):
        r = client.get(f"/api/image/{first}", params={"kind": kind})
        assert r.status_code == 200 and r.content[:8] == b"\x89PNG\r\n\x1a\n", kind
    assert client.get(f"/api/image/{first}", params={"kind": "nope"}).status_code == 400
    assert client.get("/api/predict/no/such/id").status_code == 404


def test_generate_requires_live(client):
    first = client.get("/api/images", params={"category": "pushpins"}).json()[0]["id"]
    assert client.post(f"/api/generate/{first}").status_code == 503


def test_page(client):
    r = client.get("/")
    assert r.status_code == 200 and "LOCO" in r.text and "/api/meta" in r.text
