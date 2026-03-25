"""
Quick smoke-test for the Plant Disease Classifier API.

Usage (server must already be running):
  python api/test_api.py

What it tests:
  1. GET  /health         — server is up, returns expected fields
  2. POST /predict        — real leaf image → valid prediction response
  3. POST /predict (bad)  — non-image file   → 400 error
"""

import sys
from pathlib import Path

try:
    import requests
except ImportError:
    print("requests not installed. Run: pip install requests")
    sys.exit(1)

BASE_URL   = "http://127.0.0.1:8000"
TEST_IMAGE = Path(__file__).parent / "test_leaf.jpg"


# ── helpers ────────────────────────────────────────────────────────────────
def _ok(label: str):
    print(f"  PASS  {label}")

def _fail(label: str, detail: str):
    print(f"  FAIL  {label}")
    print(f"        {detail}")
    sys.exit(1)


# ── tests ──────────────────────────────────────────────────────────────────
def test_health():
    r = requests.get(f"{BASE_URL}/health", timeout=10)
    if r.status_code != 200:
        _fail("/health", f"status {r.status_code}")
    data = r.json()
    assert data.get("status") == "ok",      f"unexpected: {data}"
    assert data.get("model")  == "ResNet50", f"unexpected: {data}"
    assert data.get("classes") == 38,        f"unexpected: {data}"
    _ok("/health")


def test_predict():
    if not TEST_IMAGE.exists():
        _fail("/predict", f"test image not found: {TEST_IMAGE}\n"
                          "        Place a leaf image at api/test_leaf.jpg")

    with open(TEST_IMAGE, "rb") as f:
        r = requests.post(
            f"{BASE_URL}/predict",
            files={"file": ("test_leaf.jpg", f, "image/jpeg")},
            timeout=30,
        )

    if r.status_code != 200:
        _fail("/predict", f"status {r.status_code} — {r.text}")

    data = r.json()

    required = {"predicted_class", "confidence", "top3", "inference_time_ms"}
    missing = required - data.keys()
    if missing:
        _fail("/predict", f"missing keys: {missing}")

    assert isinstance(data["confidence"], float),        "confidence not float"
    assert 0.0 <= data["confidence"] <= 1.0,             "confidence out of range"
    assert len(data["top3"]) == 3,                       "top3 must have 3 entries"
    assert data["top3"][0]["class"] == data["predicted_class"], "top3[0] mismatch"

    _ok("/predict")
    print(f"        Predicted : {data['predicted_class']}")
    print(f"        Confidence: {data['confidence']:.2%}")
    print(f"        Top-3     :")
    for item in data["top3"]:
        print(f"          {item['class']:<55} {item['confidence']:.4f}")
    print(f"        Inference : {data['inference_time_ms']} ms")


def test_invalid_file():
    r = requests.post(
        f"{BASE_URL}/predict",
        files={"file": ("document.txt", b"this is not an image", "text/plain")},
        timeout=10,
    )
    if r.status_code != 400:
        _fail("/predict (invalid)", f"expected 400, got {r.status_code}")
    _ok("/predict (non-image → 400)")


# ── main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Plant Disease Classifier — API tests")
    print(f"Target : {BASE_URL}\n")

    try:
        test_health()
        test_predict()
        test_invalid_file()
    except requests.exceptions.ConnectionError:
        print(f"\n  ERROR  Cannot connect to {BASE_URL}")
        print("         Start the server first:")
        print("           uvicorn api.app:app --reload")
        sys.exit(1)

    print("\nAll tests passed.")
