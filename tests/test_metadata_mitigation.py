import numpy as np
from PIL import Image

from src.metadata.metadata_check import check_metadata
from src.mitigation.provenance import ProvenanceLedger
from src.mitigation.reverse_search import ImageIndex
from src.mitigation.signature import generate_keypair, sign_file, verify_file
from src.mitigation.verify import verify_content
from src.mitigation.watermark import (detect_content_credentials, embed_lsb,
                                      extract_lsb, verify_watermark)


def make_img(path, seed=0, size=(128, 128)):
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 255, (*size, 3), dtype=np.uint8)).save(path)
    return path


def test_metadata_stripped_image_flagged(tmp_path):
    r = check_metadata(make_img(tmp_path / "a.png"))
    assert r["metadata_score"] > 0 and r["flags"]


def test_metadata_ai_hint_detected(tmp_path):
    from PIL import PngImagePlugin
    info = PngImagePlugin.PngInfo()
    info.add_text("parameters", "a cat, Steps: 20, Sampler: Euler, Stable Diffusion")
    Image.new("RGB", (32, 32)).save(tmp_path / "sd.png", pnginfo=info)
    r = check_metadata(tmp_path / "sd.png")
    assert any("AI-generation" in f for f in r["flags"])


def test_ledger_chain_and_tamper(tmp_path):
    f = make_img(tmp_path / "a.png")
    led = ProvenanceLedger(tmp_path / "ledger.jsonl")
    led.add(f, verdict="real")
    led.add(f, verdict="real")
    assert led.verify_chain() and len(led.lookup(f)) == 2
    text = (tmp_path / "ledger.jsonl").read_text().replace("real", "fake", 1)
    (tmp_path / "ledger.jsonl").write_text(text)
    assert not led.verify_chain()


def test_signature_roundtrip_and_tamper(tmp_path):
    priv, pub = tmp_path / "p.pem", tmp_path / "pub.pem"
    generate_keypair(priv, pub)
    f = make_img(tmp_path / "a.png")
    sig = sign_file(f, priv)
    assert verify_file(f, sig, pub)
    make_img(f, seed=1)  # modify file
    assert not verify_file(f, sig, pub)


def test_lsb_watermark(tmp_path):
    src = make_img(tmp_path / "a.png")
    embed_lsb(src, tmp_path / "wm.png", "PICT-2026")
    assert extract_lsb(tmp_path / "wm.png") == "PICT-2026"
    assert verify_watermark(tmp_path / "wm.png", "PICT-2026")
    assert not verify_watermark(src, "PICT-2026")


def test_content_credentials(tmp_path):
    f = tmp_path / "x.png"
    f.write_bytes(b"\x89PNG....trainedAlgorithmicMedia....")
    assert detect_content_credentials(f)["ai_generated_marker"]


def test_reverse_search_finds_resized_copy(tmp_path):
    orig = make_img(tmp_path / "orig.png", size=(256, 256))
    idx = ImageIndex(tmp_path / "idx.json")
    idx.add(orig, "original")
    Image.open(orig).resize((128, 128)).save(tmp_path / "small.jpg", quality=70)
    hits = idx.search(tmp_path / "small.jpg")
    assert hits and hits[0]["label"] == "original"
    other = make_img(tmp_path / "other.png", seed=9, size=(256, 256))
    assert not idx.search(other, max_distance=8)


def test_verify_content_end_to_end(tmp_path):
    priv, pub = tmp_path / "p.pem", tmp_path / "pub.pem"
    generate_keypair(priv, pub)
    f = make_img(tmp_path / "a.png")
    sig = sign_file(f, priv)
    rep = verify_content(f, sig, pub, ProvenanceLedger(tmp_path / "l.jsonl"),
                         ImageIndex(tmp_path / "i.json"))
    assert rep["signature_valid"] and rep["verification_risk"] <= 0.1
