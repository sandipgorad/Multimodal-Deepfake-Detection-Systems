"""One call that runs every verification step and returns a single report."""
from src.metadata.metadata_check import check_metadata
from src.mitigation.provenance import ProvenanceLedger
from src.mitigation.reverse_search import ImageIndex
from src.mitigation.signature import verify_file
from src.mitigation.watermark import detect_content_credentials

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def verify_content(path, signature_hex=None, public_key="keys/public.pem",
                   ledger=None, index=None):
    report = {"metadata": check_metadata(path)}
    risk = report["metadata"]["metadata_score"]

    cred = detect_content_credentials(path)
    report["content_credentials"] = cred
    if cred["ai_generated_marker"]:
        risk = max(risk, 0.9)

    if signature_hex:
        ok = verify_file(path, signature_hex, public_key)
        report["signature_valid"] = ok
        if ok:
            risk = min(risk, 0.1)  # trusted publisher signature
        else:
            risk = max(risk, 0.8)  # signature present but does not match

    if ledger is not None:
        report["seen_before"] = ledger.lookup(path)

    if index is not None and str(path).lower().endswith(IMAGE_EXT):
        report["similar_known_images"] = index.search(path)

    report["verification_risk"] = round(risk, 2)
    return report
