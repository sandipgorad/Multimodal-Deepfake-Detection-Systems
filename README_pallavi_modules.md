# Pallavi's modules: metadata verification + mitigation

Copy `src/metadata`, `src/mitigation`, `tests/test_metadata_mitigation.py` into the repo
(on branch `pallavi/image-detection`, based on `development`).

Dependencies: `pillow numpy cryptography pytest` (optional: `pymongo`, `ffmpeg` for audio/video tags)

## Usage
```python
from src.mitigation.verify import verify_content
from src.mitigation.provenance import ProvenanceLedger
from src.mitigation.reverse_search import ImageIndex

report = verify_content("upload.jpg", ledger=ProvenanceLedger(), index=ImageIndex())
fusion_feature = report["verification_risk"]   # 0..1, feed into Feature Fusion Engine
```
Run tests: `python -m pytest tests -q`
