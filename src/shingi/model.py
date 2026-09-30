"""Pinned identity of the released model, its calibration and its runtime."""
import hashlib
import json
import os
from pathlib import Path

from .decision import Calibration

HF_REPO = "kortexa-ai/shingi-27b"
MODEL_FILE = "shingi-27b.gguf"
MODEL_SHA256 = "277d970372fac37185bb621341d99bec3c7f1d86c6657d27ca18a6dcf930bd03"
CALIBRATION_FILE = "calibration.json"
CALIBRATION_SHA256 = "cb894624520884d7d002c958099fa285ee62eae8d4fdac0fe369e7b519f1f16c"
RUNTIME = {"repository": "https://github.com/PrismML-Eng/llama.cpp",
           "revision": "d8f26eec76da6d09bb708bcba51ef64b8cd868a3"}
# Bonsai 2 27B vision projector (Apache-2.0), used unchanged. One pinned source.
PROJECTOR = {"repository": "prism-ml/Ternary-Bonsai-2-27B-gguf",
             "revision": "6ed5e12bf84b7a63069882c91dd9e9218647d17b",
             "filename": "Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf",
             "sha256": "6807ede61d570bb86ba34b756a0fa109edc33668604de867c6ea6d8f1d631903"}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fetch(filename):
    """Download a release file into the standard Hugging Face cache (honours HF_HOME and HF_TOKEN)."""
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(HF_REPO, filename, revision=os.environ.get("SHINGI_REVISION", "main")))


def fetch_projector():
    """Download the pinned vision projector into the standard Hugging Face cache."""
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(PROJECTOR["repository"], PROJECTOR["filename"], revision=PROJECTOR["revision"]))


def verify(what, actual, expected):
    if actual != expected:
        raise RuntimeError(f"{what} SHA-256 is {actual}, expected {expected}. "
                           "Delete the file to download it again, or pass --skip-verify to use it anyway.")


def load_calibration(path):
    data = Path(path).read_bytes()
    return Calibration(**json.loads(data)["parameters"]), hashlib.sha256(data).hexdigest()
