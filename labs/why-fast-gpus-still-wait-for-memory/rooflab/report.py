"""Every script writes what it prints to results/<name>.txt, headed by the hardware and the command."""

import json
import sys
from pathlib import Path

from .hardware import hardware_note

LAB = Path(__file__).resolve().parent.parent
RESULTS = LAB / "results"


class _Tee:
    def __init__(self, path: Path):
        self.f = open(path, "w")
        self.out = sys.stdout

    def write(self, s):
        self.out.write(s)
        self.f.write(s)

    def flush(self):
        self.out.flush()
        self.f.flush()


def start(name: str, save: bool = True) -> None:
    """Print the header and, unless `save` is False, copy all later output to results/<name>.txt."""
    if save:
        RESULTS.mkdir(exist_ok=True)
        sys.stdout = _Tee(RESULTS / f"{name}.txt")
    script = Path(sys.argv[0]).resolve()
    rel = script.relative_to(LAB.parent.parent) if LAB.parent.parent in script.parents else script.name
    print(f"# {hardware_note()}")
    print(f"# command: python {rel} {' '.join(sys.argv[1:])}".rstrip())


def save_json(name: str, data) -> None:
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps({"hardware": hardware_note(), **data}, indent=1))


def load_json(name: str) -> dict | None:
    p = RESULTS / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def measured_ceilings() -> dict:
    """Measured HBM bandwidth and compute ceilings from scripts 03 and 04, with spec values as fallback."""
    from .hardware import H100, H100_TF32

    out = {"beta": H100.hbm, "bf16": H100.bf16, "fp32": H100.fp32, "tf32": H100_TF32, "source": "spec"}
    bw, cc = load_json("03_bandwidth"), load_json("04_compute_ceilings")
    if bw:
        out["beta"] = bw["hbm_bytes_per_s"]
    if cc:
        out.update({k: cc["ceilings"][k] for k in ("bf16", "fp32", "tf32") if k in cc["ceilings"]})
    if bw and cc:
        out["source"] = "measured (03_bandwidth, 04_compute_ceilings)"
    return out
