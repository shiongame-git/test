import hashlib
from pathlib import Path

from packing import read_box_sizes

ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_cube_fixture_locked():
    p = ROOT / "data" / "box_sizes_cube.txt"
    assert _sha(p) == "b57a8c73bc7b66b9807e55d58d17a23fa9294ce652c98597642ea0ed889d906a"
    c, boxes = read_box_sizes(p)
    assert (c.W, c.H, c.D) == (20, 20, 20)
    assert len(boxes) == 200
    assert sum(b.volume for b in boxes) == 9000


def test_cuboid_fixture_locked():
    p = ROOT / "data" / "box_sizes_cuboid.txt"
    assert _sha(p) == "908bb73f607d377e76b8e50c92f04a3c029e05cc3eee57e9b988c36352f3da9f"
    c, boxes = read_box_sizes(p)
    assert (c.W, c.H, c.D) == (20, 20, 20)
    assert len(boxes) == 200
    assert sum(b.volume for b in boxes) == 9000
