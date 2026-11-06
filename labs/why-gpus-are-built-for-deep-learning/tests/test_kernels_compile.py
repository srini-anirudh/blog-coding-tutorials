"""Every CUDA kernel compiles with NVRTC for sm_90. This needs no GPU."""

import pytest

from whygpu import kernels as K

DEFINES = {"chase": [{"LOAD": "*"}, {"LOAD": "__ldcg"}], "stream": [{"ILP": i} for i in (1, 2, 4, 8)],
           "mm_tiled": [{"T": t} for t in (8, 16, 32)]}


def _nvrtc_available():
    try:
        K.compile_offline("fma_chain")
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _nvrtc_available(), reason="NVRTC not available")


@pytest.mark.parametrize("name", sorted(K.SOURCES))
def test_compiles(name):
    for d in DEFINES.get(name, [{}]):
        assert len(K.compile_offline(name, **d)) > 1000


def test_maxrregcount_option_is_accepted():
    assert K.compile_offline("hungry", options=["--maxrregcount=32"])
