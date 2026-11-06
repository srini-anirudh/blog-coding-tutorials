from whyslow.memory import diagnose_oom, parse_oom

MSG = ("CUDA out of memory. Tried to allocate 1.00 GiB. GPU 0 has a total capacity of 79.19 GiB of which 76.90 GiB "
       "is free. Including non-PyTorch memory, this process has 2.03 GiB memory in use. Of the allocated memory "
       "320.00 MiB is allocated by PyTorch, and 1.25 GiB is reserved by PyTorch but unallocated. If reserved but "
       "unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")


def test_parse_oom_reads_four_numbers():
    f = parse_oom(MSG)
    assert f["requested"] == 1 << 30
    assert f["allocated"] == 320 << 20
    assert f["reserved_unallocated"] == int(1.25 * (1 << 30))
    assert abs(f["free"] - 76.90 * (1 << 30)) < 1 << 20


def test_diagnosis_fragmentation_vs_capacity():
    f = parse_oom(MSG)
    assert diagnose_oom(f, limit=2 << 30).startswith("fragmentation")
    tight = {"requested": 1 << 30, "free": 100 << 20, "allocated": 79 << 30, "reserved_unallocated": 10 << 20}
    assert diagnose_oom(tight).startswith("capacity")
