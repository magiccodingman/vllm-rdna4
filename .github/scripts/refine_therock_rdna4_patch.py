from pathlib import Path

root = Path(__file__).resolve().parents[2]
path = root / "vllm/_aiter_ops.py"
text = path.read_text()

replacements = {
    "    Checks: platform (ROCm), supported device arch (MI3xx or gfx12x), and library existence.\n": (
        "    Checks: platform (ROCm), supported device arch (MI3xx or gfx12x),\n"
        "    and library existence.\n"
    ),
    '''        return (
            aiter_ar_comm if isinstance(aiter_ar_comm, AiterCustomAllreduce) else None
        )
''': '''        if (
            isinstance(aiter_ar_comm, AiterCustomAllreduce)
            and not aiter_ar_comm.disabled
        ):
            return aiter_ar_comm
        return None
''',
}

for old, new in replacements.items():
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one patch anchor, found {count}: {old!r}")
    text = text.replace(old, new, 1)

path.write_text(text)
compile(text, str(path), "exec")
print("TheRock refinement applied and syntax-checked")
