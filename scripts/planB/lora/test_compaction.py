"""python test_compaction.py: train_lora.compacted_context rebuilds the prompt ADK sends after compactions."""
from train_lora import compacted_context

m = lambda r, t, c="": {"role": r, "_ts": t, "content": c}
msgs = [{"role": "system", "content": "S"}, m("user", 1, "U"), m("assistant", 2, "a2"), m("tool", 2, "t2"),
        m("assistant", 3, "a3"), m("tool", 3, "t3"), m("assistant", 4, "a4"), m("tool", 4, "t4"), m("assistant", 5, "a5")]
c1 = {"created": 3.5, "start": 1, "end": 2, "summary": "X"}
c2 = {"created": 4.5, "start": 1, "end": 3, "summary": "Y"}        # wider: subsumes c1
assert compacted_context(msgs, 4, [c1, c2]) == msgs[:4]            # turn at ts 3: nothing compacted yet
got = [x["content"] for x in compacted_context(msgs, 6, [c1, c2])]
assert got == ["S", "X", "a3", "t3"], got                          # turn at ts 4: c1 only
got = [x["content"] for x in compacted_context(msgs, 8, [c1, c2])]
assert got == ["S", "Y", "a4", "t4"], got                          # turn at ts 5: c2 replaces c1
print("ok")
