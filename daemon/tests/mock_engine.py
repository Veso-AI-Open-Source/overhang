#!/usr/bin/env python3
"""Mock engine speaking the PROTOCOL.md engine side, for overhangd testing.

Reads JSON-lines requests on stdin; emits the ready line on start, then for
each request emits 3 fixed tok lines [9906, 11, 220] and a done line.
Stats/logs go to stderr only.
"""
import json
import os
import sys
import time

FIXED_TOKENS = [9906, 11, 220]


def main():
    # Test hook: containers named *_bad simulate an engine that dies before
    # printing the ready line (SNAP is set by Engine::spawn).
    if os.environ.get("SNAP", "").endswith("_bad"):
        sys.exit(1)
    print(
        json.dumps({"ready": True, "model": "mock", "n_layers": 40, "vocab": 248320}),
        flush=True,
    )
    print("mock_engine: ready", file=sys.stderr, flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        rid = req.get("id")
        if req.get("stop"):
            print(
                json.dumps(
                    {
                        "id": rid,
                        "done": True,
                        "n_out": 0,
                        "prefill_s": 0.0,
                        "decode_s": 0.0,
                        "hit": 0.0,
                    }
                ),
                flush=True,
            )
            continue
        print(
            "mock_engine: request id=%s n_ids=%d n=%s reset=%s"
            % (rid, len(req.get("ids", [])), req.get("n"), req.get("reset")),
            file=sys.stderr,
            flush=True,
        )
        for tok in FIXED_TOKENS:
            time.sleep(0.01)
            print(json.dumps({"id": rid, "tok": tok}), flush=True)
        print(
            json.dumps(
                {
                    "id": rid,
                    "done": True,
                    "n_out": len(FIXED_TOKENS),
                    "prefill_s": 0.01,
                    "decode_s": 0.03,
                    "hit": 0.5,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
