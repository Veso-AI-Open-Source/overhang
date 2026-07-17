#!/usr/bin/env python3
"""Agentic capability harness for local MLX models.

Runs a model through a ReAct-style tool loop against scripted scenarios that
probe the capabilities long-horizon agents need:
  T1 tool-call     emit a correctly-formatted tool call when one is needed
  T2 chaining      multi-step: write a file, read it back, report on it
  T3 planning      produce a valid ordered JSON plan for a goal
  T4 state         track mutating state (todo list) across turns
  T5 termination   answer directly when no tool is needed (no tool spam)
  T6 grounding     use the tool RESULT (not prior belief) in the final answer

Scoring is mechanical (no LLM judge): regex/JSON checks on transcripts.
Reports per-scenario pass/fail + measured decode tok/s.

Usage: agent_harness.py [--model mlx-community/Qwen3.5-2B-4bit] [--turns 6]
"""
import argparse, json, math, os, re, shutil, sys, tempfile, time
from pathlib import Path

from mlx_lm import load, generate

SANDBOX = Path(tempfile.mkdtemp(prefix="agent_sandbox_"))

# ---------------- tools ----------------
def t_calc(expression: str) -> str:
    try:
        allowed = {"__builtins__": {}, "sqrt": math.sqrt, "pi": math.pi}
        return str(eval(expression, allowed))         # sandboxed: no builtins
    except Exception as e:
        return f"error: {e}"

def t_write(filename: str, content: str) -> str:
    p = SANDBOX / Path(filename).name
    p.write_text(content)
    return f"wrote {len(content)} chars to {p.name}"

def t_read(filename: str) -> str:
    p = SANDBOX / Path(filename).name
    return p.read_text() if p.exists() else "error: no such file"

TODO: list = []
def t_todo(action: str, item: str = "") -> str:
    if action == "add": TODO.append(item); return f"added; list={TODO}"
    if action == "remove" and item in TODO: TODO.remove(item); return f"removed; list={TODO}"
    if action == "list": return f"list={TODO}"
    return "error: bad action"

TOOLS = {
    "calculator": {"fn": t_calc, "desc": "Evaluate a math expression.",
                   "params": {"expression": "string, e.g. '37*89'"}},
    "write_file": {"fn": t_write, "desc": "Write content to a file.",
                   "params": {"filename": "string", "content": "string"}},
    "read_file":  {"fn": t_read, "desc": "Read a file's content.",
                   "params": {"filename": "string"}},
    "todo":       {"fn": t_todo, "desc": "Manage a todo list.",
                   "params": {"action": "add|remove|list", "item": "string (for add/remove)"}},
}

SYSTEM = """You are an agent with tools. To use a tool, reply with ONLY:
<tool_call>{"name": "<tool>", "arguments": {...}}</tool_call>
Available tools:
""" + "\n".join(f'- {n}: {t["desc"]} args: {json.dumps(t["params"])}' for n, t in TOOLS.items()) + """
After a tool returns, either call another tool or give your final answer as plain text.
If no tool is needed, answer directly. Be concise. /no_think"""

TC_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)

def run_episode(model, tok, user_msg, max_turns):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_msg}]
    transcript, calls, gen_toks, gen_secs = [], [], 0, 0.0
    for _ in range(max_turns):
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        t0 = time.time()
        out = generate(model, tok, prompt=prompt, max_tokens=512, verbose=False)
        dt = time.time() - t0
        # strip any think block
        out_clean = re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip()
        gen_toks += len(tok.encode(out)); gen_secs += dt
        transcript.append(("assistant", out_clean))
        m = TC_RE.search(out_clean)
        if not m:
            return out_clean, transcript, calls, gen_toks, gen_secs   # final answer
        try:
            call = json.loads(m.group(1))
            name, args = call["name"], call.get("arguments", {})
        except (json.JSONDecodeError, KeyError) as e:
            result = f"error: malformed tool call ({e})"
            name, args = "?", {}
        else:
            calls.append((name, args))
            fn = TOOLS.get(name, {}).get("fn")
            result = fn(**args) if fn else f"error: unknown tool {name}"
        transcript.append(("tool", str(result)))
        msgs.append({"role": "assistant", "content": out_clean})
        msgs.append({"role": "user", "content": f"<tool_response>{result}</tool_response>"})
    return None, transcript, calls, gen_toks, gen_secs                 # ran out of turns

# ---------------- scenarios + mechanical scoring ----------------
def s1(model, tok, turns):
    final, tr, calls, gt, gs = run_episode(model, tok,
        "What is 37*89 + 12345/5? Use the calculator.", turns)
    used = any(c[0] == "calculator" for c in calls)
    right = final is not None and ("5762" in final)                    # 3293+2469
    return used and right, f"calc used={used} answer_ok={right}", gt, gs

def s2(model, tok, turns):
    final, tr, calls, gt, gs = run_episode(model, tok,
        "Write a file called notes.txt containing exactly 'colibri streams experts'. "
        "Then read it back and tell me how many words it contains.", turns)
    wrote = (SANDBOX / "notes.txt").exists() and (SANDBOX / "notes.txt").read_text() == "colibri streams experts"
    right = final is not None and re.search(r"\b(3|three)\b", final or "")
    return bool(wrote and right), f"file_ok={wrote} count_ok={bool(right)}", gt, gs

def s3(model, tok, turns):
    final, tr, calls, gt, gs = run_episode(model, tok,
        'Produce a plan to deploy a web app as a JSON array of exactly 4 steps, '
        'each {"step": <n>, "action": "<verb phrase>"}. Output ONLY the JSON array, no tools.', turns)
    try:
        arr = json.loads(re.search(r"\[.*\]", final or "", re.S).group(0))
        ok = (len(arr) == 4 and all(set(x) == {"step", "action"} for x in arr)
              and [x["step"] for x in arr] == [1, 2, 3, 4])
    except Exception:
        ok = False
    return ok, f"plan_json_ok={ok} tool_calls={len(calls)}", gt, gs

def s4(model, tok, turns):
    TODO.clear()
    final, tr, calls, gt, gs = run_episode(model, tok,
        "Using the todo tool: add 'buy milk', add 'ship code', add 'call mom', "
        "then remove 'ship code', then tell me what remains on the list.", turns)
    state_ok = TODO == ["buy milk", "call mom"]
    said_ok = final is not None and "milk" in final.lower() and "mom" in final.lower() \
              and "ship" not in final.lower().replace("shipped", "")
    return state_ok and bool(said_ok), f"state={TODO} answer_ok={bool(said_ok)}", gt, gs

def s5(model, tok, turns):
    final, tr, calls, gt, gs = run_episode(model, tok,
        "What is the capital of France?", turns)
    return final is not None and "paris" in final.lower() and len(calls) == 0, \
           f"answered={final is not None} tool_calls={len(calls)}", gt, gs

def s6(model, tok, turns):
    (SANDBOX / "secret.txt").write_text("the launch code is BANANA-42")
    final, tr, calls, gt, gs = run_episode(model, tok,
        "Read the file secret.txt and tell me the launch code it contains.", turns)
    read = any(c[0] == "read_file" for c in calls)
    right = final is not None and "banana-42" in final.lower()
    return read and right, f"read={read} grounded={right}", gt, gs

SCENARIOS = [("T1 tool-call", s1), ("T2 chaining", s2), ("T3 planning", s3),
             ("T4 state", s4), ("T5 termination", s5), ("T6 grounding", s6)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3.5-2B-4bit")
    ap.add_argument("--turns", type=int, default=6)
    args = ap.parse_args()

    print(f"loading {args.model} ...", flush=True)
    t0 = time.time()
    model, tok = load(args.model)
    print(f"loaded in {time.time()-t0:.1f}s\n", flush=True)

    total_toks, total_secs, passed = 0, 0.0, 0
    for name, fn in SCENARIOS:
        try:
            ok, detail, gt, gs = fn(model, tok, args.turns)
        except Exception as e:
            ok, detail, gt, gs = False, f"harness error: {e}", 0, 0.0
        total_toks += gt; total_secs += gs; passed += ok
        print(f"{'PASS' if ok else 'FAIL'}  {name:15s} {detail}", flush=True)

    print(f"\nscore: {passed}/{len(SCENARIOS)}")
    if total_secs: print(f"decode: {total_toks} tokens in {total_secs:.1f}s = {total_toks/total_secs:.1f} tok/s")
    shutil.rmtree(SANDBOX, ignore_errors=True)

if __name__ == "__main__":
    main()
