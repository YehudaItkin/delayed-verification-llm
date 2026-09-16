#!/usr/bin/env python3
"""Grounded factual-QA sweep (exploratory).
Archived F1 deviated from the written plan: labels 0/1 both had lag 0, 6 had lag 5;
initial answers used temperature .7 even in the temperature-0 update condition.
Future runs use theoretical delays and the selected initial temperature by default.
A bounded NLI coordinate is a surrogate, not a proof of an absorbing truth boundary.
"""
import os, json, time, argparse, sys
from history_index import history_index, effective_delay
import requests, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

API = os.environ.get("VLLM_API", "http://localhost:8001/v1/chat/completions")
KEY = os.environ.get("VLLM_API_KEY", "")
MODEL = os.environ.get("VLLM_MODEL", "qwen3.6-35b-a3b")

def llm(messages, max_tokens=60, temperature=0.7):
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens,
            "temperature": temperature, "chat_template_kwargs": {"enable_thinking": False}}
    for attempt in range(4):
        try:
            r = requests.post(API, json=body, headers={"Authorization": f"Bearer {KEY}"}, timeout=150)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            sys.stderr.write(f"  llm retry {attempt}: {e}\n"); time.sleep(4 * (attempt + 1))
    return ""

_NLI = {}
def nli_init(name="microsoft/deberta-large-mnli", device="cuda:0", revision=None):
    tok = AutoTokenizer.from_pretrained(name, revision=revision)
    mod = AutoModelForSequenceClassification.from_pretrained(name, revision=revision).to(device).eval()
    _NLI.update(tok=tok, mod=mod, device=device, lab2id={v.upper(): int(k) for k, v in mod.config.id2label.items()})

def _dist(ans, target):
    with torch.no_grad():
        x = _NLI["tok"](f"The answer is {ans}.", f"The answer is {target}.",
                        return_tensors="pt", truncation=True, max_length=128).to(_NLI["device"])
        p = torch.softmax(_NLI["mod"](**x).logits, -1)[0].cpu().numpy()
    s = p[_NLI["lab2id"]["ENTAILMENT"]] - p[_NLI["lab2id"]["CONTRADICTION"]]
    return float((1 - s) / 2)                          # 0 = entails target, 1 = contradicts

def signed(ans, gold, wrong):                          # +1 at gold, -1 at wrong
    dg, dw = _dist(ans, gold), _dist(ans, wrong)
    return (dw - dg) / (dw + dg + 1e-9)

def parse_ans(text):
    for line in reversed([l for l in text.splitlines() if l.strip()]):
        if "ANSWER:" in line.upper():
            return line.upper().split("ANSWER:", 1)[1].strip(" .:-\"'")[:80] or line.strip()[:80]
    return (text.strip().splitlines() or [""])[-1][:80]

def cold_answer(q, temp=0.7):
    return parse_ans(llm([{"role": "system", "content": "Answer the factual question with one short phrase. Reply exactly 'ANSWER: <answer>'."},
                          {"role": "user", "content": f"Question: {q}\nReply: ANSWER: <answer>"}], max_tokens=60, temperature=temp))

def verifier(stale_ans, evidence):
    return llm([{"role": "system", "content": "You are a fact verifier. Given EVIDENCE and a claimed answer, state in one short line what the evidence actually supports."},
                {"role": "user", "content": f"Evidence: {evidence[:600]}\nClaimed answer: {stale_ans}\nReply: VERDICT: <one line>"}], max_tokens=90, temperature=0.0)

def agent_update(q, prev, peers, vnote, temp):
    note = f"\nVerifier note (may reflect an EARLIER version of your answer): {vnote}" if vnote else ""
    return parse_ans(llm([{"role": "system", "content": "You are debating a factual question. Give your best CURRENT short answer, weighing your peers and the verifier note but thinking for yourself. Reply exactly 'ANSWER: <answer>'."},
                          {"role": "user", "content": f"Question: {q}\nYour previous answer: {prev}\nPeers' latest answers: {peers}{note}\nReply: ANSWER: <answer>"}], max_tokens=60, temperature=temp))

def run_f1(item, n_free, n_faulty, T, delta, temp, delay_convention="theory", initial_temp=None,
           initial_answers=None, verifier_on=True):
    q, gold, ev, wrong = item["q"], item["gold"], item["evidence"], item["wrong"]
    effective_delay(delta, delay_convention)
    if n_free < 1 or n_faulty < 0 or T < 2:
        raise ValueError("invalid agent count or horizon")
    cur = list(initial_answers) if initial_answers is not None else [cold_answer(q, temp if initial_temp is None else initial_temp) for _ in range(n_free)]
    if len(cur) != n_free:
        raise ValueError("initial answer count differs from free-agent count")
    buf = [cur[:]]
    notes = []
    traj  = [[_dist(a, gold) for a in cur]]
    straj = [[signed(a, gold, wrong) for a in cur]]
    for t in range(1, T):
        stale = buf[history_index(t, delta, delay_convention)]
        verdicts = [verifier(stale[i], ev) for i in range(n_free)] if verifier_on else [""]*n_free
        notes.append(verdicts)
        new = []
        for i in range(n_free):
            peers = [cur[j] for j in range(n_free) if j != i] + [wrong] * n_faulty
            new.append(agent_update(q, cur[i], peers, verdicts[i], temp))
        cur = new; buf.append(cur[:])
        traj.append([_dist(a, gold) for a in cur])
        straj.append([signed(a, gold, wrong) for a in cur])
    return {"q": q, "gold": gold, "wrong": wrong, "delta": delta, "n_faulty": n_faulty, "temp": temp,
            "answers": buf, "verifier_notes": notes, "verifier_on": verifier_on,
            "agent_nli_traj": traj, "agent_signed_traj": straj,
            "delay_convention": delay_convention, "effective_delay": effective_delay(delta, delay_convention),
            "mean_traj":   [float(np.mean(x)) for x in traj],
            "mean_s_traj": [float(np.mean(x)) for x in straj]}

def amp(mt):                                            # finite-window tail standard deviation
    mt = np.asarray(mt); tail = mt[len(mt) // 2:]
    return float(np.std(tail))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--delay-convention", choices=["theory", "legacy"], default="theory")
    ap.add_argument("--initial-temp", type=float, default=None, help="defaults to update temperature; use .7 for historical initialization")
    ap.add_argument("--sel", type=str, default="selected_big.json")
    ap.add_argument("--deltas", type=str, default="0,1,6")
    ap.add_argument("--faulty", type=str, default="0,4")
    ap.add_argument("--temps", type=str, default="0.7,0.0")
    ap.add_argument("--T", type=int, default=20)
    ap.add_argument("--n_free", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--nq", type=int, default=15)
    ap.add_argument("--out", type=str, default="f1_results.json")
    a = ap.parse_args()
    nli_init()

    sel = json.load(open(a.sel))[:a.nq]
    deltas = [int(x) for x in a.deltas.split(",")]
    faulty = [int(x) for x in a.faulty.split(",")]
    temps  = [float(x) for x in a.temps.split(",")]
    res = {"meta": {"delay_convention": a.delay_convention,
                    "effective_delays": [effective_delay(d, a.delay_convention) for d in deltas],
                    "seed_role": "replicate index; no RNG seed passed to backend",
                    "T": a.T, "deltas": deltas, "faulty": faulty, "temps": temps, "n_free": a.n_free,
                    "seeds": a.seeds, "model": MODEL, "n_items": len(sel),
                    "analysis_status": "exploratory; historical plan not fulfilled",
                    "initial_temperature": a.initial_temp if a.initial_temp is not None else "same as update"}, "runs": []}
    total = len(faulty) * len(deltas) * len(temps) * a.seeds * len(sel); done = 0
    for nf in faulty:
        for delta in deltas:
            for temp in temps:
                for s in range(a.seeds):
                    for it in sel:
                        t0 = time.time(); r = run_f1(it, a.n_free, nf, a.T, delta, temp, a.delay_convention, a.initial_temp); r["seed"] = s
                        r["amp"] = amp(r["mean_traj"]); r["amp_s"] = amp(r["mean_s_traj"])
                        res["runs"].append(r); done += 1
                        print(f"[{done}/{total}] nf={nf} d={delta} T={temp} q={it['q'][:30]!r} "
                              f"amp={r['amp']:.4f} e0={r['mean_traj'][0]:.2f}->ef={r['mean_traj'][-1]:.2f} "
                              f"({time.time()-t0:.0f}s)", flush=True)
                        json.dump(res, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}  ({done} runs)")
