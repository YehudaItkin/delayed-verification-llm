#!/usr/bin/env python3
"""Variant A: larger wrong-majority study with a verification-GAIN knob (kappa) and seeds.

Differences from debate_exp.py:
  - wrong MAJORITY: n_free=3 free agents vs n_faulty=4 stubborn-wrong agents (forcing F dominates the
    social signal).
  - kappa (verification strength) operationalized as verifier ASSERTIVENESS: 'weak' (FYI note) vs
    'strong' (authoritative correction). This is the real-agent analogue of the restoring-force gain.
  - sweep kappa in {weak,strong} x delta in {1,4} x seed in {0,1}; aggregate statistics.
Theory prediction: oscillation/instability grows with BOTH kappa and delta (both push past the dose
limit kappa_max(delta)).

Reads selected.json (movable questions with pre-generated wrong answers) from debate_exp.py calibrate.
"""
import os, json, time, argparse, sys
import requests, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

API = "http://localhost:8001/v1/chat/completions"
KEY = os.environ.get("VLLM_API_KEY", "")  # export VLLM_API_KEY=... before running
MODEL = "qwen3.6-35b-a3b"

def llm(messages, max_tokens=70, temperature=0.7):
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
def nli_init(name="microsoft/deberta-large-mnli"):
    tok = AutoTokenizer.from_pretrained(name)
    mod = AutoModelForSequenceClassification.from_pretrained(name).to("cuda:0").eval()
    _NLI.update(tok=tok, mod=mod, lab2id={v.upper(): int(k) for k, v in mod.config.id2label.items()})

def err(ans, gold):
    with torch.no_grad():
        x = _NLI["tok"](f"The answer is {ans}.", f"The answer is {gold}.",
                        return_tensors="pt", truncation=True, max_length=128).to("cuda:0")
        p = torch.softmax(_NLI["mod"](**x).logits, -1)[0].cpu().numpy()
    return float((1 - (p[_NLI["lab2id"]["ENTAILMENT"]] - p[_NLI["lab2id"]["CONTRADICTION"]])) / 2)

def parse_ans(text):
    for line in reversed([l for l in text.splitlines() if l.strip()]):
        if "ANSWER:" in line.upper():
            return line.upper().split("ANSWER:", 1)[1].strip(" .:-\"'")[:80] or line.strip()[:80]
    return (text.strip().splitlines() or [""])[-1][:80]

def cold_answer(q, temp):
    return parse_ans(llm([{"role": "system", "content": "Answer the factual question with one short phrase. Reply exactly 'ANSWER: <answer>'."},
                          {"role": "user", "content": f"Question: {q}\nReply: ANSWER: <answer>"}], max_tokens=60, temperature=temp))

def verifier(stale_ans, evidence):
    return llm([{"role": "system", "content": "You are a fact verifier. Given EVIDENCE and a claimed answer, state in one short line what the evidence actually supports."},
                {"role": "user", "content": f"Evidence: {evidence[:600]}\nClaimed answer: {stale_ans}\nReply: VERDICT: <one line>"}], max_tokens=90, temperature=0.0)

def agent_update(q, prev, peers, verdict, kappa, temp):
    if kappa == "forceful":   # maximal effective gain: copy the (possibly stale) verdict verbatim
        note = f"\nVERIFIER (FINAL AUTHORITY): {verdict}. Set your answer to EXACTLY what this verifier states; discard your previous answer and your peers entirely."
        sysmsg = "You defer completely to the verifier. Reply exactly 'ANSWER: <answer>', copying the verifier's stated answer verbatim."
    elif kappa == "strong":
        note = f"\nVERIFIER CORRECTION (authoritative, from the evidence; may reflect an earlier answer): {verdict}. Update your answer to match the evidence."
        sysmsg = "You are debating a factual question. Give your best CURRENT short answer, weighing your peers and the verifier note. Reply exactly 'ANSWER: <answer>'."
    else:
        note = f"\nVerifier note (FYI, may reflect an earlier version of your answer): {verdict}"
        sysmsg = "You are debating a factual question. Give your best CURRENT short answer, weighing your peers and the verifier note. Reply exactly 'ANSWER: <answer>'."
    return parse_ans(llm([{"role": "system", "content": sysmsg},
                          {"role": "user", "content": f"Question: {q}\nYour previous answer: {prev}\nPeers' latest answers: {peers}{note}\nReply: ANSWER: <answer>"}], max_tokens=60, temperature=temp))

def run_debate(item, kappa, delta, n_free=3, n_faulty=4, T=18, temp=0.7):
    q, gold, ev, wrong = item["q"], item["gold"], item["evidence"], item["wrong"]
    cur = [cold_answer(q, temp) for _ in range(n_free)]
    buf = [cur[:]]; traj = [[err(a, gold) for a in cur]]
    for t in range(1, T):
        stale = buf[max(0, t - delta)]
        verds = [verifier(stale[i], ev) for i in range(n_free)]
        new = []
        for i in range(n_free):
            peers = [cur[j] for j in range(n_free) if j != i] + [wrong] * n_faulty
            new.append(agent_update(q, cur[i], peers, verds[i], kappa, temp))
        cur = new; buf.append(cur[:]); traj.append([err(a, gold) for a in cur])
    return [float(np.mean(x)) for x in traj]

def osc_index(mt):
    d = np.diff(np.asarray(mt)[3:]); d = d[np.abs(d) > 0.02]
    return int(np.sum(np.abs(np.diff(np.sign(d))) > 0)) if len(d) > 1 else 0

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sel", default="selected.json")
    ap.add_argument("--out", default="expA_results.json")
    ap.add_argument("--T", type=int, default=18)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--kappas", type=str, default="weak,strong")
    ap.add_argument("--deltas", type=str, default="1,4")
    ap.add_argument("--nq", type=int, default=0)   # 0 = all questions
    a = ap.parse_args()
    nli_init()
    sel = json.load(open(a.sel))
    if a.nq:
        sel = sel[:a.nq]
    kappas = a.kappas.split(",")
    deltas = [int(x) for x in a.deltas.split(",")]
    res = {"meta": {"T": a.T, "kappas": kappas, "deltas": deltas, "seeds": a.seeds,
                    "n_free": 3, "n_faulty": 4, "model": MODEL, "n_items": len(sel)}, "runs": []}
    for it in sel:
        for kappa in kappas:
            for delta in deltas:
                for s in range(a.seeds):
                    t0 = time.time()
                    mt = run_debate(it, kappa, delta, T=a.T, temp=0.7)
                    rec = {"q": it["q"], "kappa": kappa, "delta": delta, "seed": s,
                           "mean_traj": mt, "osc": osc_index(mt), "final": mt[-1],
                           "conv": int(mt[-1] < 0.1)}
                    res["runs"].append(rec)
                    print(f"  q={it['q'][:26]!r} k={kappa} d={delta} s={s} "
                          f"osc={rec['osc']} fin={mt[-1]:.2f} ({time.time()-t0:.0f}s)")
                    json.dump(res, open(a.out, "w"), indent=1)
    # aggregate
    print("\n=== AGGREGATE (mean over questions x seeds) ===")
    for kappa in kappas:
        for delta in deltas:
            rs = [r for r in res["runs"] if r["kappa"] == kappa and r["delta"] == delta]
            print(f"  kappa={kappa:6s} delta={delta}: osc={np.mean([r['osc'] for r in rs]):.2f} "
                  f"final_err={np.mean([r['final'] for r in rs]):.3f} "
                  f"conv={np.mean([r['conv'] for r in rs]):.2f}")
    print(f"wrote {a.out}")
