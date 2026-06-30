#!/usr/bin/env python3
"""Real LLM-debate front-end for the delayed-verification experiment (paper section 7).

Free agents debate a factual question while a stubborn FAULTY agent injects a fixed wrong
answer (the bias forcing F) and a VERIFIER corrects toward truth using delayed (round t-delta)
information plus wiki evidence (the rectifier R). We log the per-agent error signal
  e_{i,t} = (1 - s)/2,   s = P(entail) - P(contradict)   (deberta-large-mnli, answer vs gold)
and study convergence vs oscillation as the verification delay delta grows.

Backend: vLLM Qwen3.6-35B OpenAI API on :8001.   NLI: microsoft/deberta-large-mnli on cuda:0.
"""
import os, json, time, argparse, sys
import requests
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

API = "http://localhost:8001/v1/chat/completions"
KEY = os.environ.get("VLLM_API_KEY", "")  # export VLLM_API_KEY=... before running
MODEL = "qwen3.6-35b-a3b"

def llm(messages, max_tokens=200, temperature=0.7):
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens,
            "temperature": temperature, "chat_template_kwargs": {"enable_thinking": False}}
    for attempt in range(4):
        try:
            r = requests.post(API, json=body, headers={"Authorization": f"Bearer {KEY}"}, timeout=150)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            sys.stderr.write(f"  llm retry {attempt}: {e}\n")
            time.sleep(4 * (attempt + 1))
    return ""

# ---- NLI error signal ----
_NLI = {}
def nli_init(name="microsoft/deberta-large-mnli"):
    tok = AutoTokenizer.from_pretrained(name)
    mod = AutoModelForSequenceClassification.from_pretrained(name).to("cuda:0").eval()
    lab2id = {v.upper(): int(k) for k, v in mod.config.id2label.items()}
    _NLI.update(tok=tok, mod=mod, lab2id=lab2id)
    return mod.config.id2label

def err(ans, gold):
    prem, hyp = f"The answer is {ans}.", f"The answer is {gold}."
    with torch.no_grad():
        x = _NLI["tok"](prem, hyp, return_tensors="pt", truncation=True, max_length=128).to("cuda:0")
        p = torch.softmax(_NLI["mod"](**x).logits, -1)[0].cpu().numpy()
    s = p[_NLI["lab2id"]["ENTAILMENT"]] - p[_NLI["lab2id"]["CONTRADICTION"]]
    return float((1 - s) / 2)

def parse_ans(text):
    for line in reversed([l for l in text.splitlines() if l.strip()]):
        if "ANSWER:" in line.upper():
            return line.upper().split("ANSWER:", 1)[1].strip(" .:-")[:80] or line.strip()[:80]
    return (text.strip().splitlines() or [""])[-1][:80]

def cold_answer(q):
    return parse_ans(llm(
        [{"role": "system", "content": "Answer the factual question with one short phrase. Reply exactly 'ANSWER: <answer>'."},
         {"role": "user", "content": f"Question: {q}\nReply: ANSWER: <answer>"}], max_tokens=80))

def verifier(stale_ans, evidence):
    return llm(
        [{"role": "system", "content": "You are a fact verifier. Given EVIDENCE and a claimed answer, state in one short line what the evidence actually supports."},
         {"role": "user", "content": f"Evidence: {evidence}\nClaimed answer: {stale_ans}\nReply: VERDICT: <one line>"}],
        max_tokens=100, temperature=0.0)

def agent_update(q, prev, peers, vnote):
    note = f"\nVerifier note (may reflect an EARLIER version of your answer): {vnote}" if vnote else ""
    return parse_ans(llm(
        [{"role": "system", "content": "You are debating a factual question. Give your best CURRENT short answer, weighing your peers and the verifier note but thinking for yourself. Reply exactly 'ANSWER: <answer>'."},
         {"role": "user", "content": f"Question: {q}\nYour previous answer: {prev}\nPeers' latest answers: {peers}{note}\nReply: ANSWER: <answer>"}],
        max_tokens=80))

def run_question(item, n_free=3, T=16, delta=1, verbose=False):
    q, gold, wrong, ev = item["q"], item["gold"], item["wrong"], item["evidence"]
    cur = [cold_answer(q) for _ in range(n_free)]
    buf = [cur[:]]                       # buffer of past free answers (for delayed verifier)
    traj = [[err(a, gold) for a in cur]]
    verdicts = None
    for t in range(1, T):
        stale = buf[max(0, t - delta)]   # delayed information
        verdicts = [verifier(stale[i], ev) for i in range(n_free)]
        new = []
        for i in range(n_free):
            peers = [cur[j] for j in range(n_free) if j != i] + [wrong]   # wrong = stubborn faulty agent
            new.append(agent_update(q, cur[i], peers, verdicts[i]))
        cur = new; buf.append(cur[:])
        traj.append([err(a, gold) for a in cur])
        if verbose:
            print(f"    t={t} delta={delta} mean_err={np.mean(traj[-1]):.3f} answers={cur}")
    return {"q": q, "gold": gold, "wrong": wrong, "delta": delta, "traj": traj}

def osc_index(mean_traj):
    """sign changes of successive differences after a short transient -> oscillation proxy."""
    d = np.diff(np.asarray(mean_traj)[3:])
    return int(np.sum(np.abs(np.diff(np.sign(d[np.abs(d) > 1e-3]))) > 0))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--deltas", type=str, default="1,4")
    ap.add_argument("--nfree", type=int, default=3)
    ap.add_argument("--out", type=str, default="debate_results.json")
    args = ap.parse_args()

    QUESTIONS = [
        {"q": "In which year did the Apollo 11 mission first land humans on the Moon?",
         "gold": "1969", "wrong": "1972",
         "evidence": "Apollo 11 was the spaceflight that first landed humans on the Moon, on July 20, 1969."},
        {"q": "What is the chemical symbol for the element gold?",
         "gold": "Au", "wrong": "Gd",
         "evidence": "Gold is a chemical element; its standard symbol is Au, from the Latin aurum."},
        {"q": "Who wrote the play 'Hamlet'?",
         "gold": "William Shakespeare", "wrong": "Christopher Marlowe",
         "evidence": "Hamlet is a tragedy written by William Shakespeare around 1599-1601."},
    ]

    print("loading NLI ..."); print(" labels:", nli_init())
    print("vLLM connectivity test:", repr(llm(
        [{"role": "user", "content": "Reply exactly: ANSWER: ping"}], max_tokens=30)))

    if args.smoke:
        r = run_question(QUESTIONS[0], n_free=2, T=6, delta=4, verbose=True)
        print("smoke mean-error trajectory:", [round(np.mean(x), 3) for x in r["traj"]])
        sys.exit(0)

    deltas = [int(x) for x in args.deltas.split(",")]
    results = {"meta": {"T": args.T, "deltas": deltas, "nfree": args.nfree, "model": MODEL}, "runs": []}
    for delta in deltas:
        for item in QUESTIONS:
            t0 = time.time()
            r = run_question(item, n_free=args.nfree, T=args.T, delta=delta, verbose=True)
            mt = [float(np.mean(x)) for x in r["traj"]]
            r["mean_traj"] = mt; r["osc"] = osc_index(mt)
            results["runs"].append(r)
            print(f"  done q='{item['q'][:30]}...' delta={delta} osc={r['osc']} "
                  f"final={mt[-1]:.3f} ({time.time()-t0:.0f}s)")
            json.dump(results, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}")
