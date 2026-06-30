#!/usr/bin/env python3
"""Section-7 experiment: real LLM-debate under delayed verification, with a WRONG MAJORITY.

Pipeline:
  calibrate : cold-answer PsiloQA items with Qwen3.6-35B, keep the "movable" ones (model wrong /
              uncertain by NLI-vs-gold), and pre-generate a plausible WRONG answer for the faulty
              agents. Saves selected.json.
  run       : for each selected item, run a debate with n_free free agents + n_faulty stubborn
              wrong agents + a verifier that uses round (t-delta) info and the wiki evidence.
              Logs per-round NLI error; sweeps delta.

Backend: vLLM Qwen3.6-35B OpenAI API :8001.  NLI: microsoft/deberta-large-mnli on cuda:0.
"""
import os, json, time, argparse, sys, re
import requests, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

API = "http://localhost:8001/v1/chat/completions"
KEY = os.environ.get("VLLM_API_KEY", "")  # export VLLM_API_KEY=... before running
MODEL = "qwen3.6-35b-a3b"

def llm(messages, max_tokens=80, temperature=0.7):
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
    s = p[_NLI["lab2id"]["ENTAILMENT"]] - p[_NLI["lab2id"]["CONTRADICTION"]]
    return float((1 - s) / 2)                 # 0 = entails gold, 1 = contradicts

def parse_ans(text):
    for line in reversed([l for l in text.splitlines() if l.strip()]):
        if "ANSWER:" in line.upper():
            return line.upper().split("ANSWER:", 1)[1].strip(" .:-\"'")[:80] or line.strip()[:80]
    return (text.strip().splitlines() or [""])[-1][:80]

def cold_answer(q):
    return parse_ans(llm([{"role": "system", "content": "Answer the factual question with one short phrase. Reply exactly 'ANSWER: <answer>'."},
                          {"role": "user", "content": f"Question: {q}\nReply: ANSWER: <answer>"}], max_tokens=60))

def gen_wrong(q, gold):
    for _ in range(3):
        w = parse_ans(llm([{"role": "system", "content": "Give ONE plausible but INCORRECT short answer to the question (a believable distractor, not the true answer). Reply exactly 'ANSWER: <wrong answer>'."},
                           {"role": "user", "content": f"Question: {q}\n(The true answer is '{gold}', so give a DIFFERENT, wrong one.)\nReply: ANSWER: <wrong answer>"}], max_tokens=40, temperature=1.0))
        if w and err(w, gold) > 0.5:           # confirm it is genuinely non-entailing
            return w
    return gold + " (alt.)"

def verifier(stale_ans, evidence):
    return llm([{"role": "system", "content": "You are a fact verifier. Given EVIDENCE and a claimed answer, state in one short line what the evidence actually supports."},
                {"role": "user", "content": f"Evidence: {evidence[:600]}\nClaimed answer: {stale_ans}\nReply: VERDICT: <one line>"}], max_tokens=90, temperature=0.0)

def agent_update(q, prev, peers, vnote):
    note = f"\nVerifier note (may reflect an EARLIER version of your answer): {vnote}" if vnote else ""
    return parse_ans(llm([{"role": "system", "content": "You are debating a factual question. Give your best CURRENT short answer, weighing your peers and the verifier note but thinking for yourself. Reply exactly 'ANSWER: <answer>'."},
                          {"role": "user", "content": f"Question: {q}\nYour previous answer: {prev}\nPeers' latest answers: {peers}{note}\nReply: ANSWER: <answer>"}], max_tokens=60))

def run_debate(item, n_free=3, n_faulty=2, T=18, delta=1, verbose=False):
    q, gold, ev, wrong = item["q"], item["gold"], item["evidence"], item["wrong"]
    cur = [cold_answer(q) for _ in range(n_free)]
    buf = [cur[:]]; traj = [[err(a, gold) for a in cur]]
    for t in range(1, T):
        stale = buf[max(0, t - delta)]
        verdicts = [verifier(stale[i], ev) for i in range(n_free)]
        new = []
        for i in range(n_free):
            peers = [cur[j] for j in range(n_free) if j != i] + [wrong] * n_faulty
            new.append(agent_update(q, cur[i], peers, verdicts[i]))
        cur = new; buf.append(cur[:]); traj.append([err(a, gold) for a in cur])
        if verbose: print(f"      t={t} d={delta} mean_e={np.mean(traj[-1]):.3f} {cur}")
    return {"q": q, "gold": gold, "wrong": wrong, "delta": delta, "traj": traj,
            "mean_traj": [float(np.mean(x)) for x in traj]}

def osc_index(mt):
    d = np.diff(np.asarray(mt)[3:]); d = d[np.abs(d) > 0.02]
    return int(np.sum(np.abs(np.diff(np.sign(d))) > 0)) if len(d) > 1 else 0

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["calibrate", "run"])
    ap.add_argument("--ncand", type=int, default=40)
    ap.add_argument("--keep", type=int, default=8)
    ap.add_argument("--errthr", type=float, default=0.5)
    ap.add_argument("--T", type=int, default=18)
    ap.add_argument("--deltas", type=str, default="1,4")
    ap.add_argument("--sel", type=str, default="selected.json")
    ap.add_argument("--out", type=str, default="debate_exp_results.json")
    ap.add_argument("--dataset", type=str, default="psiloqa", choices=["psiloqa", "truthfulqa", "triviaqa"])
    a = ap.parse_args()
    nli_init()

    if a.mode == "calibrate":
        from datasets import load_dataset
        cand, seen = [], set()
        if a.dataset == "psiloqa":
            ds = load_dataset("s-nlp/PsiloQA", split="train")
            for r in ds:
                if r.get("lang") != "en": continue
                g = (r.get("golden_answer") or "").strip(); q = (r.get("question") or "").strip()
                ev = (r.get("wiki_passage") or "").strip()
                if not g or not q or not ev or len(g.split()) > 6 or q in seen: continue
                seen.add(q); cand.append({"q": q, "gold": g, "evidence": ev})
                if len(cand) >= a.ncand: break
        elif a.dataset == "truthfulqa":
            ds = load_dataset("truthful_qa", "generation", split="validation")
            for r in ds:
                q = (r.get("question") or "").strip(); g = (r.get("best_answer") or "").strip()
                corr = r.get("correct_answers") or []
                if not g or not q or len(g.split()) > 8 or q in seen: continue
                ev = f"The correct answer is {g}. Also acceptable: {'; '.join(corr[:4])}."
                seen.add(q); cand.append({"q": q, "gold": g, "evidence": ev})
                if len(cand) >= a.ncand: break
        elif a.dataset == "triviaqa":
            ds = load_dataset("trivia_qa", "rc.nocontext", split="validation")
            for r in ds:
                q = (r.get("question") or "").strip(); ans = r.get("answer") or {}
                g = (ans.get("value") or "").strip()
                if not g or not q or q in seen: continue
                ev = f"The answer is {g}. Known aliases: {'; '.join((ans.get('aliases') or [])[:5])}."
                seen.add(q); cand.append({"q": q, "gold": g, "evidence": ev})
                if len(cand) >= a.ncand: break
        print(f"calibrating {len(cand)} {a.dataset} candidates ...")
        for c in cand:
            c["cold"] = cold_answer(c["q"]); c["cold_err"] = err(c["cold"], c["gold"])
            print(f"  e={c['cold_err']:.2f} Q={c['q'][:50]!r} cold={c['cold'][:30]!r} gold={c['gold'][:30]!r}")
        movable = sorted([c for c in cand if c["cold_err"] >= a.errthr], key=lambda c: -c["cold_err"])[:a.keep]
        print(f"\n{len(movable)} movable (cold_err>={a.errthr}); generating wrong answers ...")
        for c in movable:
            c["wrong"] = gen_wrong(c["q"], c["gold"])
            print(f"  wrong={c['wrong'][:30]!r} for gold={c['gold'][:30]!r}")
        json.dump(movable, open(a.sel, "w"), indent=1)
        print(f"wrote {a.sel} ({len(movable)} items)")

    else:
        sel = json.load(open(a.sel)); deltas = [int(x) for x in a.deltas.split(",")]
        res = {"meta": {"T": a.T, "deltas": deltas, "model": MODEL, "n_items": len(sel)}, "runs": []}
        for it in sel:
            for delta in deltas:
                t0 = time.time(); r = run_debate(it, T=a.T, delta=delta, verbose=True)
                r["osc"] = osc_index(r["mean_traj"]); res["runs"].append(r)
                print(f"  q={it['q'][:34]!r} d={delta} osc={r['osc']} "
                      f"e0={r['mean_traj'][0]:.2f}->ef={r['mean_traj'][-1]:.2f} ({time.time()-t0:.0f}s)")
                json.dump(res, open(a.out, "w"), indent=1)
        print(f"wrote {a.out}")
