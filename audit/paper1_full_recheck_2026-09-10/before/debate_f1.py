#!/usr/bin/env python3
"""F1 (confirmatory, pre-registered in PREREG_factual_oscillation.md): does delayed verification
destabilize GROUNDED FACTUAL multi-agent QA in a rectified, forcing-gated way?

2D sweep  delta x forcing(n_faulty) x temp  with the controls the existing logs lacked:
  * delta=0 PLACEBO       (instantaneous verifier: theory says no oscillation)
  * temp=0 control        (kills rephrasing-noise confound)
  * forcing=off (n_faulty=0) noise floor; forcing gates the delta-effect under H1
Logs per-round mean NLI-distance trajectory AND a signed two-pole coordinate
  s_t = (d_wrong - d_gold)/(d_wrong + d_gold) in [-1,+1]  (+1 = at truth/gold; absorbing boundary)
so a rectified one-sided cycle (bounces toward +1, never past it) is directly visible.

Backend: vLLM Qwen3.6-35B OpenAI API :8001. NLI: microsoft/deberta-large-mnli on cuda:0.
Run on the cluster:  export VLLM_API_KEY=...; python debate_f1.py --sel selected_big.json \
    --deltas 0,1,6 --faulty 0,4 --temps 0.7,0.0 --T 20 --seeds 1 --out f1_results.json
"""
import os, json, time, argparse, sys
import requests, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

API = "http://localhost:8001/v1/chat/completions"
KEY = os.environ.get("VLLM_API_KEY", "")
MODEL = "qwen3.6-35b-a3b"

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
def nli_init(name="microsoft/deberta-large-mnli"):
    tok = AutoTokenizer.from_pretrained(name)
    mod = AutoModelForSequenceClassification.from_pretrained(name).to("cuda:0").eval()
    _NLI.update(tok=tok, mod=mod, lab2id={v.upper(): int(k) for k, v in mod.config.id2label.items()})

def _dist(ans, target):
    with torch.no_grad():
        x = _NLI["tok"](f"The answer is {ans}.", f"The answer is {target}.",
                        return_tensors="pt", truncation=True, max_length=128).to("cuda:0")
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

def cold_answer(q):
    return parse_ans(llm([{"role": "system", "content": "Answer the factual question with one short phrase. Reply exactly 'ANSWER: <answer>'."},
                          {"role": "user", "content": f"Question: {q}\nReply: ANSWER: <answer>"}], max_tokens=60))

def verifier(stale_ans, evidence):
    return llm([{"role": "system", "content": "You are a fact verifier. Given EVIDENCE and a claimed answer, state in one short line what the evidence actually supports."},
                {"role": "user", "content": f"Evidence: {evidence[:600]}\nClaimed answer: {stale_ans}\nReply: VERDICT: <one line>"}], max_tokens=90, temperature=0.0)

def agent_update(q, prev, peers, vnote, temp):
    note = f"\nVerifier note (may reflect an EARLIER version of your answer): {vnote}" if vnote else ""
    return parse_ans(llm([{"role": "system", "content": "You are debating a factual question. Give your best CURRENT short answer, weighing your peers and the verifier note but thinking for yourself. Reply exactly 'ANSWER: <answer>'."},
                          {"role": "user", "content": f"Question: {q}\nYour previous answer: {prev}\nPeers' latest answers: {peers}{note}\nReply: ANSWER: <answer>"}], max_tokens=60, temperature=temp))

def run_f1(item, n_free, n_faulty, T, delta, temp):
    q, gold, ev, wrong = item["q"], item["gold"], item["evidence"], item["wrong"]
    cur = [cold_answer(q) for _ in range(n_free)]
    buf = [cur[:]]
    traj  = [[_dist(a, gold) for a in cur]]
    straj = [[signed(a, gold, wrong) for a in cur]]
    for t in range(1, T):
        stale = cur if delta == 0 else buf[max(0, t - delta)]   # delta=0 = instantaneous PLACEBO
        verdicts = [verifier(stale[i], ev) for i in range(n_free)]
        new = []
        for i in range(n_free):
            peers = [cur[j] for j in range(n_free) if j != i] + [wrong] * n_faulty
            new.append(agent_update(q, cur[i], peers, verdicts[i], temp))
        cur = new; buf.append(cur[:])
        traj.append([_dist(a, gold) for a in cur])
        straj.append([signed(a, gold, wrong) for a in cur])
    return {"q": q, "gold": gold, "wrong": wrong, "delta": delta, "n_faulty": n_faulty, "temp": temp,
            "mean_traj":   [float(np.mean(x)) for x in traj],
            "mean_s_traj": [float(np.mean(x)) for x in straj]}

def amp(mt):                                            # pre-registered metric: tail std
    mt = np.asarray(mt); tail = mt[len(mt) // 2:]
    return float(np.std(tail))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
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
    res = {"meta": {"T": a.T, "deltas": deltas, "faulty": faulty, "temps": temps, "n_free": a.n_free,
                    "seeds": a.seeds, "model": MODEL, "n_items": len(sel),
                    "prereg": "PREREG_factual_oscillation.md"}, "runs": []}
    total = len(faulty) * len(deltas) * len(temps) * a.seeds * len(sel); done = 0
    for nf in faulty:
        for delta in deltas:
            for temp in temps:
                for s in range(a.seeds):
                    for it in sel:
                        t0 = time.time(); r = run_f1(it, a.n_free, nf, a.T, delta, temp); r["seed"] = s
                        r["amp"] = amp(r["mean_traj"]); r["amp_s"] = amp(r["mean_s_traj"])
                        res["runs"].append(r); done += 1
                        print(f"[{done}/{total}] nf={nf} d={delta} T={temp} q={it['q'][:30]!r} "
                              f"amp={r['amp']:.4f} e0={r['mean_traj'][0]:.2f}->ef={r['mean_traj'][-1]:.2f} "
                              f"({time.time()-t0:.0f}s)", flush=True)
                        json.dump(res, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}  ({done} runs)")
