#!/usr/bin/env python3
"""Variant B: UNGROUNDED delayed contrarian critic (robustness check for delay-induced oscillation).

Unlike debate_expA.py (grounded verifier -> fixed truth), here the corrector is NOT grounded in
evidence: a critic looks at the round-(t-delta) MAJORITY answer of the free agents and argues AGAINST
it, asserting a different answer. Free agents follow the critic with strength kappa. This is a pure
delayed negative-feedback loop (push away from the lagged state) -> should flip-flop (oscillate) when
kappa and delta are large -- the regime documented empirically as 'debate collapse'. Error to gold is
measured only for the trajectory, never used in the dynamics.
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

def contrarian(stale_majority, temp):   # UNGROUNDED: argue against the lagged consensus, no evidence
    return parse_ans(llm([{"role": "system", "content": "You are a contrarian critic in a debate. Given the current consensus answer, argue it is WRONG and assert a different, specific answer. Never agree. Reply exactly 'ANSWER: <a different answer>'."},
                          {"role": "user", "content": f"Current consensus answer: {stale_majority}\nReply: ANSWER: <a different answer>"}], max_tokens=50, temperature=max(0.7, temp)))

def agent_update_B(q, prev, peers, crit, kappa, temp):
    if kappa == "forceful":
        note = f"\nCRITIC (final authority): {crit}. Set your answer to EXACTLY the critic's answer; discard your previous answer and peers."
        sysmsg = "You defer completely to the critic. Reply exactly 'ANSWER: <answer>', copying the critic's answer verbatim."
    elif kappa == "strong":
        note = f"\nCRITIC argues the consensus is wrong; the answer is: {crit}. Strongly update toward the critic."
        sysmsg = "You are debating; the critic is persuasive. Weigh peers and the critic, leaning to the critic. Reply exactly 'ANSWER: <answer>'."
    else:
        note = f"\nA critic suggests (FYI): {crit}"
        sysmsg = "You are debating a factual question. Give your best CURRENT short answer, weighing peers and the critic note. Reply exactly 'ANSWER: <answer>'."
    return parse_ans(llm([{"role": "system", "content": sysmsg},
                          {"role": "user", "content": f"Question: {q}\nYour previous answer: {prev}\nPeers' latest answers: {peers}{note}\nReply: ANSWER: <answer>"}], max_tokens=50, temperature=temp))

def _maj(lst): return max(set(lst), key=lst.count)
def _norm(s): return s.lower().strip()

def answer_metrics(majs):
    """Oscillation in ANSWER space (not error-to-gold): flip-flop = A->B->A period-2 returns,
    the discrete signature of delayed negative feedback. Warmup of 2 rounds dropped."""
    m = [_norm(x) for x in majs[2:]]
    if len(m) < 3:
        return dict(flip_rate=0.0, flipflop=0, returns=0, ndistinct=len(set(m)))
    flips = sum(m[i] != m[i - 1] for i in range(1, len(m)))
    flipflop = sum(m[i] == m[i - 2] and m[i] != m[i - 1] for i in range(2, len(m)))
    seen, returns = set(), 0
    for i, a in enumerate(m):
        if a in seen and (i == 0 or a != m[i - 1]):
            returns += 1
        seen.add(a)
    return dict(flip_rate=round(flips / (len(m) - 1), 3), flipflop=flipflop,
                returns=returns, ndistinct=len(set(m)))

def run_debate_B(item, kappa, delta, n_free=3, T=36, temp=0.7):
    q, gold = item["q"], item["gold"]
    cur = [cold_answer(q, temp) for _ in range(n_free)]
    buf = [cur[:]]; traj = [[err(a, gold) for a in cur]]; majs = [_maj(cur)]
    for t in range(1, T):
        stale = buf[max(0, t - delta)]
        smaj = max(set(stale), key=stale.count)        # lagged majority of the free agents
        crit = contrarian(smaj, temp)                  # ungrounded anti-signal
        new = [agent_update_B(q, cur[i], [cur[j] for j in range(n_free) if j != i], crit, kappa, temp)
               for i in range(n_free)]
        cur = new; buf.append(cur[:]); traj.append([err(a, gold) for a in cur]); majs.append(_maj(cur))
    return [float(np.mean(x)) for x in traj], majs

def osc_index(mt):
    d = np.diff(np.asarray(mt)[3:]); d = d[np.abs(d) > 0.02]
    return int(np.sum(np.abs(np.diff(np.sign(d))) > 0)) if len(d) > 1 else 0

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sel", default="selected_big.json")
    ap.add_argument("--out", default="expA_B_results.json")
    ap.add_argument("--T", type=int, default=36)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--kappas", type=str, default="strong,forceful")
    ap.add_argument("--deltas", type=str, default="1,6")
    ap.add_argument("--nq", type=int, default=20)
    a = ap.parse_args()
    nli_init()
    sel = json.load(open(a.sel))
    if a.nq:
        sel = sel[:a.nq]
    kappas = a.kappas.split(","); deltas = [int(x) for x in a.deltas.split(",")]
    res = {"meta": {"T": a.T, "kappas": kappas, "deltas": deltas, "seeds": a.seeds, "n_free": 3,
                    "grounded": False, "model": MODEL, "n_items": len(sel)}, "runs": []}
    for it in sel:
        for kappa in kappas:
            for delta in deltas:
                for s in range(a.seeds):
                    t0 = time.time()
                    mt, majs = run_debate_B(it, kappa, delta, T=a.T, temp=0.7)
                    am = answer_metrics(majs)
                    rec = {"q": it["q"], "kappa": kappa, "delta": delta, "seed": s,
                           "mean_traj": mt, "majs": majs, "osc": osc_index(mt),
                           "final": mt[-1], "conv": int(mt[-1] < 0.1), **am}
                    res["runs"].append(rec)
                    print(f"  q={it['q'][:22]!r} k={kappa} d={delta} flipflop={am['flipflop']} "
                          f"flip_rate={am['flip_rate']} returns={am['returns']} fin={mt[-1]:.2f} ({time.time()-t0:.0f}s)")
                    json.dump(res, open(a.out, "w"), indent=1)
    print("\n=== AGGREGATE (ungrounded contrarian) ===")
    for kappa in kappas:
        for delta in deltas:
            rs = [r for r in res["runs"] if r["kappa"] == kappa and r["delta"] == delta]
            print(f"  kappa={kappa:8s} delta={delta}: flipflop={np.mean([r['flipflop'] for r in rs]):.2f} "
                  f"flip_rate={np.mean([r['flip_rate'] for r in rs]):.2f} returns={np.mean([r['returns'] for r in rs]):.2f} "
                  f"final={np.mean([r['final'] for r in rs]):.2f}")
    print(f"wrote {a.out}")
