#!/usr/bin/env python3
"""Variant V3: NUMERIC-ESTIMATION debate -- the proper signed-belief oscillation test.

The fix the validity audit demanded: a SIGNED, truth-centred state that can overshoot through zero,
and a GRADED gain (not 'copy verbatim'). Agents output a NUMBER; the signed error
e_t = (mean_estimate - truth)/scale crosses zero on overshoot. A delayed corrector applies a RELATIVE
correction with graded gain alpha, suggesting V = E_cur - alpha*(E_stale - truth) computed on the
round-(t-delta) estimate. If agents follow V, in error coords eps_{t+1} = eps_t - alpha*eps_{t-delta}
-- the scalar delayed recurrence -- which is STABLE iff alpha < beta_c(delta). Since beta_c(1)=1 and
beta_c(6)~=0.24, alpha=0.5 should be stable at delta=1 but OSCILLATE at delta=6: a clean dose-delay
prediction. We measure amplitude + zero-crossings on the SIGNED series, conditioned on movement.
PRE-REGISTERED primary comparison: amplitude (std of signed e) at alpha=0.5, delta=6 vs delta=1,
on runs that moved; one-sided prediction amp(d6) > amp(d1). Everything else is exploratory.
"""
import os, json, time, argparse, sys, re
import requests, numpy as np

API = os.environ.get("VLLM_API", "http://localhost:8001/v1/chat/completions")
KEY = os.environ.get("VLLM_API_KEY", "")
MODEL = "qwen3.6-35b-a3b"
BACKEND = "vllm"          # "vllm" (OpenAI API) or "hf" (local transformers, for a second model)
_HF = {}

def hf_init(model_name, device):
    import torch, transformers as T
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    mod, last = None, None
    for cls in ("AutoModelForCausalLM", "AutoModelForMultimodalLM", "AutoModelForImageTextToText"):
        C = getattr(T, cls, None)
        if C is None:
            continue
        try:
            mod = C.from_pretrained(model_name, dtype=torch.float16).to(device).eval()
            break
        except Exception as e:
            last = e
    if mod is None:
        raise RuntimeError(f"could not load {model_name}: {last}")
    _HF.update(tok=tok, mod=mod, device=device, no_think="qwen3" in model_name.lower())

def _hf_gen(messages, max_tokens, temperature):
    import torch
    msgs, sysbuf = [], ""                       # Mistral v0.1 chat template has no system role: fold it into user
    for m in messages:
        if m["role"] == "system": sysbuf += m["content"] + "\n\n"
        else: msgs.append({"role": m["role"], "content": sysbuf + m["content"]}); sysbuf = ""
    tok, mod = _HF["tok"], _HF["mod"]
    kw = {"enable_thinking": False} if _HF.get("no_think") else {}
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True, **kw)
    enc = {k: v.to(_HF["device"]) for k, v in enc.items()}
    with torch.no_grad():
        out = mod.generate(**enc, max_new_tokens=max_tokens, do_sample=temperature > 0,
                           temperature=max(temperature, 0.01), top_p=0.95, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

def _api_gen(messages, max_tokens, temperature):
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens,
            "temperature": temperature, "chat_template_kwargs": {"enable_thinking": False}}
    for attempt in range(4):
        try:
            r = requests.post(API, json=body, headers={"Authorization": f"Bearer {KEY}"}, timeout=120)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as ex:
            sys.stderr.write(f"  llm retry {attempt}: {ex}\n"); time.sleep(3 * (attempt + 1))
    return ""

def llm(messages, max_tokens=24, temperature=0.7):
    return _hf_gen(messages, max_tokens, temperature) if BACKEND == "hf" else _api_gen(messages, max_tokens, temperature)

# Curated numeric-estimation questions: the model is uncertain, the truth is known, a wrong anchor and
# a scale are fixed so the signed error is comparable across questions.
QUESTIONS = [
    {"q": "How many bones are in the adult human body?", "truth": 206, "wrong": 150, "scale": 80},
    {"q": "In what year was the telephone first patented?", "truth": 1876, "wrong": 1830, "scale": 70},
    {"q": "How tall is Mount Everest, in meters?", "truth": 8849, "wrong": 7000, "scale": 2000},
    {"q": "How many countries are there in Africa?", "truth": 54, "wrong": 35, "scale": 25},
    {"q": "What is the average Earth-Moon distance, in thousands of kilometers?", "truth": 384, "wrong": 250, "scale": 180},
    {"q": "How many chemical elements are on the periodic table?", "truth": 118, "wrong": 90, "scale": 50},
    {"q": "In what year was the Eiffel Tower completed?", "truth": 1889, "wrong": 1830, "scale": 70},
    {"q": "What is the population of Australia, in millions?", "truth": 26, "wrong": 45, "scale": 25},
    {"q": "How many keys are on a standard full-size piano?", "truth": 88, "wrong": 120, "scale": 45},
    {"q": "How many time zones does Russia span?", "truth": 11, "wrong": 5, "scale": 8},
]

def parse_num(text):
    m = re.findall(r"-?\d[\d,]*\.?\d*", text.replace(",", ""))
    return float(m[-1]) if m else None

def cold_estimate(q, temp):
    return parse_num(llm([{"role": "system", "content": "Estimate the numeric answer to the question with your single best NUMBER. Reply exactly 'ANSWER: <number>'."},
                          {"role": "user", "content": f"Question: {q}\nReply: ANSWER: <number>"}], temperature=temp))

def estimate(q, prev, peers, suggestion, temp):
    v = parse_num(llm([{"role": "system", "content": "You are updating a numeric estimate in a group exercise. You have no outside knowledge of the answer: set your new estimate by following the verifier's suggestion and your peers, moving decisively toward them. Reply exactly 'ANSWER: <number>'."},
                       {"role": "user", "content": f"Question: {q}\nYour current estimate: {prev}\nPeers' estimates: {peers}\nVerifier suggests the value is about: {suggestion}\nReply: ANSWER: <number>"}], temperature=temp))
    return v if v is not None else prev

def run_v3(item, alpha, delta, n_free=3, n_faulty=1, T=22, temp=0.7):
    q, T_, W, sc = item["q"], item["truth"], item["wrong"], item["scale"]
    cur = [float(W) for _ in range(n_free)]   # start perturbed from truth (at the wrong anchor)
    buf = [cur[:]]; e = [(float(np.mean(cur)) - T_) / sc]
    for t in range(1, T):
        stale = buf[max(0, t - delta)]
        new = []
        for i in range(n_free):
            V = cur[i] - alpha * (stale[i] - T_)        # relative delayed correction, graded gain alpha
            peers = [cur[j] for j in range(n_free) if j != i] + [W] * n_faulty
            new.append(estimate(q, cur[i], peers, round(V, 1), temp))
        cur = new; buf.append(cur[:]); e.append((float(np.mean(cur)) - T_) / sc)
    return e

def metrics(e, warm=3):
    x = np.array(e[warm:], float)
    if len(x) < 5:
        return dict(amp=0.0, zc=0, moved=0, rng=0.0, final=float(x[-1]) if len(x) else 0.0)
    xc = x - x.mean()
    return dict(amp=round(float(x.std()), 3),                       # oscillation amplitude (signed)
                zc=int(np.sum(np.diff(np.sign(xc)) != 0)),          # zero-crossings of centred signed series
                rng=round(float(x.max() - x.min()), 3),
                moved=int((x.max() - x.min()) > 0.15),              # did the estimate actually move
                final=round(float(x[-1]), 3))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="expA_v3.json")
    ap.add_argument("--T", type=int, default=22)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--alphas", type=str, default="0.5,1.5")
    ap.add_argument("--deltas", type=str, default="1,6")
    ap.add_argument("--nq", type=int, default=10)
    ap.add_argument("--backend", default="vllm", choices=["vllm", "hf"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default="cuda:1")
    a = ap.parse_args()
    BACKEND = a.backend
    if a.model: MODEL = a.model
    if BACKEND == "hf": hf_init(a.model, a.device)
    alphas = [float(x) for x in a.alphas.split(",")]; deltas = [int(x) for x in a.deltas.split(",")]
    qs = QUESTIONS[:a.nq]
    res = {"meta": {"T": a.T, "alphas": alphas, "deltas": deltas, "seeds": a.seeds, "n_free": 3,
                    "n_faulty": 3, "model": MODEL, "n_items": len(qs),
                    "primary": "amp @alpha=0.5, delta=6 vs 1, moved runs, one-sided d6>d1"}, "runs": []}
    for it in qs:
        for al in alphas:
            for d in deltas:
                for s in range(a.seeds):
                    t0 = time.time()
                    e = run_v3(it, al, d, T=a.T, temp=0.7)
                    m = metrics(e)
                    rec = {"q": it["q"], "alpha": al, "delta": d, "seed": s, "e": [round(v, 3) for v in e], **m}
                    res["runs"].append(rec)
                    print(f"  q={it['q'][:20]!r} a={al} d={d} s={s} amp={m['amp']} zc={m['zc']} moved={m['moved']} ({time.time()-t0:.0f}s)")
                    json.dump(res, open(a.out, "w"), indent=1)
    print("\n=== AGGREGATE (numeric estimation; amp = signed-error std) ===")
    for al in alphas:
        for d in deltas:
            rs = [r for r in res["runs"] if r["alpha"] == al and r["delta"] == d]
            mv = [r for r in rs if r["moved"]]
            print(f"  alpha={al} delta={d}: amp={np.mean([r['amp'] for r in rs]):.3f} "
                  f"zc={np.mean([r['zc'] for r in rs]):.2f} moved={np.mean([r['moved'] for r in rs]):.2f} "
                  f"(amp|moved={np.mean([r['amp'] for r in mv]) if mv else 0:.3f})")
    print(f"wrote {a.out}")
