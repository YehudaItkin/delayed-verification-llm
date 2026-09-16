from pathlib import Path
import json, sys
packet = json.loads((Path(__file__).parent/'packet.json').read_text())
for q in packet[int(sys.argv[1])-1:int(sys.argv[2])]:
    print(f"\nQ{q['number']}: {q['question']}\nREF: {q['reference']}\nACCEPT: {' | '.join(q['accepted'])}\nREJECT: {' | '.join(q['rejected'])}")
    if '--evidence' in sys.argv: print('EVIDENCE:', q['evidence'])
    for a in q['answers']: print(f"{a['number']}: {a['text']}")
