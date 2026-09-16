from pathlib import Path
import json,sys
for q in json.loads((Path(__file__).parent/'packet.json').read_text()):
 if int(sys.argv[1])<=q['number']<=int(sys.argv[2]):
  print(f"\nQ{q['number']}: {q['question']}\nREF: {q['reference']}")
  if '--evidence' in sys.argv: print('SOURCE:',q['evidence'])
  for a in q['answers']: print(f"{a['number']}: {a['text']}")
