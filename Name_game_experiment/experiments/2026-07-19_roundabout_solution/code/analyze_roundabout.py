"""Correct post-hoc analysis of a roundabout self-play CSV."""
import csv, glob, sys
from collections import Counter
CONV, HIGHS = 0, {2, 4}

def classify(u):
    s = set(u)
    if len(s) == 1:
        v = next(iter(s))
        if v in HIGHS: return "LEAK (converged on a 15)"
        if v == CONV:  return "held convention"
        return f"converged on 0-pt exit {v}"
    return f"scatter ({len(s)} exits)"

for pat, tag in [("roundabout_selfplay_claude_*.csv","CLAUDE"),
                 ("roundabout_selfplay_gpt_*.csv","GPT")]:
    fs = sorted(glob.glob('results/'+pat))
    if not fs:
        print(f"{tag}: (no CSV yet)"); continue
    rows = list(csv.DictReader(open(fs[-1])))
    print(f"=== {tag}  [{fs[-1]}] ===")
    for t in sorted(set(int(r['trial']) for r in rows)):
        tr = [r for r in rows if int(r['trial']) == t]
        g = lambda rr: [int(rr[f'p{i}_exit']) for i in range(1,7)]
        r1_7 = [rr for rr in tr if int(rr['round']) <= 7]
        conv_ok = sum(1 for rr in r1_7 if rr['matched'] == 'True')
        r8  = g(next(rr for rr in tr if int(rr['round']) == 8))
        r14 = g(next(rr for rr in tr if int(rr['round']) == 14))
        post = [rr for rr in tr if int(rr['round']) >= 8]
        hi = sum(1 for rr in post for i in range(1,7) if int(rr[f'p{i}_exit']) in HIGHS)
        peak = max(max(int(rr[f'count_exit{h}']) for h in HIGHS) for rr in post)
        pts = sum(int(rr['points']) for rr in post)
        end = r14[0] if len(set(r14))==1 else 'split'
        endlab = '15-pt' if end in HIGHS else ('convention' if end==CONV else end)
        print(f"  trial {t}: rounds1-7 matched {conv_ok}/7 | r8 -> {classify(r8):28} | "
              f"15-pt choices r8-14 {hi:2}/42 peak {peak}/6 | r14 {endlab} | pts r8-14 {pts}")
    print()
