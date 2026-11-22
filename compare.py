import json, sys, collections
def load(p):
    d = {}
    for l in open(p):
        r = json.loads(l)
        if "error" not in r:
            d[(r["video"], r["second"])] = r
    return d
ref, new = load(sys.argv[1]), load(sys.argv[2])
C = collections.defaultdict(lambda: [0, 0, 0])  # tp fp fn
ex = w1 = 0
for k, r in ref.items():
    n = new.get(k, {"objects": [], "people": 0})
    a, b = set(r["objects"]), set(n["objects"])
    for c in a | b:
        i = 0 if (c in a and c in b) else (1 if c in b else 2)
        C[c][i] += 1
    d = abs(r["people"] - n["people"]); ex += d == 0; w1 += d <= 1
N = len(ref)
tp = sum(v[0] for v in C.values()); fp = sum(v[1] for v in C.values()); fn = sum(v[2] for v in C.values())
f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0
print(f"{sys.argv[2]}: F1={f1:.4f} exact={ex/N:.4f} within1={w1/N:.4f} | " +
      " ".join(f"{c}:{v[0]}/{v[1]}/{v[2]}" for c, v in sorted(C.items())))
