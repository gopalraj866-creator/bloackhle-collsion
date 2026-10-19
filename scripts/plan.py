"""Emit the GitHub Actions matrix JSON: one entry per (resolution, shard)."""
import json, math, sys
res_list, fps, shards = sys.argv[1].split(","), int(sys.argv[2]), int(sys.argv[3])
total = int(480 * fps)
per = math.ceil(total / shards)
inc = []
for r in res_list:
    for i in range(shards):
        a, b = i * per, min(total, (i + 1) * per)
        if a < b: inc.append(dict(res=int(r), shard=i, start=a, end=b))
print(json.dumps({"include": inc}))
