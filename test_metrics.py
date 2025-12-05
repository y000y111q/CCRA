from modules.metrics import compute_scores

gts = {0: ["the heart is normal ."]}
res = {0: ["the heart is normal ."]}

scores = compute_scores(gts, res)
print(scores)
