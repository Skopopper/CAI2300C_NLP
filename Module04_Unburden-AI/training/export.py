"""Export the trained gensim model to a compact JSON file for the browser.

Keeps the most frequent N words, L2-normalises each vector, and stores it as
int8 (scaled by 127) in base64 so the file stays around 1.3 MB.
"""
import base64, json, sys
import numpy as np
from gensim.models import Word2Vec

N = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
m = Word2Vec.load('text8_w2v.model')
done = json.load(open('done.json'))

words = m.wv.index_to_key[:N]
vecs = np.stack([m.wv[w] for w in words]).astype(np.float32)
vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
q = np.clip(np.round(vecs * 127), -127, 127).astype(np.int8)

meta = {
    "library": "gensim " + __import__('gensim').__version__,
    "algorithm": "Skip-gram with negative sampling",
    "corpus": "text8 (first 100 MB of cleaned English Wikipedia, ~17 million words)",
    "dimensions": m.vector_size,
    "window": m.window,
    "min_count": m.min_count,
    "negative": m.negative,
    "epochs": m.epochs,
    "full_vocab": len(m.wv),
    "exported_vocab": len(words),
    "training_seconds": done["seconds"],
    "quantization": "int8 (unit vectors x 127)",
}
out = {"meta": meta, "dim": m.vector_size, "words": words,
       "vecs": base64.b64encode(q.tobytes()).decode()}
with open('model.json', 'w') as f:
    json.dump(out, f, separators=(',', ':'))

# Quick sanity check printed for the log
for w in ['king', 'business', 'money', 'system', 'manager']:
    if w in m.wv:
        print(w, [x for x, _ in m.wv.most_similar(w, topn=6)])
print('king - man + woman =', [x for x, _ in m.wv.most_similar(positive=['king', 'woman'], negative=['man'], topn=3)])
