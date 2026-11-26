# Unburden AI: Landing Page + Word2Vec Explorer

This is a static site for CAI2300C. It has two parts:

- **`/`**: the landing page for Unburden AI, a fictional company that turns a small business's SOPs into AI systems. It uses the Oxblood & Steel Glass design, with light and dark themes, an EN/ES toggle, and a Time-Back calculator.
- **`/word2vec/`**: a Word2Vec Explorer. Visitors can look up nearest neighbors by cosine similarity, view 2D and 3D PCA maps of a word's neighborhood, and solve analogies (B − A + C). It also has a plain-language "How it works" section.

## The Word2Vec model

The model was trained with **gensim** using skip-gram with negative sampling, on the **text8** corpus (about 17 million words of cleaned English Wikipedia).

| Setting | Value |
|---|---|
| Vector size | 100 |
| Window | 5 |
| Min count | 5 |
| Negative samples | 5 |
| Epochs | 5 |

The 10,000 most frequent words are exported to `word2vec/model.json`. The vectors are normalised to unit length and stored as int8 in base64, which keeps the file at about 1.3 MB. All of the math runs in the browser.

To reproduce the model, download the text8 corpus into `training/` and run these commands from that folder:

```bash
pip install gensim
python train.py      # trains and saves text8_w2v.model
python export.py     # writes model.json (copy it into word2vec/)
```

## Deploy

This is plain HTML, CSS, and JS, so there's no build step. Netlify publishes the repo root, as set in `netlify.toml`.
