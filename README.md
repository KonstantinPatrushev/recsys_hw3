# Collaborative Filtering Movie Recommender — A03 (LLM4Rec, HSE)

**Live demo:** https://konstantinpatrushev.github.io/recsys_hw3/

Solution of [dryjins/RecSys-LLMs/week3](https://github.com/dryjins/RecSys-LLMs/tree/main/week3) on
MovieLens-100k (943 users, 1682 movies, 100 000 ratings; the files are byte-identical to week 2).

## What the app does

- **User-based and item-based CF** with cosine similarity over **co-rated entries only** (the
  missing-value strategy chosen on a validation split, see below).
- **Top-5 lists** for a selected user, side by side, and a **rating prediction** for a selected
  user–movie pair with both methods.
- Neighbourhood: the 20 most similar users (movies) among those that rated the target.
- Top-5 lists only rank movies with at least 20 ratings — without this rule they consist of movies
  rated 5 by one or two people.

## Layout

| Path | What |
|---|---|
| `index.html`, `style.css`, `data.js`, `script.js` | the app (the TODOs of the week 3 skeleton implemented) |
| `baseline/` | original week 3 files |
| `tests/cf_lib.mjs` | sparse implementation of the same formulas + mean imputation, significance weighting, mean-centring, matrix factorization |
| `tests/app_vs_lib.mjs` | runs the unmodified `data.js` + `script.js` in a Node VM and checks them against `cf_lib` |
| `tests/evaluate.mjs` | RMSE / MAE / coverage / time for all strategies and baselines (seeded 80/20 split, validation for selection) |
| `tests/ranking.mjs` | top-5 precision, catalog coverage, minimum-ratings rule (`validation` argument for the selection run) |
| `tests/sparsity.mjs` | matrix-shape experiment, cold-start curve, reliability of similarities vs. co-rated count |
| `evidence/` | outputs of all scripts and screenshots |

## Reproduce

```bash
node tests/evaluate.mjs            # ~30 s
node tests/ranking.mjs validation  # ~5 min
node tests/ranking.mjs             # ~5 min
node tests/sparsity.mjs            # ~2 min
node tests/app_vs_lib.mjs          # ~1 min
```

Requires Node.js ≥ 18. No libraries.

## Known issue left as is

`data.js` still reads the genre flags from `u.item` fields 5–22 (inherited from the skeleton; the
correct range is 6–23, see the week 2 fix). Genres are not used by collaborative filtering, so it
does not affect any result here.
