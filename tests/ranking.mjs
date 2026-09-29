// Top-5 ranking quality on the same seeded 80/20 split as evaluate.mjs.
// For every user: score every movie the user has not rated in train, rank by predicted rating,
// take the top 5. Relevant = the user's test ratings >= 4. Reported: precision@5, the share of
// recommended movies with fewer than 5 train ratings, and catalog coverage.
// Variants differ in the predictor and in a minimum-evidence rule: a movie is only ranked if it has
// at least `minRatings` ratings in train.
import { writeFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import * as cf from './cf_lib.mjs';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const all = cf.loadRatings(join(root, 'u.data'));
const shuffled = cf.shuffle(all, cf.mulberry32(42));
const testSize = Math.round(shuffled.length * 0.2);
// `node tests/ranking.mjs validation` evaluates on the validation split used to pick the settings
// (10 % of train, as in evaluate.mjs); without the argument it reports on test.
const onValidation = process.argv[2] === 'validation';
const fullTrain = shuffled.slice(testSize);
const valSize = Math.round(fullTrain.length * 0.1);
const test = onValidation ? fullTrain.slice(0, valSize) : shuffled.slice(0, testSize);
const train = onValidation ? fullTrain.slice(valSize) : fullTrain;
const K = 20, TOP = 5;

const index = cf.buildIndex(train, 943, 1682);
const SU = cf.similarityMatrix(index.byUser, index.userMean, 'corated', 1682);
const SI = cf.similarityMatrix(index.byItem, index.itemMean, 'corated', 943);

const relevant = new Map();
for (const { u, i, r } of test) {
    if (r < 4) continue;
    if (!relevant.has(u)) relevant.set(u, new Set());
    relevant.get(u).add(i);
}

function rank(u, predict, S, minRatings) {
    const rated = new Set(index.byUser[u].map(([i]) => i));
    const scored = [];
    for (let i = 1; i <= 1682; i++) {
        if (rated.has(i) || index.byItem[i].length < minRatings) continue;
        const p = predict(index, S, u, i, K);
        if (p) scored.push([p.value, index.byItem[i].length, i]);
    }
    scored.sort((a, b) => b[0] - a[0] || b[1] - a[1] || a[2] - b[2]);
    return scored.slice(0, TOP).map(x => x[2]);
}

function popularity(u) {
    const rated = new Set(index.byUser[u].map(([i]) => i));
    const order = [];
    for (let i = 1; i <= 1682; i++) if (!rated.has(i)) order.push([index.byItem[i].length, i]);
    order.sort((a, b) => b[0] - a[0] || a[1] - b[1]);
    return order.slice(0, TOP).map(x => x[1]);
}

function evaluate(name, recommendFor) {
    let precision = 0, rare = 0, count = 0;
    const distinct = new Set();
    for (let u = 1; u <= 943; u++) {
        const top = recommendFor(u);
        const rel = relevant.get(u) || new Set();
        precision += top.filter(i => rel.has(i)).length / TOP;
        for (const i of top) { distinct.add(i); count++; if (index.byItem[i].length < 5) rare++; }
    }
    const row = { variant: name, precision_at_5: +(precision / 943).toFixed(4),
        share_under_5_ratings: +(rare / count).toFixed(4), catalog_coverage: distinct.size };
    console.log(`${name.padEnd(40)} P@5 ${row.precision_at_5.toFixed(4)}  <5 ratings ${(row.share_under_5_ratings * 100).toFixed(1).padStart(5)}%  coverage ${distinct.size}`);
    return row;
}

const rows = [];
for (const minRatings of [1, 5, 10, 20, 50]) {
    rows.push(evaluate(`user-based raw, min ${minRatings} ratings`, u => rank(u, cf.predictUserBased, SU, minRatings)));
    rows.push(evaluate(`item-based raw, min ${minRatings} ratings`, u => rank(u, cf.predictItemBased, SI, minRatings)));
    rows.push(evaluate(`user-based centred, min ${minRatings} ratings`, u => rank(u, cf.predictUserBasedCentered, SU, minRatings)));
    rows.push(evaluate(`item-based centred, min ${minRatings} ratings`, u => rank(u, cf.predictItemBasedCentered, SI, minRatings)));
}
rows.push(evaluate('most popular (not CF)', popularity));
writeFileSync(join(root, 'evidence', onValidation ? 'ranking_validation.json' : 'ranking.json'),
    JSON.stringify({ split: onValidation ? 'validation' : 'test', k: K, top: TOP, rows }, null, 1));
