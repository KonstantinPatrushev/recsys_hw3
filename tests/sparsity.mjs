// Experiments for the "users vs items" and "cold start / sparsity" questions.
//   A. Matrix shape: the full data (items > users) against random subsets where users > items and
//      where items >> users. Random subsets keep the density roughly constant, so only the shape
//      changes. Reported: similarity-matrix build time and RMSE for both CF directions.
//   B. Cold-start curve: 100 users keep only n of their train ratings; their test ratings are
//      predicted with user-based CF, item-based CF, and the bias baseline.
//   C. Reliability of similarities computed from few co-rated items.
// All with co-rated cosine and k = 20, on seeded splits.
import { writeFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { performance } from 'perf_hooks';
import * as cf from './cf_lib.mjs';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const all = cf.loadRatings(join(root, 'u.data'));
const K = 20;
const out = {};

function split(ratings, seed) {
    const s = cf.shuffle(ratings, cf.mulberry32(seed));
    const n = Math.round(s.length * 0.2);
    return { test: s.slice(0, n), train: s.slice(n) };
}

function rmseOf(index, S, test, predict) {
    const pairs = test.map(({ u, i, r }) => {
        const p = predict(index, S, u, i, K);
        const fb = Number.isNaN(index.userMean[u]) ? index.globalMean : index.userMean[u];
        return [cf.clip(p ? p.value : fb), r];
    });
    return cf.errors(pairs).rmse;
}

function biasBaseline(index, lambda = 10) {
    const mu = index.globalMean;
    const bi = index.byItem.map(l => l.reduce((s, [, r]) => s + r - mu, 0) / (lambda + l.length));
    const bu = index.byUser.map(l => l.reduce((s, [i, r]) => s + r - mu - bi[i], 0) / (lambda + l.length));
    return (u, i) => mu + (bu[u] || 0) + (bi[i] || 0);
}

// ---------- A. matrix shape ----------
function remap(ratings) { // compact ids so that the matrices have exactly the subset's size
    const uid = new Map(), iid = new Map();
    const mapped = ratings.map(({ u, i, r, t }) => {
        if (!uid.has(u)) uid.set(u, uid.size + 1);
        if (!iid.has(i)) iid.set(i, iid.size + 1);
        return { u: uid.get(u), i: iid.get(i), r, t };
    });
    return { mapped, users: uid.size, items: iid.size };
}

function shapeRun(name, ratings) {
    const { mapped, users, items } = remap(ratings);
    const { train, test } = split(mapped, 42);
    const index = cf.buildIndex(train, users, items);
    let t0 = performance.now();
    const SU = cf.similarityMatrix(index.byUser, index.userMean, 'corated', items);
    const userSimMs = performance.now() - t0;
    t0 = performance.now();
    const SI = cf.similarityMatrix(index.byItem, index.itemMean, 'corated', users);
    const itemSimMs = performance.now() - t0;
    const itemMean = cf.errors(test.map(({ u, i, r }) => [cf.clip(Number.isNaN(index.itemMean[i]) ? index.userMean[u] : index.itemMean[i]), r])).rmse;
    const row = {
        scenario: name, users, items, ratings: mapped.length,
        density: +(mapped.length / (users * items)).toFixed(4),
        userPairs: users * (users - 1) / 2, itemPairs: items * (items - 1) / 2,
        userSimMs: Math.round(userSimMs), itemSimMs: Math.round(itemSimMs),
        rmseUserBased: +rmseOf(index, SU, test, cf.predictUserBased).toFixed(4),
        rmseItemBased: +rmseOf(index, SI, test, cf.predictItemBased).toFixed(4),
        rmseItemMean: +itemMean.toFixed(4)
    };
    console.log(row);
    return row;
}

const rand = cf.mulberry32(11);
const itemSubset = new Set(cf.shuffle([...Array(1682).keys()].map(x => x + 1), rand).slice(0, 300));
const userSubset = new Set(cf.shuffle([...Array(943).keys()].map(x => x + 1), rand).slice(0, 300));
out.shape = [
    shapeRun('full: 943 users x 1682 items', all),
    shapeRun('users > items: 943 users x 300 random items', all.filter(x => itemSubset.has(x.i))),
    shapeRun('items >> users: 300 random users x 1682 items', all.filter(x => userSubset.has(x.u)))
];

// ---------- B. cold-start curve ----------
const { train, test } = split(all, 42);
const fullIndex = cf.buildIndex(train, 943, 1682);
const testByUser = new Map();
for (const x of test) { if (!testByUser.has(x.u)) testByUser.set(x.u, []); testByUser.get(x.u).push(x); }
const eligible = [...testByUser.keys()]
    .filter(u => fullIndex.byUser[u].length >= 40 && testByUser.get(u).length >= 5)
    .sort((a, b) => a - b);
const coldUsers = new Set(cf.shuffle(eligible, cf.mulberry32(5)).slice(0, 100));
const coldTest = test.filter(x => coldUsers.has(x.u));
const keepRand = cf.mulberry32(9);
const keepOrder = new Map([...coldUsers].map(u => [u, cf.shuffle(train.filter(x => x.u === u), keepRand)]));

out.coldStart = { users: coldUsers.size, testRatings: coldTest.length, rows: [] };
for (const n of [0, 1, 2, 3, 5, 10, 20]) {
    const kept = new Set();
    for (const u of coldUsers) keepOrder.get(u).slice(0, n).forEach(x => kept.add(x));
    const reduced = train.filter(x => !coldUsers.has(x.u) || kept.has(x));
    const index = cf.buildIndex(reduced, 943, 1682);
    const SU = cf.similarityMatrix(index.byUser, index.userMean, 'corated', 1682);
    const SI = cf.similarityMatrix(index.byItem, index.itemMean, 'corated', 943);
    const coverage = (predict, S) => coldTest.filter(({ u, i }) => predict(index, S, u, i, K)).length / coldTest.length;
    const bias = biasBaseline(index);
    const row = {
        ratingsKept: n,
        rmseUserBased: +rmseOf(index, SU, coldTest, cf.predictUserBased).toFixed(4),
        rmseItemBased: +rmseOf(index, SI, coldTest, cf.predictItemBased).toFixed(4),
        rmseBiasBaseline: +cf.errors(coldTest.map(({ u, i, r }) => [cf.clip(bias(u, i)), r])).rmse.toFixed(4),
        coverageUserBased: +coverage(cf.predictUserBased, SU).toFixed(4),
        coverageItemBased: +coverage(cf.predictItemBased, SI).toFixed(4)
    };
    console.log(row);
    out.coldStart.rows.push(row);
}
out.coldStart.newItems = {
    testRatingsOnItemsWithoutTrainRatings: test.filter(x => fullIndex.byItem[x.i].length === 0).length,
    itemsWithoutTrainRatings: new Set(test.filter(x => fullIndex.byItem[x.i].length === 0).map(x => x.i)).size
};

// ---------- C. similarities from few co-rated items ----------
const SU = cf.similarityMatrix(fullIndex.byUser, fullIndex.userMean, 'corated', 1682);
const buckets = [[1, 1], [2, 2], [3, 5], [6, 10], [11, 20], [21, 50], [51, 1e9]];
const bucketStats = buckets.map(([lo, hi]) => ({ range: hi > 1e8 ? `${lo}+` : lo === hi ? `${lo}` : `${lo}-${hi}`, pairs: 0, sum: 0, ones: 0 }));
for (let a = 1; a <= 943; a++) for (let b = a + 1; b <= 943; b++) {
    const n = SU.co[a * SU.size + b];
    if (n === 0) continue;
    const s = SU.sim[a * SU.size + b];
    const k = buckets.findIndex(([lo, hi]) => n >= lo && n <= hi);
    bucketStats[k].pairs++; bucketStats[k].sum += s; if (s > 1 - 1e-12) bucketStats[k].ones++;
}
out.similarityByCoRated = bucketStats.map(b => ({ coRated: b.range, userPairs: b.pairs,
    meanCosine: +(b.sum / b.pairs).toFixed(4), shareExactlyOne: +(b.ones / b.pairs).toFixed(4) }));
console.log(out.similarityByCoRated);

// Which neighbours does user-based CF actually use, and how does the error depend on it?
const errBuckets = { '1-5': [], '6-20': [], '21+': [] };
let neighboursUsed = 0, weakNeighbours = 0;
for (const { u, i, r } of test) {
    const scored = [];
    for (const [v, rv] of fullIndex.byItem[i]) {
        const s = SU.sim[u * SU.size + v];
        if (s > 0) scored.push([s, rv, v]);
    }
    if (scored.length === 0) continue;
    scored.sort((a, b) => b[0] - a[0] || a[2] - b[2]);
    const top = scored.slice(0, K);
    let num = 0, den = 0, co = 0;
    for (const [s, rv, v] of top) {
        num += s * rv; den += s;
        const n = SU.co[u * SU.size + v];
        co += n; neighboursUsed++; if (n <= 3) weakNeighbours++;
    }
    const meanCo = co / top.length;
    const key = meanCo <= 5 ? '1-5' : meanCo <= 20 ? '6-20' : '21+';
    errBuckets[key].push([cf.clip(num / den), r]);
}
out.neighbours = {
    shareOfNeighboursWith3OrFewerCoRated: +(weakNeighbours / neighboursUsed).toFixed(4),
    rmseByMeanCoRatedOfNeighbours: Object.fromEntries(Object.entries(errBuckets).map(([k, pairs]) =>
        [k, { predictions: pairs.length, rmse: pairs.length ? +cf.errors(pairs).rmse.toFixed(4) : null }]))
};
console.log(out.neighbours);

writeFileSync(join(root, 'evidence', 'sparsity.json'), JSON.stringify(out, null, 1));
