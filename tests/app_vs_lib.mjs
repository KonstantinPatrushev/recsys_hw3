// Runs the unmodified app code (data.js + script.js) in a Node VM context — the same way a browser
// runs two <script> tags in one global scope — and checks it against tests/cf_lib.mjs, which the
// evaluation uses. Both work on the full data set here (no train/test split).
//   1. rating predictions for 500 random (user, movie) pairs must match
//   2. the top-5 lists for 10 users must match
//   3. the closed-form mean-imputation cosine in cf_lib must match a dense computation
// Also prints how long the app code takes per user, for the efficiency discussion.
import vm from 'vm';
import { readFileSync, writeFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { performance } from 'perf_hooks';
import * as cf from './cf_lib.mjs';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const context = vm.createContext({
    console, window: {}, document: { getElementById: () => null },
    fetch: async (path) => ({ ok: true, text: async () => readFileSync(join(root, path), 'latin1') })
});
for (const file of ['data.js', 'script.js']) {
    vm.runInContext(readFileSync(join(root, file), 'utf8'), context, { filename: file });
}
await vm.runInContext('loadData()', context);
const app = name => vm.runInContext(name, context);

const all = cf.loadRatings(join(root, 'u.data'));
const index = cf.buildIndex(all, 943, 1682);
const SU = cf.similarityMatrix(index.byUser, index.userMean, 'corated', 1682);
const SI = cf.similarityMatrix(index.byItem, index.itemMean, 'corated', 943);

// 1. predictions
const rand = cf.mulberry32(3);
let maxDiff = 0, supportMismatch = 0, nullMismatch = 0;
const pairs = 500;
for (let t = 0; t < pairs; t++) {
    const u = 1 + Math.floor(rand() * 943), i = 1 + Math.floor(rand() * 1682);
    const sims = app('userSimilarities')(u);
    for (const [a, b] of [[app('predictUserBased')(u, i, sims), cf.predictUserBased(index, SU, u, i, 20)],
                          [app('predictItemBased')(u, i), cf.predictItemBased(index, SI, u, i, 20)]]) {
        if (!a || !b) { if (!!a !== !!b) nullMismatch++; continue; }
        maxDiff = Math.max(maxDiff, Math.abs(a.score - b.value));
        if (a.support !== b.support) supportMismatch++;
    }
}

// 2. top-5 lists
const titles = new Map(all.map(x => [x.i, null]));
readFileSync(join(root, 'u.item'), 'latin1').split('\n').filter(Boolean)
    .forEach(l => { const f = l.split('|'); titles.set(+f[0], f[1]); });
function libTop5(u, predict, S) {
    const rated = new Set(index.byUser[u].map(([i]) => i));
    const scored = [];
    for (let i = 1; i <= 1682; i++) {
        if (rated.has(i) || index.byItem[i].length < 20) continue;
        const p = predict(index, S, u, i, 20);
        if (p) scored.push([p.value, index.byItem[i].length, i]);
    }
    scored.sort((a, b) => b[0] - a[0] || b[1] - a[1] || a[2] - b[2]);
    return scored.slice(0, 5).map(x => titles.get(x[2]));
}
let listMismatch = 0;
const timings = [];
for (const u of [1, 13, 50, 100, 200, 300, 405, 500, 700, 943]) {
    let t0 = performance.now();
    const ub = app('getUserBasedRecommendations')(u).map(x => x.title);
    const ubMs = performance.now() - t0;
    t0 = performance.now();
    const ib = app('getItemBasedRecommendations')(u).map(x => x.title);
    const ibMs = performance.now() - t0;
    timings.push({ user: u, ratings: index.byUser[u].length, userBasedMs: Math.round(ubMs), itemBasedMs: Math.round(ibMs) });
    if (JSON.stringify(ub) !== JSON.stringify(libTop5(u, cf.predictUserBased, SU))) listMismatch++;
    if (JSON.stringify(ib) !== JSON.stringify(libTop5(u, cf.predictItemBased, SI))) listMismatch++;
}

// 3. mean imputation: closed form vs dense vectors, for 200 random user pairs
let imputeDiff = 0;
for (let t = 0; t < 200; t++) {
    const a = 1 + Math.floor(rand() * 943), b = 1 + Math.floor(rand() * 943);
    const dense = u => { const v = new Array(1682).fill(index.userMean[u]); for (const [i, r] of index.byUser[u]) v[i - 1] = r; return v; };
    const x = dense(a), y = dense(b);
    let dot = 0, nx = 0, ny = 0;
    for (let k = 0; k < 1682; k++) { dot += x[k] * y[k]; nx += x[k] * x[k]; ny += y[k] * y[k]; }
    const devA = index.byUser[a].reduce((s, [, r]) => s + (r - index.userMean[a]) ** 2, 0);
    const devB = index.byUser[b].reduce((s, [, r]) => s + (r - index.userMean[b]) ** 2, 0);
    const [closed] = cf.similarity(index.byUser[a], index.byUser[b], 'impute',
        { meanA: index.userMean[a], meanB: index.userMean[b], devA, devB, length: 1682 });
    imputeDiff = Math.max(imputeDiff, Math.abs(closed - dot / Math.sqrt(nx * ny)));
}

const result = {
    prediction_pairs: pairs, max_abs_difference: maxDiff, support_mismatches: supportMismatch,
    null_mismatches: nullMismatch, top5_list_mismatches: `${listMismatch} of 20`,
    impute_closed_form_max_difference: imputeDiff, app_timings_node: timings
};
writeFileSync(join(root, 'evidence', 'app_vs_lib.json'), JSON.stringify(result, null, 1));
console.log(JSON.stringify(result, null, 1));
