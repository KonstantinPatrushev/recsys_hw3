// Offline evaluation of user-based CF, item-based CF, three missing-value strategies and matrix
// factorization on MovieLens-100k.
//
// Split (seeded, reproducible): 20 % of all ratings = test; the remaining 80 % = train.
// The missing-value strategy for the web app is chosen on a validation split carved out of train
// (10 % of it), never on test. Every CF prediction that has no usable neighbour falls back to the
// user's mean rating; "coverage" is the share of test ratings predicted by CF itself.
import { writeFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { performance } from 'perf_hooks';
import * as cf from './cf_lib.mjs';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const K = 20;
const STRATEGIES = ['corated', 'impute', 'weighted'];
const all = cf.loadRatings(join(root, 'u.data'));
const NUM_USERS = 943, NUM_ITEMS = 1682;

const shuffled = cf.shuffle(all, cf.mulberry32(42));
const testSize = Math.round(shuffled.length * 0.2);
const test = shuffled.slice(0, testSize);
const train = shuffled.slice(testSize);
const valSize = Math.round(train.length * 0.1);
const validation = train.slice(0, valSize);
const trainSub = train.slice(valSize);

function fallback(index, u) {
    return Number.isNaN(index.userMean[u]) ? index.globalMean : index.userMean[u];
}

function evalCF(index, S, evalSet, predictor) {
    const pairs = [];
    let covered = 0;
    const t0 = performance.now();
    for (const { u, i, r } of evalSet) {
        const p = predictor(index, S, u, i, K);
        if (p) covered++;
        pairs.push([cf.clip(p ? p.value : fallback(index, u)), r]);
    }
    return { ...cf.errors(pairs), coverage: covered / evalSet.length, predictMs: performance.now() - t0 };
}

function evalSimple(evalSet, predict) {
    return cf.errors(evalSet.map(({ u, i, r }) => [cf.clip(predict(u, i)), r]));
}

function biasBaseline(index, lambda = 10) {
    const mu = index.globalMean;
    const bi = index.byItem.map(l => l.reduce((s, [, r]) => s + r - mu, 0) / (lambda + l.length));
    const bu = index.byUser.map(l => l.reduce((s, [i, r]) => s + r - mu - bi[i], 0) / (lambda + l.length));
    return (u, i) => mu + (bu[u] || 0) + (bi[i] || 0);
}

function runCF(trainSet, evalSet, strategies) {
    const index = cf.buildIndex(trainSet, NUM_USERS, NUM_ITEMS);
    const rows = [];
    for (const strategy of strategies) {
        let t0 = performance.now();
        const SU = cf.similarityMatrix(index.byUser, index.userMean, strategy, NUM_ITEMS);
        const userSimMs = performance.now() - t0;
        t0 = performance.now();
        const SI = cf.similarityMatrix(index.byItem, index.itemMean, strategy, NUM_USERS);
        const itemSimMs = performance.now() - t0;
        rows.push({ method: 'user-based', strategy, simMs: userSimMs, ...evalCF(index, SU, evalSet, cf.predictUserBased) });
        rows.push({ method: 'item-based', strategy, simMs: itemSimMs, ...evalCF(index, SI, evalSet, cf.predictItemBased) });
        if (strategy === 'corated') {
            rows.push({ method: 'user-based, readme neighbourhood (global top-20)', strategy, simMs: userSimMs,
                ...evalCF(index, SU, evalSet, cf.predictUserBasedGlobal) });
            rows.push({ method: 'user-based, mean-centred', strategy, simMs: userSimMs,
                ...evalCF(index, SU, evalSet, cf.predictUserBasedCentered) });
            rows.push({ method: 'item-based, mean-centred', strategy, simMs: itemSimMs,
                ...evalCF(index, SI, evalSet, cf.predictItemBasedCentered) });
        }
    }
    return { index, rows };
}

const round = (o) => Object.fromEntries(Object.entries(o).map(([k, v]) =>
    [k, typeof v === 'number' ? +v.toFixed(k.endsWith('Ms') ? 0 : 4) : v]));

// 1. Choose the missing-value strategy on validation
const selection = runCF(trainSub, validation, STRATEGIES).rows.map(round);

// 2. Report everything on test
const { index, rows } = runCF(train, test, STRATEGIES);
const bias = biasBaseline(index);
let t0 = performance.now();
const mf = cf.trainMF(train, NUM_USERS, NUM_ITEMS);
const mfTrainMs = performance.now() - t0;
const baselines = [
    { method: 'global mean', ...evalSimple(test, () => index.globalMean) },
    { method: 'user mean', ...evalSimple(test, (u) => fallback(index, u)) },
    { method: 'item mean', ...evalSimple(test, (u, i) => Number.isNaN(index.itemMean[i]) ? fallback(index, u) : index.itemMean[i]) },
    { method: 'bias baseline (mu + b_u + b_i)', ...evalSimple(test, bias) },
    { method: 'matrix factorization (SGD, 100 factors)', trainMs: mfTrainMs, ...evalSimple(test, mf) }
];

const summary = {
    split: { train: train.length, test: test.length, validation: validation.length, seed: 42 },
    k: K,
    cold_test_items: test.filter(x => index.byItem[x.i].length === 0).length,
    validation: selection,
    test: [...rows.map(round), ...baselines.map(round)]
};
writeFileSync(join(root, 'evidence', 'evaluation.json'), JSON.stringify(summary, null, 1));

console.log(`train=${train.length} test=${test.length} validation=${validation.length} ` +
    `cold test items=${summary.cold_test_items}`);
console.log('\nVALIDATION (used to choose the strategy)');
for (const r of selection) console.log(`${r.method.padEnd(11)} ${r.strategy.padEnd(9)} RMSE ${r.rmse.toFixed(4)}  MAE ${r.mae.toFixed(4)}  coverage ${(r.coverage * 100).toFixed(1)}%`);
console.log('\nTEST');
for (const r of summary.test) {
    const extra = r.simMs !== undefined ? `  coverage ${(r.coverage * 100).toFixed(1)}%  sim ${r.simMs} ms  predict ${r.predictMs} ms`
        : r.trainMs !== undefined ? `  train ${r.trainMs} ms` : '';
    console.log(`${(r.method + (r.strategy ? ' / ' + r.strategy : '')).padEnd(42)} RMSE ${r.rmse.toFixed(4)}  MAE ${r.mae.toFixed(4)}${extra}`);
}
