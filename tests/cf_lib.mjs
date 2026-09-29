// Sparse collaborative-filtering building blocks for the offline evaluation.
// The web app (script.js) works on the dense rating matrix the assignment prescribes; this file
// implements the same formulas on sparse lists so that thousands of predictions run in seconds.
// tests/app_vs_lib.mjs checks that both implementations give the same predictions.
import { readFileSync } from 'fs';

export function mulberry32(seed) {
    return () => {
        seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
        let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

export function shuffle(array, rand) {
    const a = array.slice();
    for (let i = a.length - 1; i > 0; i--) {
        const j = Math.floor(rand() * (i + 1));
        [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
}

export function loadRatings(path) {
    return readFileSync(path, 'utf8').split('\n').filter(l => l.trim() !== '').map(line => {
        const [u, i, r, t] = line.split('\t').map(Number);
        return { u, i, r, t };
    });
}

// Per-user and per-item rating lists, sorted by the other id so that two lists can be intersected
// with a linear merge.
export function buildIndex(train, numUsers, numItems) {
    const byUser = Array.from({ length: numUsers + 1 }, () => []);
    const byItem = Array.from({ length: numItems + 1 }, () => []);
    for (const { u, i, r } of train) {
        byUser[u].push([i, r]);
        byItem[i].push([u, r]);
    }
    const sortLists = lists => lists.forEach(l => l.sort((a, b) => a[0] - b[0]));
    sortLists(byUser); sortLists(byItem);
    const mean = lists => lists.map(l => l.length ? l.reduce((s, x) => s + x[1], 0) / l.length : NaN);
    const globalMean = train.reduce((s, x) => s + x.r, 0) / train.length;
    return { byUser, byItem, userMean: mean(byUser), itemMean: mean(byItem), globalMean, numUsers, numItems };
}

// Similarity between two sparse rating lists.
//   'corated'  cosine over co-rated entries only
//   'weighted' co-rated cosine multiplied by min(n, gamma) / gamma (significance weighting)
//   'impute'   cosine of the full vectors after filling every missing entry with the vector's own
//              mean m. With d = r - m on rated entries (d sums to 0), x·y = mx·my·D + dx·dy and
//              |x|^2 = mx^2·D + |dx|^2, where D is the vector length — so no dense vectors needed.
// Returns [similarity, number of co-rated entries].
export function similarity(a, b, strategy, ctx) {
    let p = 0, q = 0, n = 0, dot = 0, na = 0, nb = 0, dd = 0;
    const ma = ctx.meanA, mb = ctx.meanB;
    while (p < a.length && q < b.length) {
        if (a[p][0] === b[q][0]) {
            const x = a[p][1], y = b[q][1];
            dot += x * y; na += x * x; nb += y * y;
            if (strategy === 'impute') dd += (x - ma) * (y - mb);
            n++; p++; q++;
        } else if (a[p][0] < b[q][0]) p++; else q++;
    }
    if (strategy === 'impute') {
        const num = ma * mb * ctx.length + dd;
        const den = Math.sqrt((ma * ma * ctx.length + ctx.devA) * (mb * mb * ctx.length + ctx.devB));
        return [den === 0 ? 0 : num / den, n];
    }
    if (n === 0 || na === 0 || nb === 0) return [0, 0];
    let sim = dot / Math.sqrt(na * nb);
    if (strategy === 'weighted') sim *= Math.min(n, ctx.gamma) / ctx.gamma;
    return [sim, n];
}

// Full similarity matrix between the rows of `lists` (users or items).
export function similarityMatrix(lists, means, strategy, length, gamma = 50) {
    const size = lists.length;
    const sim = new Float64Array(size * size);
    const co = new Uint16Array(size * size);
    const dev = lists.map((l, k) => l.reduce((s, x) => s + (x[1] - means[k]) ** 2, 0));
    for (let x = 1; x < size; x++) {
        if (lists[x].length === 0) continue;
        for (let y = x + 1; y < size; y++) {
            if (lists[y].length === 0) continue;
            const [s, n] = similarity(lists[x], lists[y], strategy,
                { meanA: means[x], meanB: means[y], devA: dev[x], devB: dev[y], length, gamma });
            sim[x * size + y] = sim[y * size + x] = s;
            co[x * size + y] = co[y * size + x] = Math.min(n, 65535);
        }
    }
    return { sim, co, size };
}

// k-NN prediction: similarity-weighted average of the k most similar neighbours (positive
// similarity only) that have a value for the target. Returns null when no neighbour qualifies.
//   user-based: neighbours = users who rated item i, value = their rating of i
//   item-based: neighbours = items the user rated,  value = the user's rating of that item
export function knnPredict(candidates, S, target, k) {
    const scored = [];
    for (const [id, value] of candidates) {
        const s = S.sim[target * S.size + id];
        if (s > 0) scored.push([s, value, id]);
    }
    if (scored.length === 0) return null;
    scored.sort((a, b) => b[0] - a[0] || a[2] - b[2]);
    let num = 0, den = 0;
    const m = Math.min(k, scored.length);
    for (let t = 0; t < m; t++) { num += scored[t][0] * scored[t][1]; den += scored[t][0]; }
    return { value: num / den, support: m };
}

export function predictUserBased(index, S, u, i, k) {
    return knnPredict(index.byItem[i] || [], S, u, k);
}

export function predictItemBased(index, S, u, i, k) {
    return knnPredict(index.byUser[u] || [], S, i, k);
}

// Mean-centred variants (the GroupLens formula): average the neighbours' deviations from their own
// mean instead of their raw ratings, then add the target's mean back.
export function predictUserBasedCentered(index, S, u, i, k) {
    const deviations = (index.byItem[i] || []).map(([v, r]) => [v, r - index.userMean[v]]);
    const p = knnPredict(deviations, S, u, k);
    return p && { value: index.userMean[u] + p.value, support: p.support };
}

export function predictItemBasedCentered(index, S, u, i, k) {
    if (Number.isNaN(index.itemMean[i])) return null;
    const deviations = (index.byUser[u] || []).map(([j, r]) => [j, r - index.itemMean[j]]);
    const p = knnPredict(deviations, S, i, k);
    return p && { value: index.itemMean[i] + p.value, support: p.support };
}

// The neighbourhood the assignment's readme describes: the k most similar users overall, fixed per
// active user; a movie can only be scored if at least one of those k users rated it.
const globalNeighbours = new WeakMap();
export function predictUserBasedGlobal(index, S, u, i, k) {
    if (!globalNeighbours.has(S)) globalNeighbours.set(S, new Map());
    const cache = globalNeighbours.get(S);
    if (!cache.has(u)) {
        const order = [];
        for (let v = 1; v < S.size; v++) {
            const s = S.sim[u * S.size + v];
            if (v !== u && s > 0) order.push([s, v]);
        }
        order.sort((a, b) => b[0] - a[0] || a[1] - b[1]);
        cache.set(u, new Map(order.slice(0, k).map(([s, v]) => [v, s])));
    }
    const neighbours = cache.get(u);
    let num = 0, den = 0, support = 0;
    for (const [v, r] of index.byItem[i] || []) {
        const s = neighbours.get(v);
        if (s !== undefined) { num += s * r; den += s; support++; }
    }
    return support ? { value: num / den, support } : null;
}

// Biased matrix factorization trained with SGD (the model popularised by the Netflix Prize;
// hyper-parameters follow the defaults of Surprise's SVD).
export function trainMF(train, numUsers, numItems, opts = {}) {
    const f = opts.factors ?? 100, epochs = opts.epochs ?? 20, lr = opts.lr ?? 0.005, reg = opts.reg ?? 0.02;
    const rand = mulberry32(opts.seed ?? 7);
    const gauss = () => { // Box-Muller, std 0.1
        const u1 = Math.max(rand(), 1e-12), u2 = rand();
        return 0.1 * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    };
    const P = new Float64Array((numUsers + 1) * f).map(gauss);
    const Q = new Float64Array((numItems + 1) * f).map(gauss);
    const bu = new Float64Array(numUsers + 1), bi = new Float64Array(numItems + 1);
    const mu = train.reduce((s, x) => s + x.r, 0) / train.length;
    let order = train.slice();
    for (let e = 0; e < epochs; e++) {
        order = shuffle(order, rand);
        for (const { u, i, r } of order) {
            let dot = 0;
            for (let t = 0; t < f; t++) dot += P[u * f + t] * Q[i * f + t];
            const err = r - (mu + bu[u] + bi[i] + dot);
            bu[u] += lr * (err - reg * bu[u]);
            bi[i] += lr * (err - reg * bi[i]);
            for (let t = 0; t < f; t++) {
                const pu = P[u * f + t], qi = Q[i * f + t];
                P[u * f + t] += lr * (err * qi - reg * pu);
                Q[i * f + t] += lr * (err * pu - reg * qi);
            }
        }
    }
    return (u, i) => {
        let dot = 0;
        for (let t = 0; t < f; t++) dot += P[u * f + t] * Q[i * f + t];
        return mu + bu[u] + bi[i] + dot;
    };
}

export const clip = x => Math.min(5, Math.max(1, x));

export function errors(pairs) {
    let se = 0, ae = 0;
    for (const [pred, truth] of pairs) { se += (pred - truth) ** 2; ae += Math.abs(pred - truth); }
    return { rmse: Math.sqrt(se / pairs.length), mae: ae / pairs.length };
}
