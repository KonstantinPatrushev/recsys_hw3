// Reproduces the failure case in the report: top-5 lists without a minimum number of ratings,
// on the full data, for users 1 and 13 (co-rated cosine, k = 20).
import { readFileSync, writeFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import * as cf from './cf_lib.mjs';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const all = cf.loadRatings(join(root, 'u.data'));
const titles = readFileSync(join(root, 'u.item'), 'latin1').split('\n').filter(Boolean).map(l => l.split('|')[1]);
const index = cf.buildIndex(all, 943, 1682);
const SU = cf.similarityMatrix(index.byUser, index.userMean, 'corated', 1682);
const SI = cf.similarityMatrix(index.byItem, index.itemMean, 'corated', 943);
const lines = [];
for (const u of [1, 13]) {
    const rated = new Set(index.byUser[u].map(x => x[0]));
    for (const [name, predict, S] of [['user-based', cf.predictUserBased, SU], ['item-based', cf.predictItemBased, SI]]) {
        const scored = [];
        for (let i = 1; i <= 1682; i++) {
            if (rated.has(i)) continue;
            const p = predict(index, S, u, i, 20);
            if (p) scored.push([p.value, p.support, index.byItem[i].length, i]);
        }
        scored.sort((a, b) => b[0] - a[0] || b[2] - a[2] || a[3] - b[3]);
        lines.push(`user ${u} ${name}: ` + scored.slice(0, 5).map(([v, s, n, i]) =>
            `${titles[i - 1]} ${v.toFixed(2)} (support ${s}, ${n} ratings)`).join(' | '));
    }
}
writeFileSync(join(root, 'evidence', 'failure_case.txt'), lines.join('\n') + '\n');
console.log(lines.join('\n'));
