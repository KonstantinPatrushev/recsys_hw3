// Checks for two claims in the report (same seeded 80/20 split as evaluate.mjs):
//   (a) with mean imputation the filled-in means dominate the user vectors
//   (b) co-rated weighting makes item-based CF pick more popular neighbours
// Output: evidence/claim_checks.txt
import * as cf from './cf_lib.mjs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const all = cf.loadRatings(join(root, 'u.data'));
const s = cf.shuffle(all, cf.mulberry32(42)); const train = s.slice(20000), test = s.slice(0, 20000);
const idx = cf.buildIndex(train, 943, 1682);
// (a) imputation: size of mean term vs deviation term for user pairs
let meanTerm = 0, devTerm = 0, n = 0;
for (let a = 1; a <= 943; a += 7) for (let b = a + 1; b <= 943; b += 11) {
  const ma = idx.userMean[a], mb = idx.userMean[b];
  let p = 0, q = 0, dd = 0; const A = idx.byUser[a], B = idx.byUser[b];
  while (p < A.length && q < B.length) { if (A[p][0] === B[q][0]) { dd += (A[p][1] - ma) * (B[q][1] - mb); p++; q++; } else if (A[p][0] < B[q][0]) p++; else q++; }
  meanTerm += ma * mb * 1682; devTerm += Math.abs(dd); n++;
}
console.log(`impute: mean term ${(meanTerm / n).toFixed(0)}, |deviation term| ${(devTerm / n).toFixed(1)} over ${n} pairs`);
const SUi = cf.similarityMatrix(idx.byUser, idx.userMean, 'impute', 1682);
let lo = 1, hi = 0, sum = 0, c = 0; for (let a = 1; a <= 943; a++) for (let b = a + 1; b <= 943; b++) { const v = SUi.sim[a * 944 + b]; lo = Math.min(lo, v); hi = Math.max(hi, v); sum += v; c++; }
console.log(`impute user cosine: min ${lo.toFixed(4)} mean ${(sum / c).toFixed(4)} max ${hi.toFixed(4)}`);
// (b) item-based weighted vs corated: popularity of the neighbours actually used
for (const strategy of ['corated', 'weighted']) {
  const S = cf.similarityMatrix(idx.byItem, idx.itemMean, strategy, 943);
  let pop = 0, cnt = 0;
  for (const { u, i } of test) {
    const scored = [];
    for (const [j] of idx.byUser[u]) { const v = S.sim[i * S.size + j]; if (v > 0) scored.push([v, j]); }
    scored.sort((x, y) => y[0] - x[0] || x[1] - y[1]);
    for (const [, j] of scored.slice(0, 20)) { pop += idx.byItem[j].length; cnt++; }
  }
  console.log(`item-based ${strategy}: mean train ratings of neighbours used = ${(pop / cnt).toFixed(1)}`);
}
