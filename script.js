// ---------------------------------------------------------------------------
// HW3 — Collaborative Filtering core
//
// Missing-value strategy (see week3/readme.md section 6):
//
//   [x] use co-rated entries only
//
// Chosen on a validation split (tests/evaluate.mjs): it was the best of the three for item-based
// CF and tied with co-rated weighting for user-based CF, while mean imputation was worse for both.
//
// Two further choices, both measured in tests/:
//   * Neighbourhood: the NEIGHBOURS most similar users (or movies) among those that actually rated
//     the target, not a fixed top-20 per active user — the fixed set could only score 26.8 % of
//     test ratings (tests/evaluate.mjs).
//   * Top-5 lists only rank movies with at least MIN_RATINGS ratings. Without this rule the lists
//     are filled with movies rated 5 by one or two people (tests/ranking.mjs).
// ---------------------------------------------------------------------------
const NEIGHBOURS = 20;
const MIN_RATINGS = 20;

// Initialize the application when the window loads
window.onload = async function() {
    const userBased = document.getElementById('user-based-result');
    const itemBased = document.getElementById('item-based-result');

    try {
        userBased.innerHTML = '<p>Loading movie data...</p>';
        itemBased.innerHTML = '<p>Loading movie data...</p>';

        await loadData();

        populateUserDropdown();
        populateMovieDropdown();

        userBased.innerHTML = '<p>Data loaded. Select a user.</p>';
        itemBased.innerHTML = '<p>Data loaded. Select a user.</p>';
    } catch (error) {
        console.error('Initialization error:', error);
        // The error message is already shown by data.js
    }
};

// Populate the user dropdown with one option per user id found in u.data
function populateUserDropdown() {
    const selectElement = document.getElementById('user-select');

    // Clear existing options except the first placeholder
    while (selectElement.options.length > 1) {
        selectElement.remove(1);
    }

    for (let userId = 1; userId <= numUsers; userId++) {
        const option = document.createElement('option');
        option.value = userId;
        option.textContent = `User ${userId}`;
        selectElement.appendChild(option);
    }
}

// Populate the movie dropdown (used by the rating predictor), sorted by title
function populateMovieDropdown() {
    const selectElement = document.getElementById('movie-select');
    while (selectElement.options.length > 1) {
        selectElement.remove(1);
    }
    [...movies].sort((a, b) => a.title.localeCompare(b.title)).forEach(movie => {
        const option = document.createElement('option');
        option.value = movie.id;
        option.textContent = movie.title;
        selectElement.appendChild(option);
    });
}

// ---------------------------------------------------------------------------
// Cosine similarity between two rating vectors over co-rated entries only:
// positions where either vector is 0 ("not rated") are skipped, so the norms are
// computed on the co-rated entries too. Returns 0 when the two vectors share no
// rated item. Ratings are 1-5, so the result is in [0, 1].
// ---------------------------------------------------------------------------
function cosineSimilarity(a, b) {
    let dot = 0, normA = 0, normB = 0;
    for (let k = 0; k < a.length; k++) {
        if (a[k] === 0 || b[k] === 0) continue;
        dot += a[k] * b[k];
        normA += a[k] * a[k];
        normB += b[k] * b[k];
    }
    if (normA === 0 || normB === 0) return 0;
    return dot / Math.sqrt(normA * normB);
}

// Similarity-weighted average of the NEIGHBOURS most similar candidates.
// candidates: array of [similarity, id, rating]; ties are broken by the smaller id.
function weightedAverage(candidates) {
    const positive = candidates.filter(c => c[0] > 0);
    if (positive.length === 0) return null;
    positive.sort((x, y) => y[0] - x[0] || x[1] - y[1]);
    const top = positive.slice(0, NEIGHBOURS);
    let num = 0, den = 0;
    for (const [s, , rating] of top) { num += s * rating; den += s; }
    return { score: num / den, support: top.length };
}

// Similarity of the active user to every user (index = user id; 0 for the user itself)
function userSimilarities(activeUserId) {
    const sims = new Float64Array(numUsers + 1);
    for (let userId = 1; userId <= numUsers; userId++) {
        if (userId !== activeUserId) {
            sims[userId] = cosineSimilarity(ratingMatrix[activeUserId], ratingMatrix[userId]);
        }
    }
    return sims;
}

// Columns of the rating matrix (one vector per movie over all users), built on first use
let itemVectors = null;
function getItemVector(movieId) {
    if (!itemVectors) {
        itemVectors = [];
        for (let m = 0; m <= numMovies; m++) itemVectors.push(new Uint8Array(numUsers + 1));
        for (let u = 1; u <= numUsers; u++) {
            const row = ratingMatrix[u];
            for (let m = 1; m <= numMovies; m++) if (row[m] !== 0) itemVectors[m][u] = row[m];
        }
    }
    return itemVectors[movieId];
}

// User-based prediction: weighted average over the most similar users who rated the movie
function predictUserBased(activeUserId, movieId, sims) {
    const candidates = [];
    for (let userId = 1; userId <= numUsers; userId++) {
        const rating = ratingMatrix[userId][movieId];
        if (userId !== activeUserId && rating !== 0) candidates.push([sims[userId], userId, rating]);
    }
    return weightedAverage(candidates);
}

// Item-based prediction: weighted average of the user's own ratings of the movies most similar
// to the target movie (the target itself is excluded)
function predictItemBased(activeUserId, movieId) {
    const target = getItemVector(movieId);
    const row = ratingMatrix[activeUserId];
    const candidates = [];
    for (let m = 1; m <= numMovies; m++) {
        if (m !== movieId && row[m] !== 0) {
            candidates.push([cosineSimilarity(target, getItemVector(m)), m, row[m]]);
        }
    }
    return weightedAverage(candidates);
}

// Number of ratings per movie (for the MIN_RATINGS rule and for tie-breaking)
function ratingCount(movieId) {
    return getItemVector(movieId).reduce((n, r) => n + (r !== 0 ? 1 : 0), 0);
}

// Rank every movie the user has not rated by its predicted score and keep the top K
function rankMovies(activeUserId, predict, topK) {
    const row = ratingMatrix[activeUserId];
    const scored = [];
    for (const movie of movies) {
        if (row[movie.id] !== 0) continue;
        const count = ratingCount(movie.id);
        if (count < MIN_RATINGS) continue;
        const p = predict(movie.id);
        if (p) scored.push({ title: movie.title, score: p.score, count, id: movie.id });
    }
    scored.sort((a, b) => b.score - a.score || b.count - a.count || a.id - b.id);
    return scored.slice(0, topK).map(({ title, score }) => ({ title, score }));
}

// ---------------------------------------------------------------------------
// User-Based CF: top-K recommendations for the active user as { title, score }
// ---------------------------------------------------------------------------
function getUserBasedRecommendations(activeUserId, topK = 5) {
    const sims = userSimilarities(activeUserId);
    return rankMovies(activeUserId, movieId => predictUserBased(activeUserId, movieId, sims), topK);
}

// ---------------------------------------------------------------------------
// Item-Based CF: top-K recommendations for the active user as { title, score }
// ---------------------------------------------------------------------------
function getItemBasedRecommendations(activeUserId, topK = 5) {
    return rankMovies(activeUserId, movieId => predictItemBased(activeUserId, movieId), topK);
}

// Provided — read the selected user and render both recommendation lists
function getRecommendations() {
    const selectElement = document.getElementById('user-select');
    const userId = parseInt(selectElement.value, 10);

    if (isNaN(userId)) {
        renderList('user-based-result', [], 'Please select a user first.');
        renderList('item-based-result', [], 'Please select a user first.');
        return;
    }

    const favourites = movies
        .filter(movie => ratingMatrix[userId][movie.id] === 5)
        .slice(0, 3)
        .map(movie => `"${movie.title}"`);
    renderList('user-based-result', getUserBasedRecommendations(userId), null,
        'Because you are similar to other users, we recommend:');

    // Item-based compares every candidate with every movie the user rated, which takes seconds
    // for users with hundreds of ratings, so show the user-based list first.
    renderList('item-based-result', [], 'Calculating… (can take several seconds for users with many ratings)');
    setTimeout(() => {
        renderList('item-based-result', getItemBasedRecommendations(userId), null,
            favourites.length ? `Because you liked ${favourites.join(', ')} and similar movies, we recommend:`
                : 'Based on the movies you rated, we recommend:');
    }, 50);
}

// Predict the selected user's rating of the selected movie with both methods
function predictRating() {
    const userId = parseInt(document.getElementById('user-select').value, 10);
    const movieId = parseInt(document.getElementById('movie-select').value, 10);
    const box = document.getElementById('prediction-result');

    if (isNaN(userId) || isNaN(movieId)) {
        box.innerHTML = '<p>Please select a user and a movie.</p>';
        return;
    }

    const userBased = predictUserBased(userId, movieId, userSimilarities(userId));
    const itemBased = predictItemBased(userId, movieId);
    const actual = ratingMatrix[userId][movieId];
    const card = (name, p, unit) => `<div class="prediction-card"><h3>${name}</h3>` +
        (p ? `<div class="prediction-value">${p.score.toFixed(1)}</div><p>predicted from ${p.support} similar ${unit}</p>`
           : '<div class="prediction-value">—</div><p>not enough data to predict</p>') + '</div>';

    box.innerHTML = card('User-Based CF', userBased, 'users') + card('Item-Based CF', itemBased, 'movies') +
        (actual ? `<p class="actual">Actual rating by this user: ${actual} (the predictions do not use it)</p>` : '');
}

// Provided — render a list of { title, score } into the given element
function renderList(elementId, items, message, intro) {
    const el = document.getElementById(elementId);

    if (message) {
        el.innerHTML = `<p>${message}</p>`;
        return;
    }

    if (!items || items.length === 0) {
        el.innerHTML = '<p>No recommendations: this user has too few ratings in common with others.</p>';
        return;
    }

    const entries = items
        .map(item => `<li>${item.title} &mdash; ${Number(item.score).toFixed(3)}</li>`)
        .join('');
    el.innerHTML = (intro ? `<p>${intro}</p>` : '') + `<ul>${entries}</ul>`;
}
