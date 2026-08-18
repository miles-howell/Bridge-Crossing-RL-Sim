// This script runs in the background, separate from the main page UI.

let running = false;
let tickTimeoutId = null;
let apiStateUrl = '';
let qValueApiUrl = '';
let managerQValueApiUrl = '';
let csrfToken = '';
let tickCount = 0;
let speedMs = 50;
let qValuesInFlight = false;

// Q-value heatmaps only change gradually (a handful of gradient steps
// between polls), so refetching them is tied to how many training ticks
// have actually happened rather than a fixed wall-clock timer. This makes
// the heatmap refresh rate scale with the sim-speed slider automatically,
// instead of polling at a rate that's independent of - and can outrun -
// how fast the simulation is actually progressing.
const QVALUE_FETCH_TICK_INTERVAL = 10;

/**
 * The main loop that drives the simulation training by fetching state from the server.
 *
 * Each tick is scheduled only after the previous one has fully resolved,
 * rather than on a fixed setInterval. The server does real Q-learning work
 * per tick (action selection + replay training), which can take longer than
 * the configured tick speed as agents/replay buffers grow. Firing fetches on
 * a fixed timer regardless of completion causes unbounded numbers of
 * concurrent in-flight requests, which eventually exhausts the browser's
 * per-origin connection/request limit and makes fetch() reject with
 * "Failed to fetch" - even though the server is still happily working
 * through the backlog it already received.
 */
async function trainingTick() {
    if (!running) return;

    try {
        const response = await fetch(apiStateUrl);
        if (!running) return;

        if (!response.ok) {
            const errorMsg = `API Error on state update: ${response.status} ${response.statusText}`;
            self.postMessage({ type: 'error', message: errorMsg });
            stopTicking();
            return;
        }
        const gameState = await response.json();
        if (!running) return;
        self.postMessage({ type: 'update', gameState: gameState });

        tickCount++;
        if (tickCount % QVALUE_FETCH_TICK_INTERVAL === 0) {
            fetchAllQValues();
        }
    } catch (error) {
        if (!running) return;
        self.postMessage({ type: 'error', message: `Network error during training tick: ${error.message}` });
        stopTicking();
        return;
    }

    if (running) {
        tickTimeoutId = setTimeout(trainingTick, speedMs);
    }
}

function stopTicking() {
    running = false;
    if (tickTimeoutId) {
        clearTimeout(tickTimeoutId);
        tickTimeoutId = null;
    }
}

/**
 * A generic function to fetch data from a single API endpoint via POST.
 * @param {string} apiUrl - The URL to fetch from.
 * @param {string} token - The CSRF token for the POST request.
 * @returns {Promise<Object>} - A promise that resolves to the JSON response.
 */
async function fetchApi(apiUrl, token) {
    const response = await fetch(apiUrl, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': token,
        },
        body: JSON.stringify({})
    });
    if (!response.ok) {
        throw new Error(`API request to ${apiUrl} failed: ${response.status} ${response.statusText}`);
    }
    return response.json();
}

/**
 * Fetches both the worker's and manager's Q-value maps and posts them
 * together as a single update. Skips the request entirely if a previous
 * Q-value fetch is still in flight, so a slow response can't cause these
 * requests to pile up on top of each other (and on top of the tick
 * requests) the same way a fixed-interval poll would.
 */
function fetchAllQValues() {
    if (qValuesInFlight) return;
    qValuesInFlight = true;
    Promise.all([
        fetchApi(qValueApiUrl, csrfToken),
        fetchApi(managerQValueApiUrl, csrfToken)
    ]).then(([workerData, managerData]) => {
        if (!running) return;
        // On success, post a single message with both payloads
        self.postMessage({
            type: 'all_q_values_update',
            qValueData: workerData,
            managerQValueData: managerData,
        });
    }).catch(error => {
        if (!running) return;
        // On failure, post a detailed error message to the UI
        self.postMessage({ type: 'error', message: `Q-Value Fetch Failed: ${error.message}` });
    }).finally(() => {
        qValuesInFlight = false;
    });
}

/**
 * Listens for messages from the main page to control the simulation.
 */
self.onmessage = function(e) {
    const { command, data } = e.data;

    if (command === 'start') {
        // Initialize all necessary variables from the main page
        apiStateUrl = data.apiStateUrl;
        qValueApiUrl = data.qValueApiUrl;
        managerQValueApiUrl = data.managerQValueApiUrl;
        csrfToken = data.csrfToken;
        speedMs = data.speed;

        // Clear any old loop to prevent duplicates
        stopTicking();
        tickCount = 0;
        qValuesInFlight = false;

        // Start the self-pacing training tick loop
        running = true;
        trainingTick(); // Initial fetch; schedules its own follow-ups

        // Populate the heatmaps immediately rather than waiting for the
        // first QVALUE_FETCH_TICK_INTERVAL ticks to elapse.
        fetchAllQValues();

    } else if (command === 'update_speed') {
        // The next self-scheduled tick will pick up the new speed
        // automatically; no need to tear down and restart the loop.
        speedMs = data.speed;
    } else if (command === 'stop') {
        stopTicking();
    }
};
