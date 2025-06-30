// This script runs in the background, separate from the main page UI.

let trainingInterval = null;
let qValueInterval = null;
let apiStateUrl = '';
let qValueApiUrl = '';
let managerQValueApiUrl = '';
let csrfToken = '';

/**
 * The main loop that drives the simulation training by fetching state from the server.
 */
async function trainingTick() {
    try {
        const response = await fetch(apiStateUrl);
        if (!response.ok) {
            const errorMsg = `API Error on state update: ${response.status} ${response.statusText}`;
            self.postMessage({ type: 'error', message: errorMsg });
            clearInterval(trainingInterval);
            trainingInterval = null;
            return;
        }
        const gameState = await response.json();
        self.postMessage({ type: 'update', gameState: gameState });
    } catch (error) {
        self.postMessage({ type: 'error', message: `Network error during training tick: ${error.message}` });
        clearInterval(trainingInterval);
        trainingInterval = null;
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
        let speed = data.speed;

        // Clear any old intervals to prevent duplicates
        if (trainingInterval) clearInterval(trainingInterval);
        if (qValueInterval) clearInterval(qValueInterval);

        // Start the training tick interval
        trainingTick(); // Initial fetch
        trainingInterval = setInterval(trainingTick, speed);

        // A function to fetch both sets of Q-values
        const fetchAllQValues = () => {
             Promise.all([
                fetchApi(qValueApiUrl, csrfToken),
                fetchApi(managerQValueApiUrl, csrfToken)
            ]).then(([workerData, managerData]) => {
                // On success, post a single message with both payloads
                self.postMessage({
                    type: 'all_q_values_update',
                    qValueData: workerData,
                    managerQValueData: managerData,
                });
            }).catch(error => {
                // On failure, post a detailed error message to the UI
                self.postMessage({ type: 'error', message: `Q-Value Fetch Failed: ${error.message}` });
            });
        };
        
        // Start the Q-value fetching interval
        fetchAllQValues(); // Initial fetch
        qValueInterval = setInterval(fetchAllQValues, 200);

    } else if (command === 'update_speed') {
        let speed = data.speed;
        // If the simulation is running, clear and reset the interval with the new speed
        if (trainingInterval) {
            clearInterval(trainingInterval);
            trainingInterval = setInterval(trainingTick, speed);
        }
    } else if (command === 'stop') {
        // Clear all intervals when the simulation is stopped
        if (trainingInterval) clearInterval(trainingInterval);
        if (qValueInterval) clearInterval(qValueInterval);
        trainingInterval = null;
        qValueInterval = null;
    }
};
