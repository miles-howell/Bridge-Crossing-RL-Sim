// This script runs in the background, separate from the main page UI.

let trainingInterval = null;
let apiStateUrl = '';

// The main loop that drives the simulation training
async function trainingTick() {
    try {
        const response = await fetch(apiStateUrl);
        if (!response.ok) {
            console.error("Worker: Failed to fetch state, stopping loop.", response.statusText);
            self.postMessage({ type: 'error', message: 'API fetch failed' });
            clearInterval(trainingInterval);
            trainingInterval = null;
            return;
        }
        const gameState = await response.json();
        // Send the latest game state back to the main page for rendering
        self.postMessage({ type: 'update', gameState: gameState });

    } catch (error) {
        console.error("Worker: Error in training tick:", error);
        self.postMessage({ type: 'error', message: 'Network error' });
        clearInterval(trainingInterval);
        trainingInterval = null;
    }
}

// Function to fetch the Q-value visualization data from the server
async function fetchQValues(qValueApiUrl, viewState, csrfToken) {
    try {
        const response = await fetch(qValueApiUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': csrfToken,
            },
            body: JSON.stringify({ view_state: viewState })
        });
        if (!response.ok) {
            console.error("Worker: Failed to fetch Q-values.", response.statusText);
            return;
        }
        const qValueData = await response.json();
        // Send the visualization data back to the main page
        self.postMessage({ type: 'q_values_update', qValueData: qValueData });
    } catch (error) {
        console.error("Worker: Error fetching Q-values:", error);
    }
}


// Listen for messages from the main page
self.onmessage = function(e) {
    const { command, data } = e.data;

    if (command === 'start') {
        apiStateUrl = data.apiStateUrl;
        let speed = data.speed;

        if (trainingInterval) {
            clearInterval(trainingInterval);
        }
        trainingTick();
        trainingInterval = setInterval(trainingTick, speed);
    } 
    else if (command === 'update_speed') {
        let speed = data.speed;
        if (trainingInterval) {
            clearInterval(trainingInterval);
            trainingInterval = setInterval(trainingTick, speed);
        }
    }
    else if (command === 'stop') {
        if (trainingInterval) {
            clearInterval(trainingInterval);
            trainingInterval = null;
        }
    }
    else if (command === 'get_q_values') {
        fetchQValues(data.qValueApiUrl, data.viewState, data.csrfToken);
    }
};
