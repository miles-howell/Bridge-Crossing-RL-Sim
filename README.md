# Bridge-Crossing-RL-Sim: A Hierarchical Reinforcement Learning Simulation

This repository presents a Django-based web application designed for the interactive visualization and analysis of a Hierarchical Reinforcement Learning (HRL) model. The simulation demonstrates an artificial agent's learning process in a grid-world environment, where it must acquire the capability to construct a bridge and navigate to a designated 'home' location.
![image](https://github.com/user-attachments/assets/fd2c00bd-f776-4b1b-a042-68bc119b6c78)
![image](https://github.com/user-attachments/assets/0efa3d90-629a-4d40-b646-2050079067c3)
![image](https://github.com/user-attachments/assets/538f7a5d-0167-480d-9bc6-033573eef588) (Note the feedback loops hindering performance)

### 1. Introduction and Architectural Overview

The project implements a two-tiered HRL architecture, separating high-level strategic planning from low-level action execution. This design addresses challenges in complex, sparse-reward environments by decomposing the problem into more manageable sub-problems.

The application is structured as a Django web project, with core components organized as follows:
* **`simulation/`**: This Django app encapsulates the primary simulation logic.
    * `engine.py`: Contains the Reinforcement Learning algorithms, including the `ManagerAgent` and `WorkerAgent` implementations, the `ReplayBuffer` for experience storage, and the `SimulationWorld` representing individual agent environments.
    * `views.py`: Handles HTTP requests, serving the main simulation interface and exposing RESTful API endpoints for state retrieval and Q-value visualization.
    * `templates/simulation/index.html`: The main frontend template for rendering the simulation.
* **`simulation_backend/`**: The root Django project, managing configurations, URL routing (`urls.py`), and settings (`settings.py`).
* **`static/`**: Hosts static assets, notably `simulation_worker.js`, which operates as a Web Worker for asynchronous simulation updates and Q-value fetching.

This architecture faces the same issue as many other hierarchical models. Training two individual, disconnected Q-tables can be unreliable when working towards a shared goal. A series of simple mistakes on the worker's part, can result in major Q-value divergence from the optpimal strategy for the manager, even if an optimal strategy has already been developed. The margin for error on either of the models' parts is extremely thin with this setup. 
We solve for this by introducing a novel "profit-sharing" technique, reminiscent of Potential Based Reward Shaping (PBRS). The manager is rewarded for the completion of a goal or sub-goal, this reward is considered profit, hence how it knows it is doing a good job or not. The worker, who has been a loyal employee, earns a % amount of profit each step. This can be either positive or negative depending on the Q-value at that tile in the manager's Q-table. This solution solves the issue of discontinuity between the goals of the two agents. 
Using this method, the two agents are much better equipped to solve a complex problem together. The worker still using HER to map the entire environment, but also benefits from the major milestone rewards, subsequently "understanding" the actual goal. Previously, the worker had no way of telling why (2,9) is a good place to navigate to for example. The "profti-share" solution differs from a more basic linear sharing of the milestone rewards, which could possibly result in exploding Q-table values and inconsistent learning behavior.

### 2. Reinforcement Learning Implementation Details

#### 2.1 Agent Architecture

* **Manager Agent (`ManagerAgent`)**: This high-level agent learns a policy for selecting abstract sub-goals. Its state space (`_get_manager_state`) is a tuple reflecting the agent's global progress: `(has_bridge_piece, placed_bridge, has_crossed)`. The action space consists of predefined sub-goals: 'GOTO_LOG', 'GOTO_RIVER', 'GOTO_FAR_BANK', and 'GOTO_HOUSE'.
* **Worker Agent (`WorkerAgent`)**: This low-level agent is responsible for executing specific sub-goals provided by the Manager. Its state space (`_get_worker_state`) includes the agent's current coordinates (`ax`, `ay`) and the same global progress flags as the Manager. The action space comprises primitive movement actions: 'UP', 'DOWN', 'LEFT', 'RIGHT'.

Both agents utilize Q-learning with an epsilon-greedy policy for action selection and value iteration for Q-table updates.

#### 2.2 Hindsight Experience Replay (HER)

The `WorkerAgent` incorporates Hindsight Experience Replay (HER) to improve sample efficiency, particularly in sparse-reward settings. When a sub-goal is attempted, both the "real" trajectory (with respect to the desired goal and dense rewards) and "imaginary" trajectories (with respect to achieved goals and sparse hindsight rewards) are stored in the `ReplayBuffer`. This allows the agent to learn from failed attempts by reinterpreting them as successes towards different, achieved goals, thereby providing more positive training signals.

#### 2.3 Reward Functions

* **Worker Reward:** A dense reward mechanism is employed. A step penalty (`step_penalty`) is applied at each time step. Positive rewards are granted for significant progress, such as picking up the bridge piece (+100), placing the bridge (+100), and crossing the river (+100). A large negative reward (-1000) is incurred if the agent drowns.
* **Manager Reward:** The Manager receives a positive reward (+100) if its chosen sub-goal is achieved by the Worker within `WORKER_MAX_STEPS`, and a negative reward (-100) otherwise.

#### 2.4 Environment

The simulation operates on a `GRID_COLS` x `GRID_ROWS` grid. Key environmental features include:
* **River:** A central obstacle with a defined `RIVER_WATER_WIDTH` and `RIVER_BORDER_WIDTH`.
* **Bridge Piece:** An object that must be picked up by the agent.
* **Bridge Placement Zone:** A specific location in the river where the bridge piece can be placed.
* **House:** The ultimate target destination for the agent.

### 3. Frontend-Backend Interaction

The `simulation_worker.js` script leverages Web Workers to execute the `trainingTick` function asynchronously. This function periodically fetches the current `gameState` from a Django API endpoint (`apiStateUrl`) and posts updates to the main thread for rendering. Similarly, Q-value visualization data can be dynamically retrieved via a separate API call (`fetchQValues`), enabling real-time inspection of the Worker Agent's learned value function.

### 4. Setup and Installation

To set up and run this Django application locally:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/miles-howell/bridge-crossing-rl-sim.git](https://github.com/miles-howell/bridge-crossing-rl-sim.git)
    cd bridge-crossing-rl-sim
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Apply database migrations:**
    ```bash
    python manage.py migrate
    ```
5.  **Collect static files:**
    ```bash
    python manage.py collectstatic
    ```
6.  **Start the Django development server:**
    ```bash
    python manage.py runserver
    ```
The application will typically be accessible at `http://127.0.0.1:8000/simulation`.

### 5. Usage

Upon accessing the application, users can:
* Initiate and reset the HRL simulation.
* Adjust hyper-parameters such as learning rates, discount factors, exploration rates, and simulation speed to observe their impact on the learning dynamics.
* Visualize the Worker Agent's learned Q-value landscape for different sub-goals, providing insights into the learned policy and state-value estimations.

### 6. Technologies and Dependencies

* **Backend:**
    * Python 3.x
    * Django (Python web framework)
* **Frontend:**
    * HTML5, CSS3, JavaScript
    * Tailwind CSS (for styling)
    * jQuery (JavaScript library)
    * Web Workers (for background processing)
    * Select2 (for enhanced select box functionality, if used in admin or main interface)
    * XRegExp (for advanced regular expressions)
    * Thanks to ArMM1998 on OpenGameArt.org for creating this free spritepack (https://opengameart.org/content/zelda-like-tilesets-and-sprites)

### 7. License

This project is licensed under the MIT License.

### 8. Contributing

Contributions are welcome. Please refer to the standard GitHub flow for submitting pull requests or reporting issues.
