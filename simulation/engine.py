# simulation/engine.py

import copy
import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

# The networks here are tiny (two 64-unit hidden layers) and trained on small
# batches, so PyTorch's default multi-threaded matmul spends more time
# coordinating threads than the matmul itself takes. Single-threaded is
# faster for workloads this small.
torch.set_num_threads(1)

# --- CONFIGURATION ---
GRID_COLS = 20
GRID_ROWS = 12
RIVER_WATER_WIDTH = 3
RIVER_BORDER_WIDTH = 1
RIVER_TOTAL_WIDTH = RIVER_WATER_WIDTH + (RIVER_BORDER_WIDTH * 2)
BRIDGE_LENGTH_TILES = 3
HOUSE_WIDTH_TILES = 2
HOUSE_HEIGHT_TILES = 3
HER_K = 4
WORKER_MAX_STEPS = 75
GRAD_CLIP_NORM = 1.0
# Rewards the worker for physically reaching whatever coordinate the manager
# assigned as the current subgoal, independent of whether that directive
# actually advanced the task. This is deliberately decoupled from the
# manager's own reward (which IS gated on task-level correctness): the
# worker's job is to execute directives reliably, not to judge them.
WORKER_SUBGOAL_BONUS = 100


class AgentState:
    IN_PROGRESS = 'Learning...'
    HOME = 'Reached home! :)'
    DROWNED = 'Drowned :('
    TIMED_OUT = 'Failed (Time Limit)'


class ReplayBuffer:
    """ A fixed-size buffer that stores experiences. """
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, goal, done):
        self.memory.append((state, action, reward, next_state, goal, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size) if len(self.memory) >= batch_size else []

    def __len__(self):
        return len(self.memory)


class WorkerAgent:
    """The Worker uses a neural network to approximate Q-values for state-goal pairs."""

    def __init__(self, actions, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1, buffer_size=20000, target_sync_freq=500):
        self.actions = actions
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.memory = ReplayBuffer(buffer_size)

        input_dim = 7  # (ax, ay, has_bridge, bridge_placed, has_crossed, goal_x, goal_y)
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, len(actions))
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        # A frozen copy used only to compute bootstrap targets. Without this,
        # the same weights that are being updated right now are also used to
        # estimate the target a moment later, which lets a single update
        # ripple back into its own target and destabilize training.
        self.target_model = copy.deepcopy(self.model)
        self.target_sync_freq = target_sync_freq
        self.train_steps = 0

    def sync_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def _record_train_steps(self, n):
        prev = self.train_steps
        self.train_steps += n
        # Compare sync-interval boundaries crossed rather than checking for an
        # exact multiple, since a batched update can advance train_steps by
        # more than 1 at a time and could otherwise step over the boundary.
        if self.train_steps // self.target_sync_freq > prev // self.target_sync_freq:
            self.sync_target_network()

    def get_max_q_grid(self, goal, rows, cols, flags):
        """Returns a rows x cols grid of the max Q-value over all actions, for
        the given goal and fixed (has_bridge, bridge_placed, has_crossed)
        flags, using one batched forward pass instead of one call per cell."""
        states = [(c, r) + flags for r in range(rows) for c in range(cols)]
        state_goal = torch.tensor([list(s) + list(goal) for s in states], dtype=torch.float32)
        with torch.no_grad():
            q_values = self.model(state_goal)
        max_per_cell = q_values.max(dim=1).values.tolist()
        return [max_per_cell[r * cols:(r + 1) * cols] for r in range(rows)]

    def choose_action(self, state, goal):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        state_goal = torch.tensor([list(state) + list(goal)], dtype=torch.float32)
        with torch.no_grad():
            q_values = self.model(state_goal)[0]
        max_q = torch.max(q_values).item()
        best_actions = [a for a, q in zip(self.actions, q_values.tolist()) if q == max_q]
        return random.choice(best_actions)

    def train_on_batch(self, minibatch):
        """Runs one vectorized gradient step over a whole minibatch at once,
        instead of one individual forward/backward/step per sample. On CPU,
        a single batched matmul is far cheaper than looping python-side over
        many tiny ones."""
        states, actions, rewards, next_states, goals, dones = zip(*minibatch)

        state_goal = torch.tensor([list(s) + list(g) for s, g in zip(states, goals)], dtype=torch.float32)
        next_state_goal = torch.tensor([list(s) + list(g) for s, g in zip(next_states, goals)], dtype=torch.float32)
        action_index = torch.tensor([self.actions.index(a) for a in actions]).unsqueeze(1)
        reward_t = torch.tensor(rewards, dtype=torch.float32)
        done_t = torch.tensor([float(d) for d in dones], dtype=torch.float32)

        q_values = self.model(state_goal).gather(1, action_index).squeeze(1)
        with torch.no_grad():
            next_q = self.target_model(next_state_goal).max(1)[0]
            target = reward_t + self.gamma * next_q * (1 - done_t)

        loss = self.criterion(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        # Bounds the size of a single update. Without this, a single
        # large-magnitude reward (e.g. the -1000 drowning penalty) can push
        # enough ReLU units permanently negative in one step to collapse the
        # network's representational capacity ("dying ReLU").
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=GRAD_CLIP_NORM)
        self.optimizer.step()

        self._record_train_steps(len(minibatch))

    def experience_replay(self, batch_size):
        minibatch = self.memory.sample(batch_size)
        if minibatch:
            self.train_on_batch(minibatch)


class ManagerAgent:
    """High-level manager approximating Q-values with a neural network."""

    def __init__(self, actions, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1, buffer_size=10000, target_sync_freq=500):
        self.actions = actions
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.memory = ReplayBuffer(buffer_size)

        input_dim = 3  # (has_bridge_piece, bridge_placed, has_crossed)
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, len(actions))
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        # See WorkerAgent: a frozen copy for bootstrap targets so an update
        # doesn't chase a target that moves because of that same update.
        self.target_model = copy.deepcopy(self.model)
        self.target_sync_freq = target_sync_freq
        self.train_steps = 0

    def sync_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def _record_train_steps(self, n):
        prev = self.train_steps
        self.train_steps += n
        # Compare sync-interval boundaries crossed rather than checking for an
        # exact multiple, since a batched update can advance train_steps by
        # more than 1 at a time and could otherwise step over the boundary.
        if self.train_steps // self.target_sync_freq > prev // self.target_sync_freq:
            self.sync_target_network()

    def get_all_q_values(self, states):
        """Returns a list of per-action Q-value lists, one per input state,
        using a single batched forward pass instead of one call per state."""
        state_t = torch.tensor([list(s) for s in states], dtype=torch.float32)
        with torch.no_grad():
            q_values = self.model(state_t)
        return q_values.tolist()

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        state_t = torch.tensor([list(state)], dtype=torch.float32)
        with torch.no_grad():
            q_values = self.model(state_t)[0]
        max_q = torch.max(q_values).item()
        best_actions = [a for a, q in zip(self.actions, q_values.tolist()) if q == max_q]
        return random.choice(best_actions)

    def train_step(self, state, action, reward, next_state, done, store=True):
        if store:
            self.memory.add(state, action, reward, next_state, None, done)
        self.train_on_batch([(state, action, reward, next_state, done)])

    def train_on_batch(self, minibatch):
        """Runs one vectorized gradient step over a whole minibatch at once,
        instead of one individual forward/backward/step per sample."""
        states, actions, rewards, next_states, dones = zip(*minibatch)

        state_t = torch.tensor([list(s) for s in states], dtype=torch.float32)
        next_state_t = torch.tensor([list(s) for s in next_states], dtype=torch.float32)
        action_index = torch.tensor([self.actions.index(a) for a in actions]).unsqueeze(1)
        reward_t = torch.tensor(rewards, dtype=torch.float32)
        done_t = torch.tensor([float(d) for d in dones], dtype=torch.float32)

        q_values = self.model(state_t).gather(1, action_index).squeeze(1)
        with torch.no_grad():
            next_q = self.target_model(next_state_t).max(1)[0]
            target = reward_t + self.gamma * next_q * (1 - done_t)

        loss = self.criterion(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=GRAD_CLIP_NORM)
        self.optimizer.step()

        self._record_train_steps(len(minibatch))

    def experience_replay(self, batch_size):
        minibatch = self.memory.sample(batch_size)
        if minibatch:
            self.train_on_batch([(s, a, r, ns, d) for s, a, r, ns, _, d in minibatch])


class SimulationWorld:
    """ Represents one independent simulation instance for one agent. """
    def __init__(self, agent_id):
        self.id = agent_id
        self.river_start_col = (GRID_COLS // 2) - (RIVER_TOTAL_WIDTH // 2)
        self.house_start_col = GRID_COLS - HOUSE_WIDTH_TILES - 3
        self.house_start_row = 1
        self.bridge_y_pos_tiles = GRID_ROWS // 2

        self.subgoal_locations = {
            'GOTO_LOG': (2, GRID_ROWS - 3),
            'GOTO_RIVER': (self.river_start_col, self.bridge_y_pos_tiles),
            'GOTO_FAR_BANK': (self.river_start_col + RIVER_TOTAL_WIDTH - 1, self.bridge_y_pos_tiles),
            'GOTO_HOUSE': (self.house_start_col, self.house_start_row + HOUSE_HEIGHT_TILES - 1)
        }
        self.reset()

    def reset(self):
        self.agent = { "id": self.id, "ax": 1, "ay": 1, "has_bridge_piece": False, "has_crossed": False, "status": AgentState.IN_PROGRESS, "animation_frame": 0 }
        self.placed_bridge = None
        self._spawn_bridge_piece()
        self.current_subgoal_name = None
        self.current_subgoal_coord = None
        self.subgoal_trajectory = []
        self.subgoal_steps = 0

        # Track the manager's state at the moment a subgoal is issued so we can
        # later compute the transition when the subtask finishes. Without
        # storing this, the manager's update would see identical pre- and
        # post-states, preventing it from learning a useful sequence of
        # subgoals and leading to feedback loops.
        self.last_manager_state = None

        self.milestones_rewarded = {
            'log_picked_up': False,
            'bridge_placed': False,
            'crossed_bridge': False
        }

    def _spawn_bridge_piece(self):
        self.bridge_piece = { "ax": self.subgoal_locations['GOTO_LOG'][0], "ay": self.subgoal_locations['GOTO_LOG'][1] }


class SimulationEngine:
    def __init__(self, manager, worker, num_agents, milestones, batch_size=32, step_penalty=1):
        self.manager = manager; self.worker = worker; self.num_agents = num_agents
        self.milestones = milestones; self.batch_size = batch_size; self.step_penalty = step_penalty
        self.worlds = [SimulationWorld(i) for i in range(num_agents)]
        self.manager_steps = 0

    def _get_manager_state(self, world):
        return (1 if world.agent['has_bridge_piece'] else 0, 1 if world.placed_bridge else 0, 1 if world.agent['has_crossed'] else 0)

    def _get_worker_state(self, world):
        return (world.agent['ax'], world.agent['ay'], 1 if world.agent['has_bridge_piece'] else 0, 1 if world.placed_bridge else 0, 1 if world.agent['has_crossed'] else 0)

    def _compute_hindsight_reward(self, achieved_goal, desired_goal):
        # -10 rather than the more typical HER -1: these hindsight
        # experiences share a replay buffer, network, and MSE loss with
        # primary experiences carrying rewards up to +-1000+, so a -1 signal
        # is negligible by comparison and barely shapes the gradient at all.
        return 0 if achieved_goal == desired_goal else -10

    def update(self):
        for world in self.worlds:
            agent = world.agent
            if agent['status'] != AgentState.IN_PROGRESS:
                world.reset()
                continue

            if world.current_subgoal_name is None:
                self.manager_steps += 1
                manager_state = self._get_manager_state(world)
                # Remember the state before executing this subgoal so the
                # manager can learn from the resulting state transition.
                world.last_manager_state = manager_state
                subgoal_name = self.manager.choose_action(manager_state)
                world.current_subgoal_name = subgoal_name
                world.current_subgoal_coord = world.subgoal_locations.get(subgoal_name)
                if world.current_subgoal_coord is None:
                    continue

            worker_state = self._get_worker_state(world)
            worker_action = self.worker.choose_action(worker_state, world.current_subgoal_coord)
            agent['last_action'] = worker_action

            if worker_action == "UP": agent['ay'] -= 1
            elif worker_action == "DOWN": agent['ay'] += 1
            elif worker_action == "LEFT": agent['ax'] -= 1
            elif worker_action == "RIGHT": agent['ax'] += 1
            agent['ax'] = max(0, min(GRID_COLS - 1, agent['ax'])); agent['ay'] = max(0, min(GRID_ROWS - 1, agent['ay']))
            world.subgoal_steps += 1

            reward = -self.step_penalty
            current_pos = (agent['ax'], agent['ay'])
            new_milestone_achieved = False

            # --- FINAL REWARD LOGIC REFACTOR ---
            # This logic now strictly ties the manager's reward to the achievement of a NEW milestone.

            # Check for milestone events and give one-time rewards
            # Event: Pick up the log
            if not world.milestones_rewarded['log_picked_up'] and world.bridge_piece:
                log_center_x = world.bridge_piece['ax'] + 1
                if current_pos == (log_center_x, world.bridge_piece['ay']):
                    agent['has_bridge_piece'] = True; world.bridge_piece = None
                    self.milestones['picked_up'] += 1; reward += 100
                    world.milestones_rewarded['log_picked_up'] = True
                    if world.current_subgoal_name == 'GOTO_LOG':
                        new_milestone_achieved = True

            # Event: Place the bridge
            if not world.milestones_rewarded['bridge_placed'] and agent['has_bridge_piece']:
                if current_pos == world.subgoal_locations['GOTO_RIVER']:
                    world.placed_bridge = { "ax": world.river_start_col + RIVER_BORDER_WIDTH, "ay": world.bridge_y_pos_tiles }
                    agent['has_bridge_piece'] = False
                    self.milestones['placed'] += 1; reward += 100
                    world.milestones_rewarded['bridge_placed'] = True
                    if world.current_subgoal_name == 'GOTO_RIVER':
                        new_milestone_achieved = True

            # Event: Cross the bridge
            if not world.milestones_rewarded['crossed_bridge'] and world.placed_bridge:
                if agent['ax'] >= world.river_start_col + RIVER_TOTAL_WIDTH - 1:
                    agent['has_crossed'] = True
                    self.milestones['crossed'] += 1; reward += 100
                    world.milestones_rewarded['crossed_bridge'] = True
                    if world.current_subgoal_name == 'GOTO_FAR_BANK':
                        new_milestone_achieved = True

            # Event: Reach Home (This is a terminal state, so no reward flag is needed)
            if world.current_subgoal_name == 'GOTO_HOUSE' and agent['has_crossed'] and current_pos == world.subgoal_locations['GOTO_HOUSE']:
                agent['status'] = AgentState.HOME
                self.milestones['home'] += 1
                reward += 1000
                new_milestone_achieved = True

            # Drowning is always possible
            is_in_river_area = world.river_start_col < agent['ax'] < world.river_start_col + RIVER_TOTAL_WIDTH - 1
            if is_in_river_area:
                is_on_bridge = world.placed_bridge and world.placed_bridge['ay'] == agent['ay']
                if not is_on_bridge: agent['status'] = AgentState.DROWNED; reward -= 1000

            # Determine if the subtask is over
            subgoal_timed_out = (world.subgoal_steps >= WORKER_MAX_STEPS)
            subtask_is_over = new_milestone_achieved or subgoal_timed_out or agent['status'] != AgentState.IN_PROGRESS

            next_worker_state = self._get_worker_state(world)
            achieved_goal = (agent['ax'], agent['ay'])
            world.subgoal_trajectory.append((worker_state, worker_action, reward, next_worker_state, achieved_goal))

            if subtask_is_over:
                # The manager is ONLY rewarded if its command led to a new milestone.
                manager_reward = 100 if new_milestone_achieved else -100
                manager_state = world.last_manager_state
                next_manager_state = self._get_manager_state(world)
                manager_done = agent['status'] != AgentState.IN_PROGRESS
                self.manager.train_step(manager_state, world.current_subgoal_name, manager_reward, next_manager_state, manager_done)

                for state, act, rew, next_s, ach_g in world.subgoal_trajectory:
                    # The worker is rewarded for reaching the coordinate it was
                    # actually assigned, regardless of whether that directive
                    # was a good one -- that judgment belongs to the manager's
                    # reward above, not the worker's. This keeps "did I follow
                    # orders" separate from "were the orders any good".
                    reached_assigned_subgoal = (ach_g == world.current_subgoal_coord)
                    worker_rew = rew + (WORKER_SUBGOAL_BONUS if reached_assigned_subgoal else 0)
                    done = reached_assigned_subgoal or agent['status'] != AgentState.IN_PROGRESS
                    self.worker.memory.add(state, act, worker_rew, next_s, world.current_subgoal_coord, done)

                if not new_milestone_achieved:
                    imaginary_goal = world.subgoal_trajectory[-1][4]
                    for state, act, _, next_s, ach_g in world.subgoal_trajectory:
                        h_reward = self._compute_hindsight_reward(ach_g, imaginary_goal)
                        h_done = (ach_g == imaginary_goal)
                        self.worker.memory.add(state, act, h_reward, next_s, imaginary_goal, h_done)

                if agent['status'] == AgentState.IN_PROGRESS:
                    world.current_subgoal_name = None; world.subgoal_steps = 0; world.subgoal_trajectory = []

            self.worker.experience_replay(self.batch_size)
            self.manager.experience_replay(self.batch_size)

    def get_state(self):
        # Only include what the frontend actually renders: agent position/
        # animation and the two bridge objects. Everything else on
        # SimulationWorld (subgoal_trajectory, milestones_rewarded, etc.) is
        # internal training bookkeeping the client never reads, and
        # subgoal_trajectory in particular grows every tick during a subgoal
        # attempt, so shipping it made the response balloon in size for no
        # reason. river_start_col/house_start_col/house_start_row are
        # identical across every world (derived from fixed grid constants),
        # so they're sent once at the top level instead of duplicated per world.
        worlds = [
            {
                'agent': {
                    'ax': w.agent['ax'],
                    'ay': w.agent['ay'],
                    'last_action': w.agent.get('last_action'),
                    'animation_frame': w.agent['animation_frame'],
                },
                'bridge_piece': w.bridge_piece,
                'placed_bridge': w.placed_bridge,
            }
            for w in self.worlds
        ]
        first_world = self.worlds[0]
        return {
            'worlds': worlds,
            'episodes_completed': self.manager_steps,
            'milestones': self.milestones,
            'river_start_col': first_world.river_start_col,
            'house_start_col': first_world.house_start_col,
            'house_start_row': first_world.house_start_row,
        }
