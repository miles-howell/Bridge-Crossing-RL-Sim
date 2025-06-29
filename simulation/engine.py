# simulation/engine.py

import random
import math
from collections import deque, defaultdict

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
    """ The Worker. Learns to achieve specific coordinate sub-goals using HER. """
    def __init__(self, actions, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1, buffer_size=20000):
        self.actions = actions; self.alpha = learning_rate; self.gamma = discount_factor; self.epsilon = exploration_rate
        self.q_table = defaultdict(lambda: 0.0)
        self.memory = ReplayBuffer(buffer_size)

    def get_q_value(self, state, goal, action):
        return self.q_table.get((state, goal, action), 0.0)

    def choose_action(self, state, goal):
        if random.random() < self.epsilon: return random.choice(self.actions)
        q_values = [self.get_q_value(state, goal, a) for a in self.actions]
        max_q = max(q_values)
        return random.choice([a for a, q in zip(self.actions, q_values) if q == max_q])

    def update_q_table(self, state, action, reward, next_state, goal, done):
        best_next_q = 0 if done else max([self.get_q_value(next_state, goal, a) for a in self.actions])
        current_q = self.get_q_value(state, goal, action)
        new_q = current_q + self.alpha * (reward + self.gamma * best_next_q - current_q)
        self.q_table[(state, goal, action)] = new_q

    def experience_replay(self, batch_size):
        minibatch = self.memory.sample(batch_size)
        for state, action, reward, next_state, goal, done in minibatch:
            self.update_q_table(state, action, reward, next_state, goal, done)


class ManagerAgent:
    """ The Manager. Learns the high-level strategy of which sub-goal to pursue. """
    def __init__(self, actions, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1):
        self.actions = actions; self.alpha = learning_rate; self.gamma = discount_factor; self.epsilon = exploration_rate
        self.q_table = defaultdict(lambda: 0.0)

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state):
        if random.random() < self.epsilon: return random.choice(self.actions)
        q_values = [self.get_q_value(state, a) for a in self.actions]
        max_q = max(q_values)
        return random.choice([a for a, q in zip(self.actions, q_values) if q == max_q])

    def update_q_table(self, state, action, reward, next_state):
        best_next_q = max([self.get_q_value(next_state, a) for a in self.actions])
        current_q = self.get_q_value(state, action)
        new_q = current_q + self.alpha * (reward + self.gamma * best_next_q - current_q)
        self.q_table[(state, action)] = new_q


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
        return 0 if achieved_goal == desired_goal else -1

    def update(self):
        for world in self.worlds:
            agent = world.agent
            if agent['status'] != AgentState.IN_PROGRESS:
                world.reset()

            if world.current_subgoal_name is None:
                self.manager_steps += 1
                manager_state = self._get_manager_state(world)
                subgoal_name = self.manager.choose_action(manager_state)
                world.current_subgoal_name = subgoal_name
                world.current_subgoal_coord = world.subgoal_locations[subgoal_name]

            worker_state = self._get_worker_state(world)
            worker_action = self.worker.choose_action(worker_state, world.current_subgoal_coord)
            agent['last_action'] = worker_action

            if worker_action == "UP": agent['ay'] -= 1
            elif worker_action == "DOWN": agent['ay'] += 1
            elif worker_action == "LEFT": agent['ax'] -= 1
            elif worker_action == "RIGHT": agent['ax'] += 1
            agent['ax'] = max(0, min(GRID_COLS - 1, agent['ax'])); agent['ay'] = max(0, min(GRID_ROWS - 1, agent['ay']))
            world.subgoal_steps += 1

            # --- FIX: DENSE REWARD CALCULATION FOR REAL TRAJECTORY ---
            reward = -self.step_penalty
            subgoal_achieved_this_tick = False

            if not agent['has_bridge_piece'] and world.bridge_piece:
                log_center_x = world.bridge_piece['ax'] + 1
                if (agent['ax'], agent['ay']) == (log_center_x, world.bridge_piece['ay']):
                    agent['has_bridge_piece'] = True; world.bridge_piece = None; self.milestones['picked_up'] += 1; reward += 100
                    if world.current_subgoal_name == 'GOTO_LOG': subgoal_achieved_this_tick = True

            if agent['has_bridge_piece'] and not world.placed_bridge:
                if (agent['ax'], agent['ay']) == world.subgoal_locations['GOTO_RIVER']:
                    world.placed_bridge = { "ax": world.river_start_col + RIVER_BORDER_WIDTH, "ay": world.bridge_y_pos_tiles }; agent['has_bridge_piece'] = False; self.milestones['placed'] += 1; reward += 100
                    if world.current_subgoal_name == 'GOTO_RIVER': subgoal_achieved_this_tick = True

            is_in_river_area = world.river_start_col < agent['ax'] < world.river_start_col + RIVER_TOTAL_WIDTH - 1
            if is_in_river_area:
                is_on_bridge = world.placed_bridge and world.placed_bridge['ay'] == agent['ay']
                if not is_on_bridge: agent['status'] = AgentState.DROWNED; reward -= 1000

            if world.placed_bridge and not agent['has_crossed']:
                if agent['ax'] > world.river_start_col + RIVER_TOTAL_WIDTH - 1:
                    agent['has_crossed'] = True; reward += 100; self.milestones['crossed'] += 1

            if agent['has_crossed'] and (agent['ax'], agent['ay']) == world.subgoal_locations['GOTO_HOUSE']:
                agent['status'] = AgentState.HOME; self.milestones['home'] += 1; reward += 1000
                if world.current_subgoal_name == 'GOTO_HOUSE': subgoal_achieved_this_tick = True

            next_worker_state = self._get_worker_state(world)
            achieved_goal = (agent['ax'], agent['ay'])
            world.subgoal_trajectory.append((worker_state, worker_action, reward, next_worker_state, achieved_goal)) # Store DENSE reward

            subgoal_timed_out = (world.subgoal_steps >= WORKER_MAX_STEPS)
            subtask_is_over = subgoal_achieved_this_tick or subgoal_timed_out or agent['status'] != AgentState.IN_PROGRESS

            if subtask_is_over:
                manager_reward = 100 if subgoal_achieved_this_tick else -100
                manager_state = self._get_manager_state(world)
                next_manager_state = self._get_manager_state(world)
                self.manager.update_q_table(manager_state, world.current_subgoal_name, manager_reward, next_manager_state)

                # Process worker's trajectory
                # 1. Store REAL trajectory with DENSE rewards
                for state, act, rew, next_s, ach_g in world.subgoal_trajectory:
                    done = (ach_g == world.current_subgoal_coord) and subgoal_achieved_this_tick
                    self.worker.memory.add(state, act, rew, next_s, world.current_subgoal_coord, done)

                # 2. Store IMAGINARY trajectory with SPARSE rewards
                if not subgoal_achieved_this_tick:
                    imaginary_goal = world.subgoal_trajectory[-1][4]
                    for state, act, _, next_s, ach_g in world.subgoal_trajectory:
                        h_reward = self._compute_hindsight_reward(ach_g, imaginary_goal)
                        h_done = (ach_g == imaginary_goal)
                        self.worker.memory.add(state, act, h_reward, next_s, imaginary_goal, h_done)

                if agent['status'] == AgentState.IN_PROGRESS:
                    world.current_subgoal_name = None; world.subgoal_steps = 0; world.subgoal_trajectory = []
                else:
                    world.reset()

            self.worker.experience_replay(self.batch_size)

    def get_state(self):
        return { 'worlds': [w.__dict__ for w in self.worlds], 'episodes_completed': self.manager_steps, 'milestones': self.milestones }
