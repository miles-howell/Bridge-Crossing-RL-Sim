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
CURIOSITY_REWARD_CAP = 50
CURIOSITY_PUNISHMENT_CAP = 125


class AgentState:
    IN_PROGRESS = 'Learning...'
    HOME = 'Reached home! :)'
    DROWNED = 'Drowned :('
    TIMED_OUT = 'Failed (Time Limit)'


class ReplayBuffer:
    """ A fixed-size buffer to store experience tuples for HER. """
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, goal, done):
        self.memory.append((state, action, reward, next_state, goal, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class RLAgent:
    """ An agent that learns using Q-learning with Hindsight Experience Replay. """
    def __init__(self, actions, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1, buffer_size=20000):
        self.actions = actions
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.q_table = defaultdict(lambda: 0)
        self.memory = ReplayBuffer(buffer_size)

    def get_q_value(self, state, goal, action):
        return self.q_table.get((state, goal, action), 0.0)

    def choose_action(self, state, goal):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            q_values = [self.get_q_value(state, goal, a) for a in self.actions]
            max_q = max(q_values)
            best_actions = [self.actions[i] for i, q in enumerate(q_values) if q == max_q]
            return random.choice(best_actions)

    def update_q_table(self, state, action, reward, next_state, goal, done):
        if done:
            best_next_action_q = 0
        else:
            best_next_action_q = max([self.get_q_value(next_state, goal, a) for a in self.actions])

        current_q = self.get_q_value(state, goal, action)
        new_q = current_q + self.alpha * (reward + self.gamma * best_next_action_q - current_q)
        self.q_table[(state, goal, action)] = new_q

    def experience_replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = self.memory.sample(batch_size)
        for state, action, reward, next_state, goal, done in minibatch:
            self.update_q_table(state, action, reward, next_state, goal, done)


class SimulationWorld:
    """ Represents one independent simulation instance for one agent. """
    def __init__(self, agent_id):
        self.id = agent_id
        self.river_start_col = (GRID_COLS // 2) - (RIVER_TOTAL_WIDTH // 2)
        self.house_start_col = GRID_COLS - HOUSE_WIDTH_TILES - 3
        self.house_start_row = 1
        self.bridge_y_pos_tiles = GRID_ROWS // 2
        self.goal = (self.house_start_col, self.house_start_row)
        self.reset()

    def reset(self):
        self.agent = { "id": self.id, "ax": 1, "ay": 1, "has_bridge_piece": False, "has_crossed": False, "current_score": 0, "last_action": "RIGHT", "status": AgentState.IN_PROGRESS, "animation_frame": 0 }
        self.placed_bridge = None
        self._spawn_bridge_piece()
        self.trajectory = []

    def _spawn_bridge_piece(self):
        self.bridge_piece = { "ax": 2, "ay": GRID_ROWS - 3 }


class SimulationEngine:
    def __init__(self, num_agents, agent_controller, visit_count, milestones, curiosity_factor, time_limit_score, batch_size=32, step_penalty=1):
        self.agent_controller = agent_controller
        self.visit_count = visit_count
        self.milestones = milestones
        self.num_agents = num_agents
        self.worlds = []
        self.episode_scores = []
        self.episodes_completed = 0
        self.curiosity_factor = curiosity_factor
        self.time_limit_score = time_limit_score
        self.batch_size = batch_size
        self.step_penalty = step_penalty

        for i in range(self.num_agents):
            self.worlds.append(SimulationWorld(i))

    def _get_agent_state_tuple(self, agent, world):
        return (agent['ax'], agent['ay'], 1 if agent['has_bridge_piece'] else 0, 1 if world.placed_bridge else 0, 1 if agent['has_crossed'] else 0)

    def _compute_hindsight_reward(self, achieved_goal, desired_goal):
        """ Computes the sparse reward for imaginary hindsight episodes. """
        return 0 if achieved_goal == desired_goal else -1

    def update(self):
        """ Runs one tick of the simulation for each agent. """
        for world in self.worlds:
            agent = world.agent
            if agent['status'] != AgentState.IN_PROGRESS:
                continue

            agent['animation_frame'] = (agent.get('animation_frame', 0) + 1) % 16

            current_state = self._get_agent_state_tuple(agent, world)
            action = self.agent_controller.choose_action(current_state, world.goal)
            agent['last_action'] = action

            # --- START DENSE REWARD CALCULATION ---
            # This is the reward for the REAL episode
            reward = -self.step_penalty

            prev_has_bridge = agent['has_bridge_piece']
            prev_has_crossed = agent['has_crossed']
            prev_bridge_placed = bool(world.placed_bridge)

            # Standard movement logic
            if action == "UP": agent['ay'] -= 1
            elif action == "DOWN": agent['ay'] += 1
            elif action == "LEFT": agent['ax'] -= 1
            elif action == "RIGHT": agent['ax'] += 1
            agent['ax'] = max(0, min(GRID_COLS - 1, agent['ax']))
            agent['ay'] = max(0, min(GRID_ROWS - 1, agent['ay']))
            new_ax, new_ay = agent['ax'], agent['ay']

            # --- FIX: Re-add the curiosity calculation to the dense reward ---
            current_visits = self.visit_count.get((new_ax, new_ay), 0)
            self.visit_count[(new_ax, new_ay)] = current_visits + 1

            curiosity_effect = 0
            if current_visits < CURIOSITY_REWARD_CAP:
                v_delta = CURIOSITY_REWARD_CAP - current_visits
                if v_delta > 0: curiosity_effect = self.curiosity_factor * (1 - (1 / math.sqrt(v_delta)))
            elif current_visits < CURIOSITY_PUNISHMENT_CAP:
                v_delta = (current_visits - CURIOSITY_REWARD_CAP) + 1
                if v_delta > 0: curiosity_effect = -self.curiosity_factor * (1 - (1 / math.sqrt(v_delta)))
            else:
                v_delta_max = (CURIOSITY_PUNISHMENT_CAP - 1 - CURIOSITY_REWARD_CAP) + 1
                if v_delta_max > 0: curiosity_effect = -self.curiosity_factor * (1 - (1/ math.sqrt(v_delta_max)))
            reward += curiosity_effect
            # --- END OF FIX ---

            # Milestone Rewards
            if world.bridge_piece and not prev_has_bridge and agent['ax'] >= world.bridge_piece['ax'] and agent['ax'] < world.bridge_piece['ax'] + BRIDGE_LENGTH_TILES and agent['ay'] == world.bridge_piece['ay']:
                agent['has_bridge_piece'] = True; world.bridge_piece = None; reward += 100; self.milestones['picked_up'] += 1
            if agent['has_bridge_piece'] and not prev_bridge_placed and agent['ax'] == world.river_start_col:
                world.placed_bridge = { "ax": world.river_start_col + RIVER_BORDER_WIDTH, "ay": world.bridge_y_pos_tiles }; agent['has_bridge_piece'] = False; reward += 100; self.milestones['placed'] += 1

            episode_over = False
            is_in_river_area = world.river_start_col < agent['ax'] < world.river_start_col + RIVER_TOTAL_WIDTH - 1
            if is_in_river_area:
                on_bridge = False
                if world.placed_bridge and world.placed_bridge['ay'] == agent['ay']:
                    bridge = world.placed_bridge; bridge_start_x = bridge['ax']; bridge_end_x = bridge_start_x + BRIDGE_LENGTH_TILES
                    if bridge_start_x <= agent['ax'] < bridge_end_x:
                        on_bridge = True
                        if not prev_has_crossed and agent['ax'] >= bridge_end_x - 1: agent['has_crossed'] = True; reward += 100; self.milestones['crossed'] += 1
                if not on_bridge: reward -= 1000; agent['status'] = AgentState.DROWNED; episode_over = True

            achieved_goal = (agent['ax'], agent['ay'])
            is_in_house = (achieved_goal == world.goal)
            if is_in_house and agent['has_crossed']: reward += 1000; agent['status'] = AgentState.HOME; self.milestones['home'] += 1; episode_over = True

            # Store the real experience in the trajectory
            next_state = self._get_agent_state_tuple(agent, world)
            world.trajectory.append((current_state, action, reward, next_state, achieved_goal, episode_over))

            # --- If the episode is over, perform Hindsight Replay ---
            if episode_over:
                # 1. Store the original episode with the true goal and DENSE rewards
                for state, act, rew, next_s, achieved_g, done in world.trajectory:
                    self.agent_controller.memory.add(state, act, rew, next_s, world.goal, done)

                # 2. Store hindsight episodes with imaginary goals and SPARSE rewards
                final_achieved_goal = world.trajectory[-1][4]
                for state, act, _, next_s, achieved_g, _ in world.trajectory:
                    hindsight_reward = self._compute_hindsight_reward(achieved_g, final_achieved_goal)
                    done = (achieved_g == final_achieved_goal)
                    self.agent_controller.memory.add(state, act, hindsight_reward, next_s, final_achieved_goal, done)

                self.episodes_completed += 1
                world.reset()

            # Train on a batch from memory on every tick
            self.agent_controller.experience_replay(self.batch_size)

    def get_state(self):
        return { 'worlds': [w.__dict__ for w in self.worlds], 'episode_scores': self.episode_scores, 'episodes_completed': self.episodes_completed, 'milestones': self.milestones }
