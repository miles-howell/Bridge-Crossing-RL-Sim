# simulation/engine.py

import random
import math
from collections import defaultdict

# --- CONFIGURATION ---
TILE_SIZE = 40
RIVER_TOTAL_WIDTH = 5
BRIDGE_LENGTH_TILES = 3
HOUSE_WIDTH_TILES = 2
HOUSE_HEIGHT_TILES = 3
VISIT_COUNT_CAP = 100

class AgentState:
    IN_PROGRESS = 'Learning...'
    HOME = 'Reached home! :)'
    DROWNED = 'Drowned :('
    TIMED_OUT = 'Failed (Time Limit)'

class RLAgent:
    """ The SHARED brain that learns using Q-learning. """
    def __init__(self, actions, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1):
        self.actions = actions
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.q_table = defaultdict(lambda: 0)

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            q_values = [self.get_q_value(state, a) for a in self.actions]
            max_q = max(q_values)
            best_actions = [i for i, q in enumerate(q_values) if q == max_q]
            action_index = random.choice(best_actions)
            return self.actions[action_index]

    def update_q_table(self, state, action, reward, next_state):
        best_next_action_q = max([self.get_q_value(next_state, a) for a in self.actions])
        current_q = self.get_q_value(state, action)
        new_q = current_q + self.alpha * (reward + self.gamma * best_next_action_q - current_q)
        self.q_table[(state, action)] = new_q

class SimulationWorld:
    """ Represents one independent simulation instance for one agent. """
    def __init__(self, agent_id, canvas_width, canvas_height, cols, rows, river_start_col):
        self.id = agent_id
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.cols = cols
        self.rows = rows
        self.river_start_col = river_start_col
        self.house_start_col = self.cols - HOUSE_WIDTH_TILES - 1
        self.house_start_row = (self.rows // 2) - (HOUSE_HEIGHT_TILES // 2)
        # *** NEW: The bridge will always be placed in the middle ***
        self.bridge_y_pos = (self.rows // 2) * TILE_SIZE
        self.reset()

    def reset(self):
        self.agent = {
            "id": self.id,
            "x": TILE_SIZE,
            "y": self.canvas_height / 2,
            "has_bridge_piece": False,
            "has_crossed": False,
            "current_score": 0,
            "last_action": "RIGHT",
            "status": AgentState.IN_PROGRESS,
            "animation_frame": 0,
            "placed_bridge_this_episode": False
        }
        self.bridge_piece = None
        self.placed_bridge = None
        self._spawn_bridge_piece()

    def _get_discretized_pos(self, x, y):
        return int(x // TILE_SIZE), int(y // TILE_SIZE)

    def _spawn_bridge_piece(self):
        px = 4
        py = self.rows // 2
        self.bridge_piece = { "x": px * TILE_SIZE, "y": py * TILE_SIZE, "width": TILE_SIZE * BRIDGE_LENGTH_TILES, "height": TILE_SIZE }


class SimulationEngine:
    TILE_SIZE = TILE_SIZE

    def __init__(self, canvas_width, canvas_height, num_agents, agent_controller, visit_count, milestones, curiosity_factor, time_limit_score):
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.cols = math.floor(self.canvas_width / TILE_SIZE)
        self.rows = math.floor(self.canvas_height / TILE_SIZE)
        self.river_start_col = math.floor(self.cols / 2) - 2
        
        self.agent_controller = agent_controller
        self.visit_count = visit_count
        self.milestones = milestones
        
        self.num_agents = num_agents
        self.worlds = []
        self.episode_scores = []
        self.episodes_completed = 0
        
        self.curiosity_factor = curiosity_factor
        self.time_limit_score = time_limit_score

        for i in range(self.num_agents):
            self.worlds.append(SimulationWorld(i, canvas_width, canvas_height, self.cols, self.rows, self.river_start_col))

    def _get_world_state_tuple(self, world):
        agent = world.agent
        ax, ay = world._get_discretized_pos(agent['x'], agent['y'])
        has_piece = 1 if agent['has_bridge_piece'] else 0
        bridge_placed = 1 if world.placed_bridge else 0
        has_crossed = 1 if agent['has_crossed'] else 0
        return (ax, ay, has_piece, bridge_placed, has_crossed)

    def update(self):
        for world in self.worlds:
            agent = world.agent
            agent['animation_frame'] = (agent.get('animation_frame', 0) + 1) % 16
            
            state = self._get_world_state_tuple(world)
            action = self.agent_controller.choose_action(state)
            agent['last_action'] = action

            reward = -10
            
            if action == "UP": agent['y'] -= TILE_SIZE
            elif action == "DOWN": agent['y'] += TILE_SIZE
            elif action == "LEFT": agent['x'] -= TILE_SIZE
            elif action == "RIGHT": agent['x'] += TILE_SIZE
            
            agent['x'] = max(0, min(self.canvas_width - TILE_SIZE, agent['x']))
            agent['y'] = max(0, min(self.canvas_height - TILE_SIZE, agent['y']))

            new_ax, new_ay = world._get_discretized_pos(agent['x'], agent['y'])
            episode_over = False

            current_visits = self.visit_count.get((new_ax, new_ay), 0)
            if current_visits < VISIT_COUNT_CAP:
                self.visit_count[(new_ax, new_ay)] = current_visits + 1
            
            punishment_cap = -25 
            curiosity_bonus = self.curiosity_factor * (1 - (current_visits / 100.0))
            reward += max(punishment_cap, curiosity_bonus)

            if world.bridge_piece and not agent['has_bridge_piece']:
                bp_ax, bp_ay = world._get_discretized_pos(world.bridge_piece['x'], world.bridge_piece['y'])
                if new_ax >= bp_ax and new_ax < bp_ax + BRIDGE_LENGTH_TILES and new_ay == bp_ay:
                    agent['has_bridge_piece'] = True
                    world.bridge_piece = None
                    reward += 100
                    self.milestones['picked_up'] += 1

            if agent['has_bridge_piece'] and new_ax == world.river_start_col:
                # *** The bridge is now always placed at the static Y position ***
                world.placed_bridge = { "x": (world.river_start_col + 1) * TILE_SIZE, "y": world.bridge_y_pos }
                agent['has_bridge_piece'] = False
                agent['placed_bridge_this_episode'] = True
                reward += 200
                self.milestones['placed'] += 1
            
            is_in_river_area = world.river_start_col < new_ax < world.river_start_col + RIVER_TOTAL_WIDTH -1
            if is_in_river_area:
                on_bridge = False
                if world.placed_bridge and world.placed_bridge['y'] == new_ay * TILE_SIZE:
                    bridge_start_x = world.river_start_col + 1
                    bridge_end_x = bridge_start_x + BRIDGE_LENGTH_TILES
                    if bridge_start_x <= new_ax < bridge_end_x:
                        on_bridge = True
                        if not agent['has_crossed'] and new_ax == bridge_end_x - 1 and action == "RIGHT":
                            reward += 300
                            agent['has_crossed'] = True
                            self.milestones['crossed'] += 1
                
                if not on_bridge:
                    reward -= 1000
                    agent['status'] = AgentState.DROWNED
                    episode_over = True
            
            is_in_house = (world.house_start_col <= new_ax < world.house_start_col + HOUSE_WIDTH_TILES and
                           world.house_start_row <= new_ay < world.house_start_row + HOUSE_HEIGHT_TILES)
            if is_in_house:
                reward += 1000
                agent['status'] = AgentState.HOME
                episode_over = True
                self.milestones['home'] += 1

            agent['current_score'] += reward

            if not episode_over and agent['current_score'] < self.time_limit_score:
                if agent['placed_bridge_this_episode']:
                    agent['current_score'] -= 500
                agent['status'] = AgentState.TIMED_OUT
                episode_over = True

            next_state = self._get_world_state_tuple(world)
            self.agent_controller.update_q_table(state, action, reward, next_state)

            if episode_over:
                self.episode_scores.append(agent['current_score'])
                if len(self.episode_scores) > 20: self.episode_scores.pop(0)
                self.episodes_completed += 1
                world.reset()

    def get_state(self):
        return { 
            'worlds': [w.__dict__ for w in self.worlds], 
            'episode_scores': self.episode_scores, 
            'episodes_completed': self.episodes_completed,
            'milestones': self.milestones,
            'environment': { 'tile_size': self.TILE_SIZE }
        }
