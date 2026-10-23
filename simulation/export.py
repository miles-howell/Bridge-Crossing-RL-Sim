# simulation/export.py
"""Builds the downloadable run report and network-weights dump for one
SimulationEngine. Callers hold engine.lock while these run so the snapshot
is consistent."""

import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from . import engine as eng

SCHEMA_VERSION = 1

MANAGER_STATES = [(a, b, c, d) for a in (0, 1) for b in (0, 1) for c in (0, 1) for d in (0, 1)]

# The (has_bridge, bridge_placed, has_crossed) flags each worker heatmap is
# drawn under: the task state in which the manager would normally issue
# that sub-goal.
WORKER_VIS_FLAGS = {
    'GOTO_LOG':      (0, 0, 0),
    'GOTO_RIVER':    (1, 0, 0),
    'GOTO_FAR_BANK': (0, 1, 0),
    'GOTO_HOUSE':    (0, 1, 1),
}

HOW_TO_READ = {
    'overview': (
        "Hierarchical RL run of 'Nigel's Quest': each agent must pick up a log, place it as a "
        "bridge over a river, cross, and walk into a house. A manager network picks one of four "
        "sub-goals (GOTO_LOG, GOTO_RIVER, GOTO_FAR_BANK, GOTO_HOUSE) from a 4-bit state; a worker "
        "network moves UP/DOWN/LEFT/RIGHT on a grid to reach the sub-goal's coordinate. Both are "
        "DQNs with target networks, replay, and per-state (manager) / per-goal (worker) epsilon. "
        "Grid coordinates are (x=column, y=row), origin top-left."
    ),
    'metadata': "When and from which code the run was exported, plus the engine's fixed constants.",
    'hyperparameters': "Values the engine was built with at reset (tick_ms is the dashboard's tick at export time).",
    'headline': (
        "Snapshot totals. manager_steps counts sub-goal decisions. success_rate is the share of the "
        "last success_sample finished episodes (home or drowned) that reached home."
    ),
    'timeseries': (
        "One sample every timeseries_interval manager steps (oldest dropped past timeseries_size). "
        "Milestone and episode counts are cumulative; losses are means over each agent's recent "
        "training batches. Use this to spot plateaus or regressions."
    ),
    'manager': (
        "All 16 manager states keyed 'has_log,bridge_placed,has_crossed,last_subgoal_failed'. "
        "Each has Q-values per sub-goal, the greedy (argmax) sub-goal, its current epsilon, and "
        "whether it has been visited. The correct plan is LOG -> RIVER -> FAR_BANK -> HOUSE from "
        "states 0,0,0,x / 1,0,0,x / 0,1,0,x / 0,1,1,x respectively."
    ),
    'worker': (
        "q_maps holds, per sub-goal, a rows x cols grid of max_a Q(s, a) with the flags shown, "
        "i.e. what the dashboard heatmap draws. visit_counts is the rows x cols grid of visits "
        "driving the novelty bonus. epsilon_by_goal is the worker's exploration rate per goal."
    ),
    'events': (
        "Concluded sub-goals, oldest first (bounded to event_history_size). outcome is one of "
        "milestone (the sub-goal achieved its milestone, +100 to the manager), off-plan milestone "
        "(a milestone or home happened under a different sub-goal, -100), reached, no progress "
        "(walked to the coordinate but it advanced nothing, -100), timeout (worker_max_steps "
        "without reaching it, -100) or drowned (-100). Manager states are 4-bit lists as above."
    ),
    'agents': "Each world's agent at export time: position, task flags, status, current sub-goal and steps spent on it.",
    'canvases': "If present: PNG data URLs of the dashboard's environment, worker heatmap and manager graph canvases, added by the browser.",
}

_git_commit = None


def git_commit():
    global _git_commit
    if _git_commit is None:
        try:
            _git_commit = subprocess.run(
                ['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parent,
                capture_output=True, text=True, timeout=5, check=True,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            _git_commit = ''
    return _git_commit or None


def state_key(state):
    return ",".join(map(str, state))


def manager_q_table(manager):
    """Q-values per action for every manager state, keyed like '0,1,0,0'."""
    q_rows = manager.get_all_q_values(MANAGER_STATES)
    return {state_key(s): dict(zip(manager.actions, q)) for s, q in zip(MANAGER_STATES, q_rows)}


def worker_q_maps(worker, subgoal_locations):
    """Max-Q grid per sub-goal under that sub-goal's usual task flags."""
    return {
        name: {'q_map': worker.get_max_q_grid(coord, eng.GRID_ROWS, eng.GRID_COLS, WORKER_VIS_FLAGS[name])}
        for name, coord in subgoal_locations.items()
    }


def _lr(agent):
    return agent.optimizer.param_groups[0]['lr']


def build_run_report(engine, tick_ms=None):
    manager, worker = engine.manager, engine.worker
    world0 = engine.worlds[0]
    now = time.time()

    q_table = manager_q_table(manager)
    manager_states = {}
    for s in MANAGER_STATES:
        key = state_key(s)
        q = q_table[key]
        manager_states[key] = {
            'q_values': q,
            'greedy_action': max(q, key=q.get),
            'epsilon': manager.epsilon_by_state.get(s, manager.epsilon_start),
            'visited': s in manager.epsilon_by_state,
        }

    goal_names = {coord: name for name, coord in world0.subgoal_locations.items()}
    visit_grid = [[worker.visit_counts.get((c, r), 0) for c in range(eng.GRID_COLS)] for r in range(eng.GRID_ROWS)]

    recent = engine.recent_outcomes
    return {
        'schema_version': SCHEMA_VERSION,
        'how_to_read': HOW_TO_READ,
        'metadata': {
            'exported_at': datetime.fromtimestamp(now, timezone.utc).isoformat(),
            'started_at': datetime.fromtimestamp(engine.started_at, timezone.utc).isoformat(),
            'wall_clock_seconds': round(now - engine.started_at, 3),
            'git_commit': git_commit(),
            'constants': {
                'grid_cols': eng.GRID_COLS,
                'grid_rows': eng.GRID_ROWS,
                'river_water_width': eng.RIVER_WATER_WIDTH,
                'river_border_width': eng.RIVER_BORDER_WIDTH,
                'river_total_width': eng.RIVER_TOTAL_WIDTH,
                'bridge_length_tiles': eng.BRIDGE_LENGTH_TILES,
                'worker_max_steps': eng.WORKER_MAX_STEPS,
                'worker_subgoal_bonus': eng.WORKER_SUBGOAL_BONUS,
                'drowning_penalty': eng.DROWNING_PENALTY,
                'novelty_bonus_scale': eng.NOVELTY_BONUS_SCALE,
                'her_k': eng.HER_K,
                'grad_clip_norm': eng.GRAD_CLIP_NORM,
                'success_window': eng.SUCCESS_WINDOW,
                'event_history_size': eng.EVENT_HISTORY_SIZE,
                'timeseries_interval': eng.TIMESERIES_INTERVAL,
                'timeseries_size': eng.TIMESERIES_SIZE,
            },
            'layout': {
                'subgoal_locations': {k: list(v) for k, v in world0.subgoal_locations.items()},
                'house_tiles': sorted(list(t) for t in world0.house_tiles),
                'river_start_col': world0.river_start_col,
                'bridge_row': world0.bridge_y_pos_tiles,
                'agent_start': [1, 1],
            },
        },
        'hyperparameters': {
            'agents': engine.num_agents,
            'discount_factor': manager.gamma,
            'manager_learning_rate': _lr(manager),
            'worker_learning_rate': _lr(worker),
            'batch_size': engine.batch_size,
            'step_penalty': engine.step_penalty,
            'tick_ms': tick_ms,
            'manager_epsilon': {'start': manager.epsilon_start, 'min': manager.epsilon_min, 'decay': manager.epsilon_decay},
            'worker_epsilon': {'start': worker.epsilon_start, 'min': worker.epsilon_min, 'decay': worker.epsilon_decay},
            'manager_buffer_capacity': manager.memory.memory.maxlen,
            'worker_buffer_capacity': worker.memory.memory.maxlen,
            'target_sync_freq': {'manager': manager.target_sync_freq, 'worker': worker.target_sync_freq},
        },
        'headline': {
            'manager_steps': engine.manager_steps,
            'milestones': dict(engine.milestones),
            'episodes': dict(engine.episode_outcomes),
            'success_rate': (sum(recent) / len(recent)) if recent else None,
            'success_sample': len(recent),
            'subgoals_concluded': engine.next_event_id,
            'train_steps': {'manager': manager.train_steps, 'worker': worker.train_steps},
            'buffer_sizes': {'manager': len(manager.memory), 'worker': len(worker.memory)},
        },
        'timeseries': list(engine.timeseries),
        'manager': {
            'actions': list(manager.actions),
            'state_order': ['has_log', 'bridge_placed', 'has_crossed', 'last_subgoal_failed'],
            'states': manager_states,
        },
        'worker': {
            'actions': list(worker.actions),
            'q_map_flags': {k: {'has_bridge': f[0], 'bridge_placed': f[1], 'has_crossed': f[2]} for k, f in WORKER_VIS_FLAGS.items()},
            'q_maps': {k: v['q_map'] for k, v in worker_q_maps(worker, world0.subgoal_locations).items()},
            'visit_counts': visit_grid,
            'epsilon_by_goal': [
                {'goal': list(g), 'subgoal': goal_names.get(g), 'epsilon': e}
                for g, e in sorted(worker.epsilon_by_goal.items())
            ],
        },
        'events': list(engine.event_history),
        'agents': [
            {
                'id': w.id,
                'position': [w.agent['ax'], w.agent['ay']],
                'status': w.agent['status'],
                'has_log': w.agent['has_bridge_piece'],
                'bridge_placed': bool(w.placed_bridge),
                'has_crossed': w.agent['has_crossed'],
                'last_subgoal_failed': w.last_subgoal_failed,
                'manager_state': list(engine._get_manager_state(w)),
                'current_subgoal': w.current_subgoal_name,
                'steps_on_subgoal': w.subgoal_steps,
            }
            for w in engine.worlds
        ],
    }


def _state_dict_lists(model):
    return {k: v.tolist() for k, v in model.state_dict().items()}


def build_weights_export(engine):
    """Raw weights of both networks and their target copies, as nested lists
    keyed by PyTorch state_dict names (layers 0, 2, 4 are the Linear layers
    of a Linear-ReLU-Linear-ReLU-Linear stack)."""
    manager, worker = engine.manager, engine.worker
    return {
        'schema_version': SCHEMA_VERSION,
        'exported_at': datetime.now(timezone.utc).isoformat(),
        'git_commit': git_commit(),
        'manager': {
            'input': ['has_log', 'bridge_placed', 'has_crossed', 'last_subgoal_failed'],
            'outputs': list(manager.actions),
            'model': _state_dict_lists(manager.model),
            'target_model': _state_dict_lists(manager.target_model),
        },
        'worker': {
            'input': ['x', 'y', 'has_log', 'bridge_placed', 'has_crossed', 'goal_x', 'goal_y'],
            'outputs': list(worker.actions),
            'model': _state_dict_lists(worker.model),
            'target_model': _state_dict_lists(worker.target_model),
        },
    }
