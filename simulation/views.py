# simulation/views.py
from django.shortcuts import render
from django.http import JsonResponse
import json
from collections import defaultdict
import uuid
import math

from .engine import SimulationEngine, RLAgent, SimulationWorld, ReplayBuffer, GRID_ROWS, GRID_COLS

# --- SERVER-SIDE CACHE ---
server_engines = {}

def index(request):
    """ Renders the main simulation page and clears any old session data. """
    if 'engine_id' in request.session and request.session['engine_id'] in server_engines:
        del server_engines[request.session['engine_id']]
    if 'engine_id' in request.session:
        del request.session['engine_id']
    return render(request, 'simulation/index.html')

def get_or_create_engine(session, data=None):
    """
    Gets the user's existing simulation engine from the server cache.
    If it doesn't exist, or if a reset is forced, it creates a new one using the optimized defaults.
    """
    engine_id = session.get('engine_id')
    if not engine_id or engine_id not in server_engines or (data and data.get('command') == 'reset'):
        engine_id = str(uuid.uuid4())
        session['engine_id'] = engine_id

        # --- UPDATED DEFAULTS to match optimized parameters ---
        actions = ["UP", "DOWN", "LEFT", "RIGHT"]
        agent_controller = RLAgent(
            actions,
            learning_rate=float(data.get('learningRate', 0.5)),
            discount_factor=float(data.get('discountFactor', 0.9)),
            exploration_rate=float(data.get('explorationRate', 0.6)),
            buffer_size=20000
        )

        visit_count = defaultdict(int)
        milestones = { 'picked_up': 0, 'placed': 0, 'crossed': 0, 'home': 0 }

        engine = SimulationEngine(
            num_agents=int(data.get('numAgents', 10)),
            agent_controller=agent_controller,
            visit_count=visit_count,
            milestones=milestones,
            curiosity_factor=float(data.get('curiosityFactor', 15)),
            time_limit_score=int(data.get('timeLimitScore', -600)),
            batch_size=int(data.get('batchSize', 32)),
            step_penalty=int(data.get('costOfLiving', 14))
        )
        server_engines[engine_id] = engine

    return server_engines[engine_id]


def api_state(request):
    """
    API endpoint for the simulation. Now fully stateful on the server.
    """
    if request.method == 'GET':
        engine_id = request.session.get('engine_id')
        if not engine_id or engine_id not in server_engines:
            return JsonResponse({'status': 'error', 'message': 'Simulation not initialized. Please reset.'}, status=400)

        engine = server_engines[engine_id]
        engine.update()
        return JsonResponse(engine.get_state())

    elif request.method == 'POST':
        data = json.loads(request.body)
        if data.get('command') == 'reset':
            engine = get_or_create_engine(request.session, data)
            return JsonResponse(engine.get_state())
        else:
            return JsonResponse({'status': 'error', 'message': 'Invalid command'}, status=400)

    else:
        return JsonResponse({'status': 'error', 'message': 'Unsupported method'}, status=405)


def api_q_values(request):
    """ API endpoint to fetch the Q-value maps for all visualization states. """
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'Only POST method is allowed.'}, status=405)

    engine_id = request.session.get('engine_id')
    if not engine_id or engine_id not in server_engines:
        return JsonResponse({'status': 'error', 'message': 'Brain not initialized.'}, status=400)

    engine = server_engines[engine_id]
    agent_controller = engine.agent_controller
    true_goal = engine.worlds[0].goal

    states_to_visualize = {
        'no_bridge':     {'has_bridge': False, 'bridge_placed': False, 'has_crossed': False},
        'has_bridge':    {'has_bridge': True,  'bridge_placed': False, 'has_crossed': False},
        'bridge_placed': {'has_bridge': False, 'bridge_placed': True,  'has_crossed': False},
        'crossed_bridge':{'has_bridge': False, 'bridge_placed': True,  'has_crossed': True},
    }

    all_q_maps = {}

    for state_name, state_conditions in states_to_visualize.items():
        q_map = []
        policy_map = []
        for r in range(GRID_ROWS):
            q_row = []; policy_row = []
            for c in range(GRID_COLS):
                agent_state = (c, r,
                               1 if state_conditions['has_bridge'] else 0,
                               1 if state_conditions['bridge_placed'] else 0,
                               1 if state_conditions['has_crossed'] else 0)

                q_values = {a: agent_controller.get_q_value(agent_state, true_goal, a) for a in agent_controller.actions}

                if not q_values or all(v == 0 for v in q_values.values()):
                    max_q = 0; best_action = 'NONE'
                else:
                    max_q = max(q_values.values()); best_action = max(q_values, key=q_values.get)
                q_row.append(max_q); policy_row.append(best_action)
            q_map.append(q_row); policy_map.append(policy_row)
        all_q_maps[state_name] = {'q_map': q_map, 'policy_map': policy_map}

    return JsonResponse({ 'q_maps': all_q_maps, 'rows': GRID_ROWS, 'cols': GRID_COLS })
