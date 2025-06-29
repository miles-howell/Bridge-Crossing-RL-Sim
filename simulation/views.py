# simulation/views.py
from django.shortcuts import render
from django.http import JsonResponse
import json
from collections import defaultdict
import uuid
import math

from .engine import SimulationEngine, ManagerAgent, WorkerAgent, SimulationWorld, GRID_ROWS, GRID_COLS

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
    If it doesn't exist, or if a reset is forced, it creates a new HRL engine.
    """
    engine_id = session.get('engine_id')
    if not engine_id or engine_id not in server_engines or (data and data.get('command') == 'reset'):
        engine_id = str(uuid.uuid4())
        session['engine_id'] = engine_id

        manager_actions = ['GOTO_LOG', 'GOTO_RIVER', 'GOTO_HOUSE']
        manager = ManagerAgent(
            manager_actions,
            learning_rate=float(data.get('learningRate', 0.1)),
            discount_factor=float(data.get('discountFactor', 0.9)),
            exploration_rate=float(data.get('explorationRate', 0.3))
        )

        worker_actions = ["UP", "DOWN", "LEFT", "RIGHT"]
        worker = WorkerAgent(
            worker_actions,
            learning_rate=float(data.get('learningRate', 0.6)),
            discount_factor=float(data.get('discountFactor', 0.9)),
            exploration_rate=float(data.get('explorationRate', 0.7)),
            buffer_size=20000
        )

        milestones = { 'picked_up': 0, 'placed': 0, 'crossed': 0, 'home': 0 }

        engine = SimulationEngine(
            manager=manager,
            worker=worker,
            num_agents=int(data.get('numAgents', 10)),
            milestones=milestones,
            batch_size=int(data.get('batchSize', 32)),
            step_penalty=int(data.get('costOfLiving', 14))
        )
        server_engines[engine_id] = engine

    return server_engines[engine_id]


def api_state(request):
    """ API endpoint for the HRL simulation. """
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
    """ API endpoint to fetch the Q-value maps for visualization. """
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'Only POST method is allowed.'}, status=405)

    engine_id = request.session.get('engine_id')
    if not engine_id or engine_id not in server_engines:
        return JsonResponse({'status': 'error', 'message': 'Brain not initialized.'}, status=400)

    engine = server_engines[engine_id]
    worker = engine.worker

    # --- FIX: Get subgoal locations from the engine's world template ---
    goals_to_visualize = engine.worlds[0].subgoal_locations

    states_to_visualize = {
        'no_bridge':     {'has_bridge': False, 'bridge_placed': False, 'has_crossed': False},
        'has_bridge':    {'has_bridge': True,  'bridge_placed': False, 'has_crossed': False},
        'bridge_placed': {'has_bridge': False, 'bridge_placed': True,  'has_crossed': False},
        'crossed_bridge':{'has_bridge': False, 'bridge_placed': True,  'has_crossed': True},
    }

    all_q_maps = {}

    for vis_name, goal_coord in goals_to_visualize.items():
        q_map = []
        # Use vis_name to select the correct state context
        state_conditions = states_to_visualize.get(vis_name.replace('GOTO_', '').lower(), {})

        for r in range(GRID_ROWS):
            q_row = []
            for c in range(GRID_COLS):
                agent_state = (c, r,
                               1 if state_conditions.get('has_bridge') else 0,
                               1 if state_conditions.get('bridge_placed') else 0,
                               1 if state_conditions.get('has_crossed') else 0)

                q_values = {a: worker.get_q_value(agent_state, goal_coord, a) for a in worker.actions}

                if not q_values or all(v == 0 for v in q_values.values()):
                    max_q = 0
                else:
                    max_q = max(q_values.values())
                q_row.append(max_q)
            q_map.append(q_row)
        # Match the old visualization keys
        vis_key = vis_name.replace('GOTO_LOG', 'no_bridge').replace('GOTO_RIVER', 'has_bridge').replace('GOTO_HOUSE', 'bridge_placed')
        all_q_maps[vis_key] = {'q_map': q_map}

    return JsonResponse({ 'q_maps': all_q_maps, 'rows': GRID_ROWS, 'cols': GRID_COLS })
