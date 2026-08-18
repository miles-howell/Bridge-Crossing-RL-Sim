# simulation/views.py
from django.shortcuts import render
from django.http import JsonResponse
import json
import uuid

from .engine import SimulationEngine, ManagerAgent, WorkerAgent, GRID_ROWS, GRID_COLS

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

        manager_actions = ['GOTO_LOG', 'GOTO_RIVER', 'GOTO_FAR_BANK', 'GOTO_HOUSE']
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

    goals_to_visualize = engine.worlds[0].subgoal_locations

    states_to_visualize = {
        'GOTO_LOG':      {'has_bridge': False, 'bridge_placed': False, 'has_crossed': False},
        'GOTO_RIVER':    {'has_bridge': True,  'bridge_placed': False, 'has_crossed': False},
        'GOTO_FAR_BANK': {'has_bridge': False, 'bridge_placed': True,  'has_crossed': False},
        'GOTO_HOUSE':    {'has_bridge': False, 'bridge_placed': True,  'has_crossed': True},
    }

    all_q_maps = {}

    for vis_name, goal_coord in goals_to_visualize.items():
        state_conditions = states_to_visualize.get(vis_name, {})
        flags = (
            1 if state_conditions.get('has_bridge') else 0,
            1 if state_conditions.get('bridge_placed') else 0,
            1 if state_conditions.get('has_crossed') else 0,
        )
        q_map = worker.get_max_q_grid(goal_coord, GRID_ROWS, GRID_COLS, flags)
        all_q_maps[vis_name] = {'q_map': q_map}

    return JsonResponse({ 'q_maps': all_q_maps, 'rows': GRID_ROWS, 'cols': GRID_COLS })

# --- NEW: API endpoint for Manager's Q-values ---
def api_manager_q_values(request):
    """ API endpoint to fetch the Manager's Q-table for visualization. """
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'Only POST method is allowed.'}, status=405)

    engine_id = request.session.get('engine_id')
    if not engine_id or engine_id not in server_engines:
        return JsonResponse({'status': 'error', 'message': 'Brain not initialized.'}, status=400)

    engine = server_engines[engine_id]
    manager = engine.manager

    states = [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]
    q_rows = manager.get_all_q_values(states)
    serialized = {
        ",".join(map(str, state)): dict(zip(manager.actions, q_row))
        for state, q_row in zip(states, q_rows)
    }

    return JsonResponse({'manager_q_table': serialized})
