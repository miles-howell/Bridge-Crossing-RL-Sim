# simulation/views.py
from django.shortcuts import render
from django.http import JsonResponse
import json
import uuid
from datetime import datetime, timezone

from .engine import SimulationEngine, ManagerAgent, WorkerAgent, GRID_ROWS, GRID_COLS
from .export import build_run_report, build_weights_export, manager_q_table, worker_q_maps

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
            learning_rate=float(data.get('managerLearningRate', 0.01)),
            discount_factor=float(data.get('discountFactor', 0.9)),
            exploration_rate=float(data.get('explorationRate', 0.3))
        )

        worker_actions = ["UP", "DOWN", "LEFT", "RIGHT"]
        worker = WorkerAgent(
            worker_actions,
            learning_rate=float(data.get('workerLearningRate', 0.001)),
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
            step_penalty=int(data.get('costOfLiving', 2))
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
        with engine.lock:
            engine.update()
            state = engine.get_state()
        return JsonResponse(state)

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
    with engine.lock:
        all_q_maps = worker_q_maps(engine.worker, engine.worlds[0].subgoal_locations)

    return JsonResponse({ 'q_maps': all_q_maps, 'rows': GRID_ROWS, 'cols': GRID_COLS })

# --- NEW: API endpoint for Manager's Q-values ---
def api_manager_q_values(request):
    """ API endpoint to fetch the Manager's learned Q-values for visualization. """
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'Only POST method is allowed.'}, status=405)

    engine_id = request.session.get('engine_id')
    if not engine_id or engine_id not in server_engines:
        return JsonResponse({'status': 'error', 'message': 'Brain not initialized.'}, status=400)

    engine = server_engines[engine_id]
    with engine.lock:
        serialized = manager_q_table(engine.manager)

    return JsonResponse({'manager_q_values': serialized})


def _export_response(payload, prefix):
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    response = JsonResponse(payload)
    response['Content-Disposition'] = f'attachment; filename="{prefix}-{stamp}.json"'
    return response


def _engine_for_export(request):
    if request.method != 'GET':
        return None, JsonResponse({'status': 'error', 'message': 'Only GET method is allowed.'}, status=405)
    engine_id = request.session.get('engine_id')
    if not engine_id or engine_id not in server_engines:
        return None, JsonResponse({'status': 'error', 'message': 'Simulation not initialized. Please reset.'}, status=400)
    return server_engines[engine_id], None


def api_export(request):
    """ Downloads a full, agent-readable report of the current run. """
    engine, error = _engine_for_export(request)
    if error:
        return error
    try:
        tick_ms = int(request.GET['tick_ms'])
    except (KeyError, ValueError):
        tick_ms = None
    # Holding the lock pauses training for the moment the snapshot takes.
    with engine.lock:
        report = build_run_report(engine, tick_ms=tick_ms)
    return _export_response(report, 'nigel-run')


def api_export_weights(request):
    """ Downloads the raw weights of the manager and worker networks. """
    engine, error = _engine_for_export(request)
    if error:
        return error
    with engine.lock:
        weights = build_weights_export(engine)
    return _export_response(weights, 'nigel-weights')
