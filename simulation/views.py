# simulation/views.py
from django.shortcuts import render
from django.http import JsonResponse
import json
from collections import defaultdict
import uuid
import math

from .engine import SimulationEngine, RLAgent, SimulationWorld

# --- SERVER-SIDE CACHE ---
server_brains = {}

def index(request):
    """ 
    Renders the main simulation page. 
    Clears out any old server-side brain or session data.
    """
    if 'brain_id' in request.session and request.session['brain_id'] in server_brains:
        del server_brains[request.session['brain_id']]
    if 'simulation_state' in request.session:
        del request.session['simulation_state']
    if 'brain_id' in request.session:
        del request.session['brain_id']
    return render(request, 'simulation/index.html')


def get_or_create_session_brain(session):
    """
    Looks for a brain for the current user session. If one doesn't exist, it creates it.
    """
    if 'brain_id' not in session or session['brain_id'] not in server_brains:
        session['brain_id'] = str(uuid.uuid4())
        server_brains[session['brain_id']] = {
            'agent_controller': None,
            'visit_count': defaultdict(int),
            'milestones': { 'picked_up': 0, 'placed': 0, 'crossed': 0, 'home': 0 }
        }
    return server_brains[session['brain_id']]


def api_state(request):
    """ API endpoint for the simulation. Now uses a server-side brain. """
    
    session_brain = get_or_create_session_brain(request.session)
    agent_controller = session_brain['agent_controller']
    visit_count = session_brain['visit_count']
    milestones = session_brain['milestones']
    engine = None

    if request.method == 'GET':
        if 'simulation_state' not in request.session or not agent_controller:
            return JsonResponse({'status': 'error', 'message': 'Simulation not initialized.'}, status=400)
        
        state = request.session['simulation_state']
        
        engine = SimulationEngine(
            canvas_width=state['canvas_width'],
            canvas_height=state['canvas_height'],
            num_agents=state['num_agents'],
            agent_controller=agent_controller,
            visit_count=visit_count,
            milestones=milestones,
            curiosity_factor=state['curiosity_factor'],
            time_limit_score=state['time_limit_score']
        )
        engine.worlds = []
        for world_dict in state['worlds']:
            world = SimulationWorld(world_dict['id'], world_dict['canvas_width'], world_dict['canvas_height'], world_dict['cols'], world_dict['rows'], world_dict['river_start_col'])
            world.agent = world_dict['agent']
            world.bridge_piece = world_dict['bridge_piece']
            world.placed_bridge = world_dict['placed_bridge']
            engine.worlds.append(world)
        engine.episode_scores = state['episode_scores']
        engine.episodes_completed = state['episodes_completed']
        
        engine.update()

    elif request.method == 'POST':
        data = json.loads(request.body)
        if data.get('command') == 'reset':
            request.session['brain_id'] = str(uuid.uuid4())
            actions = ["UP", "DOWN", "LEFT", "RIGHT"]
            agent_controller = RLAgent(actions, float(data.get('learningRate', 0.1)), exploration_rate=float(data.get('explorationRate', 0.5)))
            visit_count = defaultdict(int)
            milestones = { 'picked_up': 0, 'placed': 0, 'crossed': 0, 'home': 0 }
            server_brains[request.session['brain_id']] = {
                'agent_controller': agent_controller,
                'visit_count': visit_count,
                'milestones': milestones
            }

            canvas_size = data.get('canvasSize')
            num_agents = int(data.get('numAgents', 1))
            curiosity = float(data.get('curiosityFactor', 10))
            time_limit = int(data.get('timeLimitScore', -1000))

            engine = SimulationEngine(
                canvas_width=canvas_size['width'], 
                canvas_height=canvas_size['height'],
                num_agents=num_agents,
                agent_controller=agent_controller,
                visit_count=visit_count,
                milestones=milestones,
                curiosity_factor=curiosity,
                time_limit_score=time_limit
            )
        else:
            return JsonResponse({'status': 'error', 'message': 'Invalid command'}, status=400)
    
    else:
        return JsonResponse({'status': 'error', 'message': 'Unsupported method'}, status=405)
    
    state_to_save_and_send = {
        'canvas_width': engine.canvas_width,
        'canvas_height': engine.canvas_height,
        'num_agents': engine.num_agents,
        'worlds': [w.__dict__ for w in engine.worlds],
        'episode_scores': engine.episode_scores,
        'episodes_completed': engine.episodes_completed,
        'milestones': engine.milestones,
        'curiosity_factor': engine.curiosity_factor,
        'time_limit_score': engine.time_limit_score,
        'environment': { 'tile_size': engine.TILE_SIZE }
    }
    
    request.session['simulation_state'] = state_to_save_and_send
    
    return JsonResponse(state_to_save_and_send)


def api_q_values(request):
    """
    API endpoint to fetch the Q-value and policy map for visualization.
    """
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'Only POST method is allowed.'}, status=405)

    session_brain = get_or_create_session_brain(request.session)
    agent_controller = session_brain['agent_controller']

    if not agent_controller:
        return JsonResponse({'status': 'error', 'message': 'Brain not initialized.'}, status=400)

    try:
        data = json.loads(request.body)
        view_state = data.get('view_state', {})
        has_bridge = view_state.get('has_bridge_piece', False)
        bridge_placed = view_state.get('bridge_placed', False)
        has_crossed = view_state.get('has_crossed', False)
    except json.JSONDecodeError:
        return JsonResponse({'status': 'error', 'message': 'Invalid JSON.'}, status=400)

    sim_state = request.session.get('simulation_state')
    if not sim_state:
         return JsonResponse({'status': 'error', 'message': 'Main simulation state not found.'}, status=400)

    cols = math.floor(sim_state['canvas_width'] / SimulationEngine.TILE_SIZE)
    rows = math.floor(sim_state['canvas_height'] / SimulationEngine.TILE_SIZE)

    q_map = []
    policy_map = []
    
    for r in range(rows):
        q_row = []
        policy_row = []
        for c in range(cols):
            current_tile_state = (c, r, 1 if has_bridge else 0, 1 if bridge_placed else 0, 1 if has_crossed else 0)
            
            q_values = {a: agent_controller.get_q_value(current_tile_state, a) for a in agent_controller.actions}
            
            if not q_values or all(v == 0 for v in q_values.values()):
                max_q = 0
                best_action = 'NONE'
            else:
                max_q = max(q_values.values())
                best_action = max(q_values, key=q_values.get)

            q_row.append(max_q)
            policy_row.append(best_action)
        q_map.append(q_row)
        policy_map.append(policy_row)

    return JsonResponse({
        'q_map': q_map,
        'policy_map': policy_map,
        'rows': rows,
        'cols': cols
    })
