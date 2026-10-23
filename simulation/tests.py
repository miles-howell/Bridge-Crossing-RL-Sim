import json
from unittest import mock

from django.test import SimpleTestCase, TestCase
from django.urls import reverse

from . import engine as engine_module
from .engine import (
    AgentState, DROWNING_PENALTY, ManagerAgent, SimulationEngine, WorkerAgent,
    WORKER_SUBGOAL_BONUS,
)


class EngineRewardTests(SimpleTestCase):
    """Drives one SimulationEngine.update() step from a hand-built world state,
    with the worker's move forced and the manager's training call recorded."""

    def setUp(self):
        manager = ManagerAgent(['GOTO_LOG', 'GOTO_RIVER', 'GOTO_FAR_BANK', 'GOTO_HOUSE'])
        worker = WorkerAgent(["UP", "DOWN", "LEFT", "RIGHT"])
        milestones = {'picked_up': 0, 'placed': 0, 'crossed': 0, 'home': 0}
        self.engine = SimulationEngine(manager, worker, num_agents=1, milestones=milestones)
        self.world = self.engine.worlds[0]

        self.manager_updates = []
        manager.train_step = lambda *args, **kwargs: self.manager_updates.append(args)
        # Keep replay from mutating anything the assertions read.
        manager.experience_replay = lambda batch_size: None
        worker.experience_replay = lambda batch_size: None

    def _place_crossed_agent(self, pos, subgoal_name):
        world = self.world
        world.agent.update(ax=pos[0], ay=pos[1], has_crossed=True)
        world.bridge_piece = None
        world.placed_bridge = {"ax": world.river_start_col + 1, "ay": world.bridge_y_pos_tiles}
        world.milestones_rewarded = {k: True for k in world.milestones_rewarded}
        world.current_subgoal_name = subgoal_name
        world.current_subgoal_coord = world.subgoal_locations[subgoal_name]
        world.last_manager_state = self.engine._get_manager_state(world)

    def _step(self, action):
        self.engine.worker.choose_action = lambda state, goal: action
        self.engine.update()

    def _manager_reward(self):
        self.assertEqual(len(self.manager_updates), 1)
        return self.manager_updates[0][2]

    def test_house_footprint_matches_sprite(self):
        w = self.world
        self.assertEqual(w.house_tiles, {(15, 1), (15, 2), (15, 3), (16, 1), (16, 2), (16, 3)})
        self.assertIn(w.subgoal_locations['GOTO_HOUSE'], w.house_tiles)

    def test_every_house_tile_counts_as_home(self):
        for col, row in sorted(self.world.house_tiles):
            with self.subTest(tile=(col, row)):
                self.setUp()
                # Step in from outside the footprint: left of col 15, right of col 16.
                if col == self.world.house_start_col:
                    start, action = (col - 1, row), "RIGHT"
                else:
                    start, action = (col + 1, row), "LEFT"
                self._place_crossed_agent(start, 'GOTO_HOUSE')
                self._step(action)

                self.assertEqual(self.world.agent['status'], AgentState.HOME)
                self.assertEqual(self.engine.milestones['home'], 1)
                self.assertEqual(self._manager_reward(), 100)
                last = self.engine.worker.memory.memory[-1]
                self.assertGreater(last[2], WORKER_SUBGOAL_BONUS - 50)
                self.assertTrue(last[5])

    def test_home_under_wrong_subgoal_ends_episode_without_credit(self):
        self._place_crossed_agent((14, 2), 'GOTO_FAR_BANK')
        self._step("RIGHT")

        self.assertEqual(self.world.agent['status'], AgentState.HOME)
        self.assertEqual(self.engine.milestones['home'], 1)
        self.assertEqual(self._manager_reward(), -100)
        # Terminal for the manager.
        self.assertTrue(self.manager_updates[0][4])
        # No reward for the worker under the far-bank goal...
        primary = [e for e in self.engine.worker.memory.memory if e[4] == self.world.subgoal_locations['GOTO_FAR_BANK']]
        self.assertEqual(len(primary), 1)
        self.assertLess(primary[0][2], 50)
        # ...but the hindsight relabel records reaching the house tile.
        relabeled = [e for e in self.engine.worker.memory.memory if e[4] == (15, 2)]
        self.assertEqual(len(relabeled), 1)
        self.assertTrue(relabeled[0][5])

    def test_reaching_stale_subgoal_ends_subtask_as_failure(self):
        far_bank = self.world.subgoal_locations['GOTO_FAR_BANK']
        self._place_crossed_agent((far_bank[0] + 1, far_bank[1]), 'GOTO_FAR_BANK')
        self._step("LEFT")

        self.assertEqual(self.world.agent['status'], AgentState.IN_PROGRESS)
        self.assertEqual(self.world.subgoal_steps, 0)
        self.assertIsNone(self.world.current_subgoal_name)
        self.assertTrue(self.world.last_subgoal_failed)
        self.assertEqual(self._manager_reward(), -100)
        # Worker still gets its bonus for following orders, recorded once
        # (no duplicate hindsight relabel onto the same goal).
        entries = list(self.engine.worker.memory.memory)
        self.assertEqual(len(entries), 1)
        self.assertGreater(entries[0][2], WORKER_SUBGOAL_BONUS - 50)
        self.assertTrue(entries[0][5])

    def test_off_goal_milestone_ends_subtask_as_failure(self):
        world = self.world
        log = world.subgoal_locations['GOTO_LOG']
        # Walk onto the log while assigned a different goal.
        world.agent.update(ax=log[0] - 1, ay=log[1])
        world.current_subgoal_name = 'GOTO_RIVER'
        world.current_subgoal_coord = world.subgoal_locations['GOTO_RIVER']
        world.last_manager_state = self.engine._get_manager_state(world)
        self._step("RIGHT")

        self.assertTrue(world.agent['has_bridge_piece'])
        self.assertEqual(self.engine.milestones['picked_up'], 1)
        # The manager gets no credit for the lucky pickup and re-plans
        # from the new state.
        self.assertEqual(self._manager_reward(), -100)
        self.assertEqual(self.manager_updates[0][3][0], 1)  # next state has the log
        self.assertTrue(world.last_subgoal_failed)
        self.assertIsNone(world.current_subgoal_name)
        self.assertEqual(world.agent['status'], AgentState.IN_PROGRESS)
        # The milestone adds nothing to the worker's reward under GOTO_RIVER...
        primary = [e for e in self.engine.worker.memory.memory if e[4] == world.subgoal_locations['GOTO_RIVER']]
        self.assertEqual(len(primary), 1)
        self.assertLess(primary[0][2], 50)
        self.assertFalse(primary[0][5])
        # ...and the hindsight relabel records reaching the log tile.
        relabeled = [e for e in self.engine.worker.memory.memory if e[4] == log]
        self.assertEqual(len(relabeled), 1)
        self.assertTrue(relabeled[0][5])

    def test_drowning_penalty(self):
        world = self.world
        x = world.river_start_col + 1
        world.agent.update(ax=x - 1, ay=world.bridge_y_pos_tiles + 1)
        world.current_subgoal_name = 'GOTO_RIVER'
        world.current_subgoal_coord = world.subgoal_locations['GOTO_RIVER']
        world.last_manager_state = self.engine._get_manager_state(world)
        self._step("RIGHT")

        self.assertEqual(world.agent['status'], AgentState.DROWNED)
        primary = [e for e in self.engine.worker.memory.memory if e[4] == world.current_subgoal_coord]
        self.assertEqual(len(primary), 1)
        self.assertLess(primary[0][2], -DROWNING_PENALTY + 50)
        self.assertTrue(primary[0][5])

    def test_drowning_tick_is_visible_in_state(self):
        # The map's splash and log icon read these get_state fields: the
        # drowning tick still shows the agent on its water tile, no longer in
        # progress (and still carrying the log); the next tick resets it.
        world = self.world
        x = world.river_start_col + 1
        world.agent.update(ax=x - 1, ay=world.bridge_y_pos_tiles + 1, has_bridge_piece=True)
        world.bridge_piece = None
        world.milestones_rewarded['log_picked_up'] = True
        world.current_subgoal_name = 'GOTO_RIVER'
        world.current_subgoal_coord = world.subgoal_locations['GOTO_RIVER']
        world.last_manager_state = self.engine._get_manager_state(world)
        self._step("RIGHT")

        drowned = self.engine.get_state()['worlds'][0]
        self.assertFalse(drowned['in_progress'])
        self.assertEqual((drowned['agent']['ax'], drowned['agent']['ay']), (x, world.bridge_y_pos_tiles + 1))
        self.assertEqual(drowned['manager_state'][0], 1)

        self.engine.update()
        reset = self.engine.get_state()['worlds'][0]
        self.assertTrue(reset['in_progress'])
        self.assertEqual((reset['agent']['ax'], reset['agent']['ay']), (1, 1))
        self.assertEqual(reset['manager_state'][0], 0)

    def test_concluded_subgoal_is_logged_as_event(self):
        self._place_crossed_agent((14, 2), 'GOTO_FAR_BANK')
        self._step("RIGHT")

        events = self.engine.get_state()['events']
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['subgoal'], 'GOTO_FAR_BANK')
        self.assertEqual(events[0]['outcome'], 'off-plan milestone')
        self.assertEqual(events[0]['reward'], -100)

    def test_success_rate_counts_finished_episodes(self):
        self.assertIsNone(self.engine.get_state()['success_rate'])
        self.world.agent['status'] = AgentState.HOME
        self.engine.update()
        self.world.agent['status'] = AgentState.DROWNED
        self.engine.update()

        state = self.engine.get_state()
        self.assertEqual(state['success_rate'], 0.5)
        self.assertEqual(state['success_sample'], 2)


class ExportTests(TestCase):
    REPORT_SECTIONS = {
        'schema_version', 'how_to_read', 'metadata', 'hyperparameters', 'headline',
        'timeseries', 'manager', 'worker', 'events', 'agents',
    }

    def _reset(self):
        response = self.client.post(
            reverse('api_state'),
            data=json.dumps({'command': 'reset', 'numAgents': 2, 'batchSize': 8}),
            content_type='application/json',
        )
        self.assertEqual(response.status_code, 200)

    def test_export_requires_a_running_simulation(self):
        self.assertEqual(self.client.get(reverse('api_export')).status_code, 400)
        self.assertEqual(self.client.get(reverse('api_export_weights')).status_code, 400)

    def test_report_has_every_section(self):
        self._reset()
        for _ in range(5):
            self.client.get(reverse('api_state'))
        response = self.client.get(reverse('api_export'), {'tick_ms': 40})
        self.assertEqual(response.status_code, 200)
        self.assertIn('nigel-run-', response['Content-Disposition'])

        report = response.json()
        self.assertEqual(set(report), self.REPORT_SECTIONS)
        self.assertEqual(set(report['how_to_read']) - {'overview', 'canvases'}, self.REPORT_SECTIONS - {'schema_version', 'how_to_read'})
        self.assertEqual(report['hyperparameters']['agents'], 2)
        self.assertEqual(report['hyperparameters']['batch_size'], 8)
        self.assertEqual(report['hyperparameters']['tick_ms'], 40)
        self.assertEqual(len(report['manager']['states']), 16)
        self.assertEqual(set(report['worker']['q_maps']), {'GOTO_LOG', 'GOTO_RIVER', 'GOTO_FAR_BANK', 'GOTO_HOUSE'})
        self.assertEqual(len(report['worker']['visit_counts']), engine_module.GRID_ROWS)
        self.assertEqual(len(report['agents']), 2)

    def test_weights_export(self):
        self._reset()
        weights = self.client.get(reverse('api_export_weights')).json()
        self.assertEqual(len(weights['manager']['model']['0.weight']), 64)
        self.assertEqual(len(weights['worker']['target_model']['4.weight']), 4)


class ExportBoundsTests(SimpleTestCase):
    def test_history_stays_bounded_on_long_runs(self):
        from .export import build_run_report
        with mock.patch.multiple(engine_module, EVENT_HISTORY_SIZE=10, TIMESERIES_INTERVAL=1, TIMESERIES_SIZE=5):
            engine = SimulationEngine(
                ManagerAgent(['GOTO_LOG', 'GOTO_RIVER', 'GOTO_FAR_BANK', 'GOTO_HOUSE']),
                WorkerAgent(["UP", "DOWN", "LEFT", "RIGHT"]),
                num_agents=2, milestones={'picked_up': 0, 'placed': 0, 'crossed': 0, 'home': 0}, batch_size=4,
            )
            for _ in range(400):
                engine.update()

        self.assertGreater(engine.next_event_id, 10)
        report = build_run_report(engine)
        self.assertEqual(len(report['events']), 10)
        self.assertEqual(len(report['timeseries']), 5)
        self.assertEqual(report['events'][-1]['id'], engine.next_event_id - 1)
        self.assertEqual(report['timeseries'][-1]['manager_step'], engine.manager_steps)
        json.dumps(report)
