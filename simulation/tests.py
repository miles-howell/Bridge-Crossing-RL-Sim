from django.test import SimpleTestCase

from .engine import (
    AgentState, ManagerAgent, SimulationEngine, WorkerAgent, WORKER_SUBGOAL_BONUS,
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
                self.assertGreater(last[2], 1000 + WORKER_SUBGOAL_BONUS - 50)
                self.assertTrue(last[5])

    def test_home_under_wrong_subgoal_ends_episode_without_credit(self):
        self._place_crossed_agent((14, 2), 'GOTO_FAR_BANK')
        self._step("RIGHT")

        self.assertEqual(self.world.agent['status'], AgentState.HOME)
        self.assertEqual(self.engine.milestones['home'], 1)
        self.assertEqual(self._manager_reward(), -100)
        # Terminal for the manager.
        self.assertTrue(self.manager_updates[0][4])
        # No +1000 for the worker under the far-bank goal...
        primary = [e for e in self.engine.worker.memory.memory if e[4] == self.world.subgoal_locations['GOTO_FAR_BANK']]
        self.assertEqual(len(primary), 1)
        self.assertLess(primary[0][2], 100)
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
