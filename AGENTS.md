# AGENTS.md

Guidance for AI coding agents working in this repo. `README.md` has the full write-up of the RL design; this file is the practical part: how to run things, where things live, and what will trip you up.

## What this is

A Django app that trains and visualizes a two-level hierarchical RL agent ("Nigel's Quest"). Each simulated agent must pick up a log, place it as a bridge over a river, cross, and walk into a house.

- **Manager** (DQN): picks one of four sub-goals (`GOTO_LOG`, `GOTO_RIVER`, `GOTO_FAR_BANK`, `GOTO_HOUSE`) from a 4-bit task state.
- **Worker** (goal-conditioned DQN with Hindsight Experience Replay): moves `UP/DOWN/LEFT/RIGHT` on a 20x12 grid toward the assigned sub-goal's coordinate.
- Training runs **server-side**. The browser polls `api/state/`, and each poll advances the simulation one tick.

## Run it

Python 3.13, `pip install -r requirements.txt` (installs the CPU-only PyTorch build via the extra index in `requirements.txt`, which skips the multi-GB CUDA packages), preferably in a venv.

**The app will not work until two things are set up:**

1. **Environment variables.** Copy the template: `cp .env.example .env`. `settings.py` reads it with python-dotenv.
   - `SECRET_KEY` is **required**. Without it every page returns a 500 (Django cannot sign the session cookie), and `manage.py test` errors in the view tests. Any non-empty string works locally.
   - `ALLOWED_HOSTS` is optional and comma-separated. Unset means localhost/127.0.0.1 only. Set it for any other hostname, or requests get a 400.
   - `.env` is loaded with `override=True`, so a value in `.env` beats one exported in your shell. If a variable "isn't taking effect", check `.env`.
2. **`python manage.py migrate`.** Sessions are stored in the SQLite database (`db.sqlite3`, git-ignored). The index page loads without it, but the first "reset"/start POST to `api/state/` fails with `no such table: django_session`.

Then:

```bash
python manage.py runserver      # http://127.0.0.1:8000/simulation/  (note the /simulation/ prefix; / is a 404)
python manage.py test           # needs SECRET_KEY; no migrate needed, the runner builds its own DB
python manage.py test simulation.tests.EngineRewardTests   # a single class
```

Without a `.env` you can pass them inline: `SECRET_KEY=x python manage.py test`.

There is no CI, linter, or formatter config. Match the style of the surrounding code.

## Layout

| Path | Role |
| --- | --- |
| `simulation/engine.py` | All RL logic: constants, `ReplayBuffer`, `WorkerAgent`, `ManagerAgent`, `SimulationWorld` (one agent's environment), `SimulationEngine` (steps all worlds, `get_state()` for the UI). |
| `simulation/views.py` | `index` plus JSON endpoints under `/simulation/api/`: `state` (GET = tick, POST `{"command":"reset", ...hyperparams}` = new engine), `q_values`, `manager_q_values`, `export`, `export/weights`. |
| `simulation/export.py` | Builds the downloadable run report and weights dump, plus the Q-table helpers the views reuse. |
| `simulation/templates/simulation/index.html` | The whole UI in one file (markup, CSS, canvas rendering, controls, export modal). Server URLs are injected with `{% url %}`. |
| `static/js/simulation_worker.js` | Web Worker that drives the polling loop and Q-value refreshes. |
| `static/images/` | Sprite sheets. `character.png` is Nigel (agent A01); `characters/*.png` are agents A02–A10. |
| `tools/sprites/` | Generator for the agent character sheets: `build_sheets.py` holds the shared body template, `characters/<name>.py` each character's palette and pixel grids. Needs Pillow (not in `requirements.txt`). Edit the `.py` and rebuild rather than editing the PNGs. |
| `simulation_backend/` | Django project: `settings.py`, `urls.py`, wsgi/asgi. |
| `simulation/tests.py` | Reward/termination tests (`SimpleTestCase`, hand-built worlds) and export tests. |

`simulation/models.py` and `admin.py` are empty; there are no models and no app migrations.

## Things that will trip you up

- **Engines live in a module-level dict.** `views.server_engines` maps a session's `engine_id` to its `SimulationEngine`, in process memory. State is lost on restart, loading `index` deliberately discards the session's engine, and running several worker processes would break it. Keep to a single process. Take `engine.lock` around anything that reads or mutates an engine.
- **Rewards are intentionally decoupled.** The manager gets +100/-100 for whether its sub-goal produced a milestone. The worker gets only a reward it fully controls: a step penalty, a novelty bonus, `WORKER_SUBGOAL_BONUS` for reaching the assigned coordinate, and `DROWNING_PENALTY`. It gets **no** task-milestone rewards. The engine comments and `tests.py` explain why; read them before changing any reward. `README.md` section 2.3 documents the numbers, so update it with any reward change.
- **The first world's `subgoal_locations` are used for all worlds** (the Q-value endpoints read `engine.worlds[0]`). Worlds are identical by construction, so keep it that way.
- **`/static/` is git-ignored, but the files in it are tracked.** A new asset needs `git add -f`. `staticfiles/` and `db.sqlite3` are ignored and should stay untracked.
- **`DEBUG = True` is hard-coded** in `settings.py`. Be careful if you touch deployment settings; `waitress` and `whitenoise` are in `requirements.txt`.
- **Export schema is versioned.** If you change the shape of the report or weights JSON in `export.py`, bump `SCHEMA_VERSION`.
- **Export filenames use the `nigel-` prefix** (`nigel-run-<timestamp>.json`), from the app's "Nigel's Quest" name.
- **`torch.set_num_threads(1)` is set deliberately** in `engine.py`; the networks are tiny and multithreading is slower. Don't remove it without measuring.
- **The README's "Technologies" list is partly stale** (it mentions Tailwind, jQuery, and Select2). Check `index.html` for what the page actually loads.

## Conventions

- Short, imperative commit subjects that say what changed and why it matters (e.g. "End the subtask when any milestone fires, not just the assigned one").
- Work happens on feature branches that are merged by pull request.
- Add or update tests in `simulation/tests.py` for engine behavior changes. Reward and termination tests drive `SimulationEngine.update()` from a hand-built world, so copy that pattern.
