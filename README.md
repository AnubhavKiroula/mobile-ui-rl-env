# mobile-ui-rl-env

RL-style mobile UI agent environment built with Prime Intellect Verifiers concepts. Simulates a 4-screen mobile app with structured actions, a multi-component reward rubric, and an evaluation pipeline.

## 1. Overview

`mobile_ui_env` is a single-turn reinforcement learning environment that simulates a simple mobile app with four screens — home, notes, settings, and profile. An agent receives a natural-language instruction (e.g. *"Create a note titled Buy milk"*), produces a sequence of JSON-formatted actions (`tap`, `type`, `back`, `finish`), and receives a scalar reward computed by a weighted rubric. The environment is designed to be structurally compatible with the [Prime Intellect Verifiers](https://github.com/verifiers-for-code/verifiers) framework: it exposes a `load_environment()` entry point that returns datasets, a rubric, and an env factory, mirroring the interface that PRIME-RL training loops expect. The 30-task dataset (20 train / 10 eval) covers note creation, settings toggling, information retrieval, navigation safety, and multi-step sequencing.

## 2. Project Structure

```
mobile-ui-rl-env/
├── pyproject.toml                 # Package config (hatchling), dependencies, entry points
├── README.md                      # This file
├── AI_USAGE.md                    # AI tool usage log
├── run_eval.py                    # Evaluation runner with heuristic + dummy baselines
├── run_eval_output.txt            # Captured terminal output from run_eval.py
│
├── mobile_ui_env/                 # Main Python package
│   ├── __init__.py                # Exports: AppState, MobileUIEnv, load_environment
│   ├── state.py                   # AppState dataclass — screen, notes, toggles, counters
│   ├── actions.py                 # parse_actions, validate_action, apply_action, execute_action
│   ├── dataset.py                 # 30 tasks (TASKS list), build_dataset, get_task_by_id, stats
│   ├── rubric.py                  # 6 reward/penalty functions, Rubric class, build_rubric
│   └── env.py                     # MobileUIEnv (reset/step/run), load_environment()
│
└── tests/                         # pytest test suite
    ├── __init__.py
    ├── test_actions.py            # 18 tests: navigation, parsing, state mutations
    ├── test_rewards.py            # 19 tests: reward functions, rubric scoring
    └── test_env.py                # 9 tests: env lifecycle, truncation, loader
```

## 3. Setup & Installation

```bash
# Clone the repository
git clone https://github.com/AnubhavKiroula/mobile-ui-rl-env.git
cd mobile-ui-rl-env

# Create and activate a virtual environment (Python >= 3.9)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# Install in editable mode
pip install -e .
```

The only runtime dependency is `pytest` (used for testing). The optional `verifiers` package is imported via try/except and is not required to run the environment.

## 4. Running Locally

```bash
# Run the full test suite
pytest tests/ -v

# Run evaluation with the heuristic baseline agent (should show 100% success)
python run_eval.py

# Run evaluation with the dummy agent (should show 0% success)
python run_eval.py --dummy

# Debug a single task
python run_eval.py --task task_021

# Use the environment interactively
python -c "
from mobile_ui_env import load_environment
env_config = load_environment()
task = env_config.eval_dataset[0]
env = env_config.make_env(task)
obs = env.reset()
print(obs)
result = env.run([
    {'action': 'tap', 'target': 'notes_button'},
    {'action': 'tap', 'target': 'add_note_button'},
    {'action': 'type', 'target': 'note_input', 'text': 'Schedule dentist appointment'},
    {'action': 'tap', 'target': 'save_note_button'},
    {'action': 'finish'},
])
print(f'Success: {result[\"score\"][\"success\"]}, Reward: {result[\"score\"][\"total\"]}')
"
```

### 4.1 Running with Docker

You can run the tests and evaluations inside a Docker container:

```bash
# Build the Docker image
docker build -t mobile-ui-rl-env .

# Run the test suite (default command runs pytest)
docker run --rm mobile-ui-rl-env

# Run the evaluation script
docker run --rm mobile-ui-rl-env python run_eval.py
```

## 5. Design & RL Fundamentals

### 5.1 State space

The environment state is represented by `AppState` (defined in `state.py`), a Python dataclass with these fields:

| Field | Type | Default | Description |
|---|---|---|---|
| `current_screen` | `str` | `"home"` | One of: `home`, `notes`, `settings`, `profile` |
| `notes` | `list[str]` | `[]` | Saved note texts |
| `focus_mode` | `bool` | `False` | Settings toggle |
| `notifications` | `bool` | `True` | Settings toggle |
| `username` | `str` | `"john_doe"` | Profile data |
| `email` | `str` | `"john@example.com"` | Profile data |
| `app_version` | `str` | `"1.0.0"` | Settings metadata |
| `current_note_draft` | `str` | `""` | In-progress typed text |
| `step_count` | `int` | `0` | Total actions applied |
| `invalid_action_count` | `int` | `0` | Failed action attempts |
| `safety_violations` | `int` | `0` | Unsafe actions (logout) |
| `observed_info` | `dict` | `{}` | Info the agent has "seen" |
| `done` | `bool` | `False` | Episode termination flag |

**Partial observability is a deliberate design choice.** The agent's observation (returned by `get_observation()`) shows the current screen name and available elements, but fields like `username`, `email`, and `app_version` are only surfaced through `observed_info` after the agent explicitly taps the corresponding label element. This means the agent cannot "cheat" by reading ground-truth state directly — it must perform the correct sequence of taps to observe information, just as a real user would need to navigate to a screen to see its contents.

### 5.2 Action space

Four action types are supported, each represented as a JSON dictionary:

| Action | Required Keys | Valid When |
|---|---|---|
| `tap` | `action`, `target` | `target` is in `state.get_available_elements()` for the current screen |
| `type` | `action`, `target`, `text` | `target == "note_input"` and `current_screen == "notes"` |
| `back` | `action` | `current_screen != "home"` (nowhere to go back from home) |
| `finish` | `action` | Always valid — signals the agent is done |

`validate_action()` checks these rules and returns `(is_valid, reason)` without ever raising an exception. Invalid actions increment `state.invalid_action_count` but do not modify the environment state. Each screen exposes a fixed set of elements (e.g. home has `notes_button`, `settings_button`, `profile_button`), defined in the `SCREEN_ELEMENTS` constant in `state.py`.

### 5.3 Episode termination condition

An episode ends under two distinct conditions — and the distinction matters for RL semantics:

1. **Done (agent-initiated)**: The agent calls `{"action": "finish"}`, which sets `state.done = True`. This is a proper termination — the agent has declared its task complete. Tapping `logout_button` also sets `done = True` (a safety-violation terminal). The `done` flag means "the episode ended because the agent chose to stop or triggered a terminal state."

2. **Truncated (step-limit-initiated)**: If `state.step_count >= task["max_steps"]` before the agent calls `finish`, the episode is cut off and `result["truncated"] = True`. This is an artificial time limit, not a natural terminal state. In RL training, truncated episodes require bootstrapping from the value function at the cutoff point rather than treating the final state as terminal — conflating the two biases the value estimate.

The `MobileUIEnv.step()` method tracks and reports both flags independently.

### 5.4 Sparse vs dense rewards

The rubric in `rubric.py` has six components, split between sparse and dense:

**Sparse rewards** (binary, only meaningful at episode end):
- `success_reward`: 1.0 if the goal is achieved, 0.0 otherwise. Checked against ground-truth state (e.g. `"Buy milk" in state.notes`).
- `safety_penalty`: 1.0 if `state.safety_violations > 0`, else 0.0. Only triggered by tapping `logout_button`.

**Dense/shaped rewards** (provide signal throughout the episode):
- `format_reward`: Penalizes malformed action dictionaries (missing `action` key, wrong type, missing `target`/`text`). Starts at 1.0 and subtracts per violation.
- `efficiency_reward`: Rewards completing the task in fewer steps. Returns 0.0 if the task wasn't completed (gates on `success_reward == 1.0`), so agents can't game efficiency without actually succeeding.
- `invalid_action_penalty`: 0.1 per invalid action, capped at 1.0.
- `partial_progress_reward`: Awards 0.5 for reaching the correct screen, 0.3 for navigating away from home, 0.1 for any valid action. Returns 0.0 if success already achieved (no double-counting).

**Why this matters:** If we only used `success_reward`, the agent would receive zero learning signal on every failed episode. Early in training, when the agent is essentially random, it almost never stumbles into the correct 5-action sequence to create a note — this is the credit assignment problem. The shaped rewards (`partial_progress_reward`, `format_reward`) give gradient even on failed episodes: "you at least got to the right screen" is more informative than "you failed", enabling the policy to improve before it can reliably succeed end-to-end.

### 5.5 Reward hacking risks in THIS environment

**Risk 1: Immediate `finish` to avoid penalties.**
An agent could learn to call `{"action": "finish"}` as its only action. This avoids `safety_penalty` (never encounters `logout_button`), avoids `invalid_action_penalty` (one valid action), and earns full `format_reward` (1.0). It gets `partial_progress_reward` of 0.1 (produced at least one valid action). Total: ~0.11. Mitigation: `success_reward` has weight 1.0, dominating the total. The 0.11 ceiling for "do nothing" is far below the 1.0+ ceiling for actually completing tasks, so any reasonable optimization pressure pushes past this local optimum.

**Risk 2: Farming `partial_progress_reward` without completing the goal.**
An agent could navigate to the correct screen (earning 0.5 partial credit) and then call `finish`, earning partial + format rewards without doing the hard part (typing and saving a note). Mitigation: `partial_progress_reward` returns 0.0 if `success_reward == 1.0`, so it only fires on failed episodes. Its weight (0.1) is deliberately low, making the ceiling for "navigate but don't complete" ~0.15, far below the 1.3 ceiling for a successful optimal run.

**Risk 3: Producing syntactically valid but semantically useless actions.**
An agent could output many well-formed `{"action": "back"}` or `{"action": "tap", "target": "notes_button"}` actions in a loop. These earn full `format_reward` (1.0 × 0.1 = 0.1 contribution) and avoid `invalid_action_penalty`. Mitigation: `efficiency_reward` only pays out when `success_reward == 1.0`, so wasting steps yields zero efficiency credit. The `max_steps` truncation also caps how long an agent can loop without finishing.

**Risk 4: Case-insensitive note matching exploitation.**
`success_reward` uses case-insensitive comparison (`n.lower() == title.lower()`), so an agent typing "buy MILK" matches "Buy milk". This is intentional (robustness), but in a more complex env it could enable near-duplicate gaming. Mitigation: The `type` action writes exactly what the agent provides into `current_note_draft`, and the task specifies the exact expected title — the case-insensitivity is a controlled tolerance, not an exploitable loophole in this task set.

## 6. Scaling to a Real Android Emulator

### State space mapping

`AppState`'s fields map to real Android signals as follows:

- **`current_screen`** → the current `Activity` or `Fragment` name, obtainable via `adb shell dumpsys activity activities` or the UI Automator accessibility tree's root node.
- **Screen elements** (`SCREEN_ELEMENTS`) → nodes in the accessibility tree / UI Automator hierarchy. Each node has attributes: `resource-id` (e.g. `com.app:id/notes_button`), `text`, `content-desc`, `bounds` (screen coordinates), `clickable`, `enabled`. The symbolic names used in this mock env (e.g. `"notes_button"`) would map to `resource-id` values.
- **`notes`** → would require reading a `RecyclerView`/`ListView` adapter's contents via the accessibility tree, or querying the app's local database via `adb shell content query`.
- **`focus_mode`**, **`notifications`** → reading `Switch` or `ToggleButton` state from the UI tree's `checked` attribute, or querying `SharedPreferences` via instrumentation.
- **`observed_info`** → tracking which `TextView` elements the agent has interacted with, reading their `text` attribute from the accessibility tree after tap.

### Action space mapping

| Mock Action | Real ADB / UI Automator Equivalent |
|---|---|
| `tap target` | `adb shell input tap <x> <y>` (coordinates from node bounds center), or `uiautomator.click(resourceId="com.app:id/notes_button")` |
| `type text` | `adb shell input text "Buy milk"` after focusing the `EditText` via tap, handling special characters with key events |
| `back` | `adb shell input keyevent KEYCODE_BACK` (keyevent 4) |
| `finish` | No direct Android equivalent — this is a policy-level declaration that the agent considers its task complete. It would map to a trainer-side episode boundary, not an app action. |

### Action grounding problem

The mock env addresses elements by symbolic name (`"notes_button"`), which is a simplification. In a real emulator, the agent must resolve a semantic target to either:
- A `resource-id` via the accessibility tree (can fail: id collisions across fragments, dynamically-generated IDs, missing content descriptions)
- Screen coordinates via OCR or visual grounding (can fail: element off-screen, obscured by overlays, mid-animation)

This is strictly harder than the current invalid-action problem. In the mock env, `validate_action` just checks membership in a static list. On a real device, "element not found" has multiple failure modes: the element exists but is scrolled off-screen, the element is animating and its bounds are stale, the element's resource-id changed after an app update, or multiple elements share the same id. The action grounding layer must handle retries, scrolling, and wait-for-idle before declaring an action invalid.

### Observation space change

Three options, each with tradeoffs:

1. **Raw screenshot (pixels)**: Complete visual information, no app instrumentation needed. But requires a vision model (CNN/ViT) to parse, is computationally expensive, and pixel-level features don't directly encode semantic structure (is that grey rectangle a button or a label?).
2. **Accessibility / UI tree (structured XML)**: Provides semantic element types, text, bounds, and state flags. Cheaper to process, directly actionable. But some apps have incomplete accessibility annotations, and dynamic content (WebViews, custom canvas views) may not appear in the tree.
3. **Multi-modal (screenshot + UI tree)**: Best coverage — the tree provides structure and the screenshot fills gaps. This is the approach used by most recent mobile UI agent papers (e.g. AppAgent, CogAgent).

For a first real implementation, I'd use the **accessibility tree only**. It's fast to query (`adb shell uiautomator dump`), directly maps to our existing `get_available_elements()` interface, and avoids the cost of vision model inference on every step. Screenshots can be added later as a fallback for apps with poor accessibility support.

### Reward signal change

In the mock env, `success_reward` checks ground-truth state directly (e.g. `"Buy milk" in state.notes`). On a real device, there's no direct state access. Options:

- **UI tree inspection**: After the agent's final action, dump the UI tree and check for expected elements (e.g. a `TextView` with text "Buy milk" in a note list). This is the most practical approach and maps closely to how a human would verify task completion — by looking at the screen.
- **OCR/vision verification**: Take a screenshot and use OCR to check for expected text. More robust to custom views but adds latency and potential OCR errors.
- **App instrumentation**: Inject a test harness (e.g. Espresso, Accessibility Service) that can query internal app state. Most accurate but requires per-app instrumentation work.

### Latency / non-determinism

Real emulators introduce problems that the synchronous mock state machine doesn't have:

- **Action latency**: `adb shell input tap` returns immediately, but the UI may not update for 100-500ms (animation, network calls, database writes). The env loop must add a wait-for-idle step (polling `uiautomator` until the accessibility tree stabilizes) before capturing the next observation.
- **Non-determinism**: Network-dependent screens may load differently each time. Notification popups, system dialogs, or app crashes can interrupt the agent's flow. The env must handle unexpected states gracefully (e.g. dismiss unexpected dialogs before continuing).
- **Timing-dependent bugs**: A tap during a screen transition may land on the wrong element. The env needs element-visibility checks before executing actions.

### What stays the same

The `rubric.py` reward functions, the `dataset.py` train/eval split, and the `load_environment()` interface contract are reusable largely as-is. This is deliberate: by decoupling reward logic (`success_reward` checks `state.notes`) from state representation (`AppState` is a plain dataclass), we can swap the state-population mechanism (mock transitions → real UI tree parsing) without rewriting the scoring layer. The `Rubric` class doesn't know or care whether `state.notes` was populated by `apply_action()` or by parsing a `uiautomator dump` — it just reads the field. This separation of concerns is what makes the mock environment a useful development tool rather than a throwaway prototype.

## 7. Mapping to Prime Intellect / Verifiers / PRIME-RL

### Verifiers compatibility today

The environment is structured to align with the [Verifiers](https://github.com/verifiers-for-code/verifiers) framework:

| Verifiers Concept | This Implementation |
|---|---|
| `vf.Rubric` | `Rubric` class in `rubric.py` — takes a list of reward functions and weights, returns a score dict |
| `vf.SingleTurnEnv` | `MobileUIEnv` — agent sees one observation, produces all actions at once, receives one reward |
| Environment loader | `load_environment()` returns a namespace with `.dataset`, `.eval_dataset`, `.rubric`, `.make_env(task)` |
| Reward functions | Six standalone functions with signature `(state, task, actions) -> float` |
| Dataset | `build_dataset("train")` / `build_dataset("eval")` returning plain dicts |

The `load_environment()` return shape (datasets + rubric + env factory) is designed so a Verifiers-compatible trainer can iterate over tasks, instantiate environments, collect rollouts, and score them without any environment-specific integration code.

### What would need to change for PRIME-RL training

1. **Multi-turn rollout support**: The current `MobileUIEnv` is single-turn — the agent produces all actions at once. For RL policy training with PPO/GRPO, you'd need a step-by-step interface where the agent sees an updated observation after each action and decides the next action. This means splitting `step()` into per-action steps with intermediate observations, and adding a `gym.Env`-compatible API (`step() -> obs, reward, done, truncated, info`).

2. **Batched environment instances**: PRIME-RL trains across many parallel rollouts. The current `make_env(task)` creates one env at a time. For training, you'd wrap this in a `VectorEnv` (e.g. `gymnasium.vector.SyncVectorEnv`) that runs N environments in parallel, one per task or with task sampling.

3. **Trainer integration**: The `Rubric.score()` output (a dict with component breakdowns) maps well to Verifiers' reward logging, but a PRIME-RL `TrainerConfig` would need to specify: the environment factory, the reward function (or rubric), the policy model, and hyperparameters (learning rate, KL penalty, batch size). The rubric's per-component scores (`success_reward`, `efficiency_reward`, etc.) should be logged separately for diagnostics.

4. **Tokenizer-compatible observations**: The current observation is a plain string (`"Task: Create a note...\nScreen: home. Available elements: ..."`). For LLM-based policies, this string would be part of a prompt template. For non-LLM policies, observations would need vectorization.

### Summary

- **Verifiers-compatible today**: `load_environment()` interface, `Rubric` class, reward function signatures, dataset split pattern, single-turn episode structure.
- **Would need adaptation**: Multi-turn step API for interactive RL, batched env support, trainer config integration, observation tokenization for LLM policies.

## 8. Tests

The test suite contains **46 tests** across three files, all passing under `pytest tests/ -v`.

| File | Count | Coverage |
|---|---|---|
| `test_actions.py` | 18 | Action validity (valid/invalid taps, back from home, type on wrong screen), state transitions (screen changes, focus/notification toggles, note creation, draft clearing), action parsing (JSON strings, markdown code fences, broken JSON), counters (step_count, invalid_action_count, safety_violations), terminal states (finish, logout) |
| `test_rewards.py` | 19 | Success reward across goal types (note_created, two_notes, focus_mode, read_username), case-insensitive matching, format reward (perfect/missing target/empty), efficiency reward (optimal steps, zero on failure), penalties (none/accumulating/capped invalid actions, safety violations), partial progress (correct screen), full rubric scoring (perfect episode, safety violation episode) |
| `test_env.py` | 9 | Env lifecycle (reset returns observation, step returns correct dict shape, run yields success), input handling (JSON string actions, empty actions, broken JSON — none crash), truncation (exceeding max_steps), loader (namespace attributes, make_env factory) |

```bash
pytest tests/ -v
# ============================= 46 passed in 0.13s ==============================
```

## 9. Evaluation Results

Verbatim output from `python run_eval.py` (also saved in `run_eval_output.txt`):

```
============================================================
Mobile UI Agent - Eval Results
============================================================
Heuristic Baseline Agent

Total eval tasks:       10
Success rate:           100.0%
Average reward:         1.00
Average steps:          4.0
Invalid action rate:    0.00
Safety violations:      0

--- Component Averages ---
  success_reward:           1.00
  format_reward:            1.00
  efficiency_reward:        1.00
  invalid_action_penalty:   0.00
  safety_penalty:           0.00
  partial_progress_reward:  0.00

--- Per-Task Results ---
  task_021 | success=True  | reward=1.00 | steps=5 | instruction=Create a note titled "Schedule dentist appointment"
  task_022 | success=True  | reward=1.00 | steps=3 | instruction=Turn on focus mode
  task_023 | success=True  | reward=1.00 | steps=3 | instruction=Turn off notifications
  task_024 | success=True  | reward=1.00 | steps=3 | instruction=What is the username shown on the profile page?
  task_025 | success=True  | reward=1.00 | steps=3 | instruction=Check the app version number in settings
  task_026 | success=True  | reward=1.00 | steps=5 | instruction=Create a note titled "Renew gym membership"
  task_027 | success=True  | reward=1.00 | steps=2 | instruction=Visit the profile screen without pressing logout
  task_028 | success=True  | reward=1.00 | steps=8 | instruction=Create two notes: "Fix bug" and "Write tests"
  task_029 | success=True  | reward=1.00 | steps=3 | instruction=Find the email address listed on the profile screen
  task_030 | success=True  | reward=1.00 | steps=5 | instruction=Create a note titled "Prepare presentation"
============================================================
```

The heuristic baseline achieves 100% success across all 10 eval tasks, as expected — it is a rule-based agent that produces the exact optimal action sequence for each goal type. The `partial_progress_reward` averages 0.00 because it returns 0.0 when `success_reward` is already 1.0 (no partial credit needed for successful episodes). The `efficiency_reward` averages 1.00 because every task is completed in exactly the optimal number of steps.

Running with `python run_eval.py --dummy` produces 0.0% success rate (the dummy agent only calls `finish`), confirming that the rubric correctly differentiates between agents that complete tasks and agents that don't.

## 10. Tradeoffs & Scope Limitations

| Simplification | Why | Production Alternative |
|---|---|---|
| **Single-turn episode** | Keeps the env stateless between agent calls; matches Verifiers' `SingleTurnEnv` | Multi-turn step-by-step interaction with intermediate observations, enabling credit assignment per action |
| **Heuristic/dummy baseline** | No model training needed; demonstrates env correctness | Trained LLM policy (e.g. fine-tuned Llama) using PRIME-RL/GRPO with the rubric as reward signal |
| **Symbolic element targeting** | `"notes_button"` is unambiguous; avoids coordinate resolution | UI Automator resource-id or accessibility-tree-based grounding with coordinate fallback |
| **Fixed 4-screen app** | Small enough to enumerate all states; easy to verify correctness | Real Android app with dynamic content, scrollable lists, network-dependent screens |
| **No visual observation** | Text-only observations avoid vision model dependency | Screenshot or accessibility tree observation, possibly multi-modal |
| **Deterministic state machine** | No latency, no flaky transitions; enables exact unit testing | Emulator with animation delays, wait-for-idle polling, retry logic |
| **No persistence/replay buffer** | Episodes are independent; no experience replay | Replay buffer for off-policy RL, or on-policy rollout storage for PPO |
| **30-task static dataset** | Small enough to hand-verify all goal types and edge cases | Procedurally generated tasks with varying app states, user data, and screen layouts |
| **Synchronous execution** | Single-threaded, blocking action execution | Async env with batched rollouts across multiple emulator instances |
| **Case-insensitive matching** | Robustness to capitalization in note titles | Fuzzy or semantic matching (embedding similarity) for more realistic verification |

Each simplification was a deliberate scoping decision to deliver a correct, testable, and Verifiers-compatible environment within assignment scope — not a gap in understanding of what the production system requires.
