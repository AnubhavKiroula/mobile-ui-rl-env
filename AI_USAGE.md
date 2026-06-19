# AI_USAGE.md

## 1. What I asked AI tools

I used two AI tools for different parts of the workflow:

- **Claude** — for research, task diagnosis, and prompt design. Before writing any
  code, I used Claude to think through the approach: how to structure state, actions,
  and rewards for an RL-style environment. I'd previously built an RL-based data
  cleaning platform, so I brought that prior experience into the discussion rather
  than starting from zero. I used Claude to turn that approach into precise,
  deliverable-driven prompts (explicit system/user prompt structure) for each phase,
  rather than vague one-line requests.
- **Antigravity** — for actually writing the code, phase by phase, using the prompts
  built with Claude.

I split the work into 6 phases instead of one large prompt:

1. Project structure + core state/actions (with placeholder files first)
2. Dataset (30 tasks, train/eval split) + early eval file groundwork
3. Rubric + reward logic
4. Eval script + environment loader
5. Unit tests (tests/test_actions.py, tests/test_rewards.py, tests/test_env.py)
6. README + this file

I kept the first 1-2 phases intentionally lighter (structure/placeholders) so that
Antigravity had grounded context by the time it reached the heavier reward and
environment-logic phases (3-4), rather than generating those cold.

All of this was done on Antigravity's free tier. I had already written a global
`rules.md` for the IDE covering conventions I reuse across projects, which kept
token usage efficient enough to handle the heavier phases without hitting limits.

## 2. What code I accepted from AI tools

I accepted the AI-generated implementation for:

- The `AppState` model and screen/element definitions
- Action parsing and execution logic (`parse_actions`, `validate_action`,
  `apply_action`, `execute_action`)
- The dataset builder and the 30 task definitions
- The reward functions and rubric aggregation (`success_reward`, `format_reward`,
  `efficiency_reward`, `invalid_action_penalty`, `safety_penalty`,
  `partial_progress_reward`, `build_rubric`)
- `MobileUIEnv`, `load_environment`, and `run_eval.py`
- The full test suite across all three test files

Each phase's output was checked manually by actually running it (`pytest`,
`python run_eval.py`, manual calls into the env) rather than just reading the
diff and assuming it was correct.

## 3. What I modified myself

The main issue I caught: the first version of the action-handling code did not
have proper error handling for invalid input. When I manually tested edge cases
(bad/nonexistent targets, malformed actions) instead of just the happy path, the
environment crashed instead of returning `valid=False` gracefully. I traced this
by deliberately running invalid-input commands myself, identified that this broke
the "must not crash on invalid actions" requirement from the assignment, and
re-prompted Antigravity specifically to fix error handling for worst-case /
malformed input, then re-ran the same edge cases to confirm it no longer crashed
and degraded safely instead.

Outside of that, I didn't need to rewrite logic myself, but I treated "the AI
says it's done" as a starting point, not a finish line, and verified actual
behavior against the assignment's requirements (especially section 9's required
tests and the "invalid actions should not crash the environment" rule in
section 3) before moving to the next phase.

## 4. What I learned

- **Sparse vs. dense rewards**: this distinction wasn't new to me in the abstract —
  I'd run into it before in my earlier RL-based data cleaning project — but
  working through `rubric.py` here made the practical difference more concrete:
  `success_reward` and `safety_penalty` only give meaningful signal at/near the
  end of an episode, while `efficiency_reward`, `partial_progress_reward`, and
  `invalid_action_penalty` give the agent something to learn from even on
  episodes that ultimately fail. Seeing that tradeoff implemented directly,
  rather than just reading about it, made it click more solidly. That earlier
  RL project also helped me place top 800 out of 52,000+ developers in a
  national-level hackathon final organized by MetaX/PyTorch, so this assignment
  built on ground I'd already started developing.
- **Don't trust "it ran once" as "it's correct"**: the invalid-action crash only
  showed up because I deliberately tested bad input instead of just the
  documented happy path. That reinforced that verifying AI-generated code means
  actively trying to break it, not just confirming the intended case works.
- **Prompt precision compounds across phases**: being explicit about deliverables,
  required tests, and constraints in every phase prompt (rather than one vague
  ask) meant Antigravity's output needed very little correction overall — the
  one real bug I found was specifically because I tested an edge case the
  original prompt hadn't explicitly called out.
