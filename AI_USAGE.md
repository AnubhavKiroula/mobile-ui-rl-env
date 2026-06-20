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

**Reward weight mismatch in `rubric.py`.** This was the most significant issue I
caught, and it took several rounds of pushing back before it was actually resolved
correctly. After Phase 5 was complete, I ran a full read-only audit of the codebase
against the assignment spec. The audit flagged that `build_rubric()`'s `weights`
array didn't match the coefficients used in the hardcoded `total` formula inside
`Rubric.score()` — specifically, the weights for `invalid_action_penalty` (should
be 0.2) and `safety_penalty` (should be 0.3) were swapped.

I didn't accept this at face value either way. I asked Antigravity to confirm the
bug independently before touching any code, by printing the actual weights array
and the actual formula coefficients side by side. It confirmed the mismatch was
real, fixed the weights array, and reported "WEIGHT FIX VERIFIED" with a test case.

I checked that verification output myself by hand-summing the reported
`weighted_components` values. They summed to 0.45, but the reported `total` was
0.0 — a discrepancy the agent's own "verified" claim had missed. I pushed back and
asked it to show me the literal source of the `total` calculation rather than
explain it in prose. It turned out `total` isn't a sum of `weighted_components` at
all — it's a separate subtractive formula (`success + 0.1*format + 0.2*efficiency +
0.1*partial - 0.2*invalid_pen - 0.3*safety_pen`, then clipped to `[0, 1]`), which is
a valid design but meant the two values were never supposed to match in the first
place. I verified this formula by hand-computing the expected total for the test
scenario myself (`-0.19`, clipping to `0.0`) and got the same result independently.

While checking this, I also asked whether the existing test
(`test_rubric_score_safety_violation`, which asserted `total < 0.5`) would actually
have caught the original weight-swap bug. It wouldn't have — under several broken
weight/sign scenarios I worked through, `total` still lands under 0.5, so the bound
was too loose to be real evidence the rubric was correct. I had the assertion
tightened to the exact expected value (`total == 0.0`) instead, with the expected
value hand-computed and independently verified rather than just asserted, and
confirmed it doesn't introduce floating-point flakiness for this case before
accepting it.

**Crash on invalid input.** Separately, in an earlier phase, I manually tested edge
cases (bad/nonexistent targets, malformed actions) instead of just the happy path,
and found the environment crashed instead of returning `valid=False` gracefully.
I traced this by deliberately running invalid-input commands myself, identified
that this broke the "must not crash on invalid actions" requirement from the
assignment, and re-prompted Antigravity specifically to fix error handling for
worst-case input, then re-ran the same edge cases to confirm it degraded safely
instead of crashing.

In general, I treated "the AI says it's verified" as a claim to check, not a fact
to accept — every "done" or "verified" message in this project was followed by me
either re-running the relevant command myself or independently redoing the
underlying arithmetic before moving on.

**Docker build.** The agent that wrote the Dockerfile didn't have a running Docker
desktop in its environment, so it could not build or run the container itself and
said so explicitly rather than claiming success. I built and ran the container
myself afterward — `docker build` succeeded, the image shows up in Docker Desktop,
and the test/eval commands inside the container ran successfully.

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
- **An agent saying "verified" is not the same as something being verified**: the
  reward-weight bug taught me this directly. The agent printed "WEIGHT FIX
  VERIFIED" with output that, on inspection, didn't actually add up — the reported
  total didn't match the sum of the reported weighted components. The fix itself
  turned out to be correct, but the verification step needed a second layer of
  scrutiny (hand-computing the expected value myself) before I could trust it. I
  now treat any agent claim of "passed," "verified," or "confirmed" as a starting
  point for my own check, not as the check itself.
- **Loose test assertions can hide real bugs**: `total < 0.5` felt like a
  reasonable safety-violation test until I worked through what values would
  satisfy it. A correctly-implemented rubric, a sign-flipped rubric, and a rubric
  that silently ignored the safety penalty entirely would all have passed that
  assertion. Tightening it to an exact, hand-verified value (`total == 0.0`) was a
  small change, but it's the difference between a test that documents intent and
  one that would actually catch a regression.
- **Prompt precision compounds across phases**: being explicit about deliverables,
  required tests, and constraints in every phase prompt (rather than one vague
  ask) meant Antigravity's output needed very little correction overall — most of
  the issues I found were specifically because I tested edge cases or re-derived
  numbers the original prompts hadn't explicitly forced a check on.