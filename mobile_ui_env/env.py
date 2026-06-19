"""Environment module for the mobile UI RL agent simulation.

Defines the MobileUIEnv class and the load_environment function.
"""

from types import SimpleNamespace
from typing import Any, Dict, List, Union
from mobile_ui_env.actions import parse_actions, execute_action
from mobile_ui_env.dataset import build_dataset
from mobile_ui_env.rubric import build_rubric, Rubric
from mobile_ui_env.state import AppState


class MobileUIEnv:
    """Single-turn RL environment for mobile UI agent tasks.

    The agent receives an instruction (observation), produces
    a list of JSON actions, and receives a reward score.
    This is a single-turn environment: the agent sees the
    task instruction once and produces ALL actions at once
    (not step by step interactively).
    """

    def __init__(self, task: Dict[str, Any], rubric: Rubric):
        """Initializes the environment with a specific task and scoring rubric.

        Args:
            task: The task specification dictionary.
            rubric: The Rubric scoring instance.
        """
        self.task = task
        self.rubric = rubric
        self.state = AppState()
        self.actions_taken: List[Dict[str, Any]] = []

    def reset(self) -> str:
        """Resets the environment state and returns the initial observation.

        Returns:
            str: The initial task and screen description observation.
        """
        self.state.reset()
        self.actions_taken = []
        screen_obs = self.state.get_observation()
        return f"Task: {self.task.get('instruction', '')}\n{screen_obs}"

    def step(self, raw_actions: Union[str, List[Any]]) -> Dict[str, Any]:
        """Runs a full episode using the provided sequence of actions.

        Processes the actions sequentially, stops early if state.done is True
        or the task's max_steps limit is exceeded, and scores the final state.

        Args:
            raw_actions: Either a JSON string containing actions or a list of actions.

        Returns:
            Dict[str, Any]: The execution result dictionary.
        """
        actions_parsed = parse_actions(raw_actions)
        actions_executed = 0
        truncated = False

        for action in actions_parsed:
            if self.state.done:
                break
            if self.state.step_count >= self.task.get("max_steps", 8):
                truncated = True
                break

            self.state, success, reason = execute_action(action, self.state)
            actions_executed += 1

        # Check if truncated at the end of the action loop
        if (
            self.state.step_count >= self.task.get("max_steps", 8)
            and not self.state.done
        ):
            truncated = True

        score_res = self.rubric.score(self.state, self.task, actions_parsed)

        return {
            "task_id": self.task.get("task_id", ""),
            "instruction": self.task.get("instruction", ""),
            "actions_parsed": actions_parsed,
            "actions_executed": actions_executed,
            "state": {
                "current_screen": self.state.current_screen,
                "notes": list(self.state.notes),
                "focus_mode": self.state.focus_mode,
                "notifications": self.state.notifications,
                "observed_info": dict(self.state.observed_info),
                "step_count": self.state.step_count,
                "invalid_action_count": self.state.invalid_action_count,
                "safety_violations": self.state.safety_violations,
                "done": self.state.done,
            },
            "score": score_res,
            "truncated": truncated,
        }

    def run(self, raw_actions: Union[str, List[Any]]) -> Dict[str, Any]:
        """Convenience method that resets the environment and runs the episode.

        Args:
            raw_actions: A sequence of actions.

        Returns:
            Dict[str, Any]: The step result dictionary.
        """
        self.reset()
        return self.step(raw_actions)


def load_environment() -> SimpleNamespace:
    """Builds and returns a configured environment namespace.

    Returns:
        SimpleNamespace: Containing the dataset, eval_dataset, rubric, and make_env.
    """
    dataset = build_dataset("train")
    eval_dataset = build_dataset("eval")
    rubric = build_rubric()

    def make_env(task: Dict[str, Any]) -> MobileUIEnv:
        return MobileUIEnv(task=task, rubric=rubric)

    return SimpleNamespace(
        dataset=dataset,
        eval_dataset=eval_dataset,
        rubric=rubric,
        make_env=make_env,
    )
