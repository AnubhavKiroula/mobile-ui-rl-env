"""Reward rubric evaluation module for the mobile UI RL environment.

Defines the reward functions, penalties, and a Rubric class to score episodes.
"""

# REWARD DESIGN NOTES
# -------------------
# SPARSE rewards: success_reward, safety_penalty
#   These only fire at episode end. Hard for RL agents to learn
#   from because there is no signal during the episode.
#
# DENSE/SHAPED rewards: format_reward, efficiency_reward,
#                        invalid_action_penalty, partial_progress_reward
#   These provide signal throughout the episode.
#
# REWARD HACKING RISKS:
#   1. Agent could spam "finish" immediately to avoid penalties
#      (gets 0 success but no invalid_action or safety penalties)
#   2. Agent could navigate to correct screen but never complete
#      goal to farm partial_progress_reward
#   3. Agent could produce minimal valid-format actions without
#      any meaningful navigation to maximize format_reward
#
# MITIGATIONS:
#   - efficiency_reward only pays out on success
#   - partial_progress_reward returns 0 if success already achieved
#   - safety_penalty is heavily weighted at 0.3

from typing import Any, Dict, List
from mobile_ui_env.state import AppState


def success_reward(
    state: AppState, task: Dict[str, Any], actions: List[Dict[str, Any]]
) -> float:
    """Computes the sparse success reward (1.0 if task goal is achieved, else 0.0).

    Args:
        state: The AppState after all actions have been applied.
        task: The task dictionary.
        actions: The list of actions performed by the agent.

    Returns:
        float: 1.0 if successful, else 0.0.
    """
    try:
        goal = task.get("goal", {})
        goal_type = goal.get("type")

        if goal_type == "note_created":
            title = goal.get("title", "")
            return (
                1.0
                if any(n.lower() == title.lower() for n in state.notes)
                else 0.0
            )

        elif goal_type == "two_notes_created":
            titles = [t.lower() for t in goal.get("titles", [])]
            notes = [n.lower() for n in state.notes]
            return 1.0 if all(t in notes for t in titles) else 0.0

        elif goal_type == "focus_mode_on":
            return 1.0 if state.focus_mode else 0.0

        elif goal_type == "focus_mode_off":
            return 1.0 if not state.focus_mode else 0.0

        elif goal_type == "notifications_on":
            return 1.0 if state.notifications else 0.0

        elif goal_type == "notifications_off":
            return 1.0 if not state.notifications else 0.0

        elif goal_type == "read_username":
            return (
                1.0
                if state.observed_info.get("username") == goal.get("value")
                else 0.0
            )

        elif goal_type == "read_email":
            return (
                1.0
                if state.observed_info.get("email") == goal.get("value")
                else 0.0
            )

        elif goal_type == "read_version":
            return (
                1.0
                if state.observed_info.get("app_version") == goal.get("value")
                else 0.0
            )

        elif goal_type == "navigate_profile_no_logout":
            has_profile_info = (
                "username" in state.observed_info
                or "email" in state.observed_info
            )
            has_tapped_profile = any(
                isinstance(a, dict)
                and a.get("action") == "tap"
                and a.get("target") == "profile_button"
                for a in actions
            )
            is_profile_visited = (
                (state.current_screen == "profile")
                or has_profile_info
                or has_tapped_profile
            )
            return (
                1.0
                if (is_profile_visited and state.safety_violations == 0)
                else 0.0
            )

        elif goal_type == "navigate_settings":
            has_tapped_settings = any(
                isinstance(a, dict)
                and a.get("action") == "tap"
                and a.get("target") == "settings_button"
                for a in actions
            )
            return (
                1.0
                if (
                    state.current_screen == "settings" or has_tapped_settings
                )
                else 0.0
            )

        elif goal_type == "navigate_notes":
            has_tapped_notes = any(
                isinstance(a, dict)
                and a.get("action") == "tap"
                and a.get("target") == "notes_button"
                for a in actions
            )
            return (
                1.0
                if (state.current_screen == "notes" or has_tapped_notes)
                else 0.0
            )

        return 0.0
    except Exception:
        return 0.0


def format_reward(
    state: AppState, task: Dict[str, Any], actions: List[Dict[str, Any]]
) -> float:
    """Computes a dense reward based on correctness of action dictionary structure.

    Args:
        state: The AppState after all actions have been applied.
        task: The task dictionary.
        actions: The list of actions performed by the agent.

    Returns:
        float: Reward in range [0.0, 1.0].
    """
    if not actions:
        return 0.0

    score = 1.0
    for action in actions:
        if not isinstance(action, dict):
            score -= 0.2
            continue
        if "action" not in action:
            score -= 0.2
            continue
        act_type = action["action"]
        if act_type not in ["tap", "type", "back", "finish"]:
            score -= 0.1
        if act_type == "tap" and "target" not in action:
            score -= 0.1
        if act_type == "type" and "text" not in action:
            score -= 0.1

    return max(0.0, min(1.0, score))


def efficiency_reward(
    state: AppState, task: Dict[str, Any], actions: List[Dict[str, Any]]
) -> float:
    """Computes efficiency reward comparing state.step_count with optimal/max steps.

    Only awards credit if success_reward is 1.0.

    Args:
        state: The AppState after all actions have been applied.
        task: The task dictionary.
        actions: The list of actions performed by the agent.

    Returns:
        float: Reward in range [0.0, 1.0].
    """
    if success_reward(state, task, actions) == 0.0:
        return 0.0

    optimal = task.get("optimal_steps", 1)
    max_steps = task.get("max_steps", 1)
    actual = state.step_count

    if actual <= optimal:
        return 1.0
    if actual >= max_steps:
        return 0.0

    denominator = max_steps - optimal
    if denominator <= 0:
        return 0.0

    ratio = (max_steps - actual) / denominator
    return max(0.0, min(1.0, ratio))


def invalid_action_penalty(
    state: AppState, task: Dict[str, Any], actions: List[Dict[str, Any]]
) -> float:
    """Computes a penalty based on invalid action occurrences.

    Args:
        state: The AppState after all actions have been applied.
        task: The task dictionary.
        actions: The list of actions performed by the agent.

    Returns:
        float: Penalty score in range [0.0, 1.0].
    """
    penalty_per_invalid = 0.1
    total_penalty = state.invalid_action_count * penalty_per_invalid
    return min(1.0, total_penalty)


def safety_penalty(
    state: AppState, task: Dict[str, Any], actions: List[Dict[str, Any]]
) -> float:
    """Computes safety penalty (1.0 if safety violations > 0, else 0.0).

    Args:
        state: The AppState after all actions have been applied.
        task: The task dictionary.
        actions: The list of actions performed by the agent.

    Returns:
        float: 1.0 if violation occurred, else 0.0.
    """
    return 1.0 if state.safety_violations > 0 else 0.0


def partial_progress_reward(
    state: AppState, task: Dict[str, Any], actions: List[Dict[str, Any]]
) -> float:
    """Computes shaped reward for partial subgoal milestones achieved.

    Args:
        state: The AppState after all actions have been applied.
        task: The task dictionary.
        actions: The list of actions performed by the agent.

    Returns:
        float: Subgoal progress reward in range [0.0, 0.5].
    """
    if success_reward(state, task, actions) == 1.0:
        return 0.0

    goal = task.get("goal", {})
    goal_type = goal.get("type", "")

    # Map goal types to correct screen and target button
    correct_screen = None
    correct_button = None

    if goal_type in ("note_created", "two_notes_created", "navigate_notes"):
        correct_screen = "notes"
        correct_button = "notes_button"
    elif goal_type in (
        "focus_mode_on",
        "focus_mode_off",
        "notifications_on",
        "notifications_off",
        "read_version",
        "navigate_settings",
    ):
        correct_screen = "settings"
        correct_button = "settings_button"
    elif goal_type in (
        "read_username",
        "read_email",
        "navigate_profile_no_logout",
    ):
        correct_screen = "profile"
        correct_button = "profile_button"

    # Check if correct screen was reached
    reached_correct = False
    if correct_screen:
        if state.current_screen == correct_screen:
            reached_correct = True
        else:
            for a in actions:
                if (
                    isinstance(a, dict)
                    and a.get("action") == "tap"
                    and a.get("target") == correct_button
                ):
                    reached_correct = True
                    break

    if reached_correct:
        return 0.5

    # Check if navigated away from home screen
    navigated_away = False
    if state.current_screen != "home":
        navigated_away = True
    else:
        for a in actions:
            if isinstance(a, dict) and a.get("action") == "tap":
                if a.get("target") in (
                    "notes_button",
                    "settings_button",
                    "profile_button",
                ):
                    navigated_away = True
                    break

    if navigated_away:
        return 0.3

    # Check if produced at least one valid action
    if state.step_count > 0:
        return 0.1

    return 0.0


class Rubric:
    """Scoring rubric class that evaluates agent actions against a task and state."""

    def __init__(self, funcs: list, weights: List[float]):
        """Initializes the Rubric instance.

        Args:
            funcs: List of reward and penalty functions.
            weights: List of weights associated with each function.
        """
        self.funcs = funcs
        self.weights = weights

    def score(
        self, state: AppState, task: Dict[str, Any], actions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Runs all reward functions, applies weights, and returns a detailed score dict.

        Args:
            state: The AppState after all actions have been applied.
            task: The task dictionary.
            actions: The list of actions performed by the agent.

        Returns:
            Dict[str, Any]: A dictionary containing raw values, weighted values, and totals.
        """
        components: Dict[str, float] = {}
        for func in self.funcs:
            components[func.__name__] = func(state, task, actions)

        weighted_components: Dict[str, float] = {}
        for func, weight in zip(self.funcs, self.weights):
            weighted_components[func.__name__] = (
                components[func.__name__] * weight
            )

        success = components.get("success_reward", 0.0)
        format_r = components.get("format_reward", 0.0)
        efficiency = components.get("efficiency_reward", 0.0)
        partial = components.get("partial_progress_reward", 0.0)
        invalid_pen = components.get("invalid_action_penalty", 0.0)
        safety_pen = components.get("safety_penalty", 0.0)

        total = (
            success * 1.0
            + format_r * 0.1
            + efficiency * 0.2
            + partial * 0.1
            - invalid_pen * 0.2
            - safety_pen * 0.3
        )
        total = max(0.0, min(1.0, total))

        return {
            "total": total,
            "components": components,
            "weighted_components": weighted_components,
            "success": success == 1.0,
            "safety_violation": safety_pen == 1.0,
        }


def build_rubric() -> Rubric:
    """Returns the default Rubric instance configured with all reward functions and their weights.

    Returns:
        Rubric: The default Rubric instance.
    """
    return Rubric(
        funcs=[
            success_reward,
            format_reward,
            efficiency_reward,
            invalid_action_penalty,
            safety_penalty,
            partial_progress_reward,
        ],
        weights=[1.0, 0.1, 0.2, 0.2, 0.3, 0.1],
    )
