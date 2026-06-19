"""Unit tests for the reward and penalty functions, and the Rubric class in mobile_ui_env.

Tests success verification, formatting checks, efficiency metrics, penalties, and overall rubric scores.
"""

import pytest
from mobile_ui_env.state import AppState
from mobile_ui_env.actions import execute_action
from mobile_ui_env.rubric import (
    success_reward,
    format_reward,
    efficiency_reward,
    invalid_action_penalty,
    safety_penalty,
    partial_progress_reward,
    build_rubric,
)
from mobile_ui_env.dataset import get_task_by_id


def test_success_reward_note_created() -> None:
    """success_reward returns 1.0 when correct note exists."""
    task = {
        "goal": {"type": "note_created", "title": "Buy milk"},
        "max_steps": 8,
        "optimal_steps": 5,
    }
    s = AppState()
    s.notes = ["Buy milk"]
    assert success_reward(s, task, []) == 1.0


def test_success_reward_note_case_insensitive() -> None:
    """success_reward must match note title case-insensitively."""
    task = {
        "goal": {"type": "note_created", "title": "Buy Milk"},
        "max_steps": 8,
        "optimal_steps": 5,
    }
    s = AppState()
    s.notes = ["buy milk"]
    assert success_reward(s, task, []) == 1.0


def test_success_reward_note_not_created() -> None:
    """success_reward returns 0.0 when note is missing."""
    task = {
        "goal": {"type": "note_created", "title": "Buy milk"},
        "max_steps": 8,
        "optimal_steps": 5,
    }
    s = AppState()
    assert success_reward(s, task, []) == 0.0


def test_success_reward_two_notes() -> None:
    """success_reward returns 1.0 only when both notes exist."""
    task = {
        "goal": {"type": "two_notes_created", "titles": ["Note A", "Note B"]},
        "max_steps": 14,
        "optimal_steps": 10,
    }
    s = AppState()
    s.notes = ["Note A"]
    assert success_reward(s, task, []) == 0.0
    s.notes = ["Note A", "Note B"]
    assert success_reward(s, task, []) == 1.0


def test_success_reward_focus_mode() -> None:
    """success_reward checks focus_mode state correctly."""
    task = {
        "goal": {"type": "focus_mode_on"},
        "max_steps": 6,
        "optimal_steps": 3,
    }
    s = AppState()
    assert success_reward(s, task, []) == 0.0
    s.focus_mode = True
    assert success_reward(s, task, []) == 1.0


def test_success_reward_read_username() -> None:
    """success_reward checks observed_info for username."""
    task = {
        "goal": {"type": "read_username", "value": "john_doe"},
        "max_steps": 6,
        "optimal_steps": 3,
    }
    s = AppState()
    assert success_reward(s, task, []) == 0.0
    s.observed_info["username"] = "john_doe"
    assert success_reward(s, task, []) == 1.0


def test_format_reward_perfect_actions() -> None:
    """format_reward returns 1.0 for well-formed actions."""
    actions = [{"action": "tap", "target": "notes_button"}, {"action": "finish"}]
    task = {
        "goal": {"type": "navigate_notes"},
        "max_steps": 4,
        "optimal_steps": 2,
    }
    s = AppState()
    assert format_reward(s, task, actions) == 1.0


def test_format_reward_missing_target() -> None:
    """format_reward penalizes tap action missing target key."""
    actions = [{"action": "tap"}]
    task = {
        "goal": {"type": "navigate_notes"},
        "max_steps": 4,
        "optimal_steps": 2,
    }
    s = AppState()
    score = format_reward(s, task, actions)
    assert score < 1.0


def test_format_reward_empty_actions() -> None:
    """format_reward returns 0.0 for empty action list."""
    task = {
        "goal": {"type": "navigate_notes"},
        "max_steps": 4,
        "optimal_steps": 2,
    }
    s = AppState()
    assert format_reward(s, task, []) == 0.0


def test_efficiency_reward_optimal() -> None:
    """efficiency_reward returns 1.0 when steps == optimal."""
    task = {
        "goal": {"type": "note_created", "title": "Buy milk"},
        "max_steps": 8,
        "optimal_steps": 5,
    }
    s = AppState()
    s.notes = ["Buy milk"]
    s.step_count = 5
    assert efficiency_reward(s, task, []) == 1.0


def test_efficiency_reward_zero_on_failure() -> None:
    """efficiency_reward returns 0.0 when task not completed."""
    task = {
        "goal": {"type": "note_created", "title": "Buy milk"},
        "max_steps": 8,
        "optimal_steps": 5,
    }
    s = AppState()
    s.step_count = 5
    assert efficiency_reward(s, task, []) == 0.0


def test_invalid_action_penalty_none() -> None:
    """invalid_action_penalty returns 0.0 with no invalid actions."""
    task = {
        "goal": {"type": "navigate_notes"},
        "max_steps": 4,
        "optimal_steps": 2,
    }
    s = AppState()
    assert invalid_action_penalty(s, task, []) == 0.0


def test_invalid_action_penalty_accumulates() -> None:
    """invalid_action_penalty grows with invalid action count."""
    task = {
        "goal": {"type": "navigate_notes"},
        "max_steps": 4,
        "optimal_steps": 2,
    }
    s = AppState()
    s.invalid_action_count = 3
    penalty = invalid_action_penalty(s, task, [])
    assert penalty == pytest.approx(0.3)


def test_invalid_action_penalty_caps_at_one() -> None:
    """invalid_action_penalty must not exceed 1.0."""
    task = {
        "goal": {"type": "navigate_notes"},
        "max_steps": 4,
        "optimal_steps": 2,
    }
    s = AppState()
    s.invalid_action_count = 20
    assert invalid_action_penalty(s, task, []) == 1.0


def test_safety_penalty_no_violation() -> None:
    """safety_penalty returns 0.0 with no violations."""
    task = {
        "goal": {"type": "navigate_notes"},
        "max_steps": 4,
        "optimal_steps": 2,
    }
    s = AppState()
    assert safety_penalty(s, task, []) == 0.0


def test_safety_penalty_with_violation() -> None:
    """safety_penalty returns 1.0 when violations > 0."""
    task = {
        "goal": {"type": "navigate_notes"},
        "max_steps": 4,
        "optimal_steps": 2,
    }
    s = AppState()
    s.safety_violations = 1
    assert safety_penalty(s, task, []) == 1.0


def test_partial_progress_reward_correct_screen() -> None:
    """partial_progress_reward returns >= 0.3 when agent reached the correct screen or navigated away but did not complete the goal."""
    task = {
        "goal": {"type": "note_created", "title": "Buy milk"},
        "max_steps": 8,
        "optimal_steps": 5,
    }
    s = AppState()
    actions = [{"action": "tap", "target": "notes_button"}]
    score = partial_progress_reward(s, task, actions)
    assert score >= 0.3


def test_rubric_score_full_episode() -> None:
    """Full rubric score for perfect episode must be >= 0.8."""
    task = get_task_by_id("task_001")
    assert task is not None
    s = AppState()
    actions = [
        {"action": "tap", "target": "notes_button"},
        {"action": "tap", "target": "add_note_button"},
        {"action": "type", "target": "note_input", "text": "Buy milk"},
        {"action": "tap", "target": "save_note_button"},
        {"action": "finish"},
    ]
    for a in actions:
        execute_action(a, s)
    rubric = build_rubric()
    result = rubric.score(s, task, actions)
    assert result["total"] >= 0.8
    assert result["success"] is True


def test_rubric_score_safety_violation() -> None:
    """Rubric total must be heavily penalized for logout."""
    task = get_task_by_id("task_010")
    assert task is not None
    s = AppState()
    actions = [
        {"action": "tap", "target": "profile_button"},
        {"action": "tap", "target": "logout_button"},
    ]
    for a in actions:
        execute_action(a, s)
    rubric = build_rubric()
    result = rubric.score(s, task, actions)
    assert result["safety_violation"] is True
    assert result["total"] == 0.0
