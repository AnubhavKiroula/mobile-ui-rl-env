"""Unit tests for the environment and wrapper loader of mobile_ui_env.

Tests reset, step, run, action parsing, max step truncation, and loader namespaces.
"""

import json
import pytest
from mobile_ui_env.env import MobileUIEnv, load_environment
from mobile_ui_env.dataset import get_task_by_id
from mobile_ui_env.rubric import build_rubric


def test_env_reset_returns_observation() -> None:
    """reset() must return a string containing the instruction."""
    task = get_task_by_id("task_001")
    assert task is not None
    rubric = build_rubric()
    env = MobileUIEnv(task=task, rubric=rubric)
    obs = env.reset()
    assert isinstance(obs, str)
    assert "Buy milk" in obs


def test_env_step_returns_dict() -> None:
    """step() must return a dict with required keys."""
    task = get_task_by_id("task_001")
    assert task is not None
    rubric = build_rubric()
    env = MobileUIEnv(task=task, rubric=rubric)
    env.reset()
    actions = [
        {"action": "tap", "target": "notes_button"},
        {"action": "tap", "target": "add_note_button"},
        {"action": "type", "target": "note_input", "text": "Buy milk"},
        {"action": "tap", "target": "save_note_button"},
        {"action": "finish"},
    ]
    result = env.step(actions)
    required_keys = [
        "task_id",
        "instruction",
        "actions_parsed",
        "actions_executed",
        "state",
        "score",
        "truncated",
    ]
    for key in required_keys:
        assert key in result, f"Missing key: {key}"


def test_env_run_success() -> None:
    """run() on correct actions must yield success=True."""
    task = get_task_by_id("task_001")
    assert task is not None
    rubric = build_rubric()
    env = MobileUIEnv(task=task, rubric=rubric)
    actions = [
        {"action": "tap", "target": "notes_button"},
        {"action": "tap", "target": "add_note_button"},
        {"action": "type", "target": "note_input", "text": "Buy milk"},
        {"action": "tap", "target": "save_note_button"},
        {"action": "finish"},
    ]
    result = env.run(actions)
    assert result["score"]["success"] is True


def test_env_run_with_string_actions() -> None:
    """run() must accept raw JSON string as input."""
    task = get_task_by_id("task_001")
    assert task is not None
    rubric = build_rubric()
    env = MobileUIEnv(task=task, rubric=rubric)
    actions_str = json.dumps(
        [
            {"action": "tap", "target": "notes_button"},
            {"action": "tap", "target": "add_note_button"},
            {"action": "type", "target": "note_input", "text": "Buy milk"},
            {"action": "tap", "target": "save_note_button"},
            {"action": "finish"},
        ]
    )
    result = env.run(actions_str)
    assert result["score"]["success"] is True


def test_env_handles_empty_actions() -> None:
    """run() with empty action list must not crash and must return success=False."""
    task = get_task_by_id("task_001")
    assert task is not None
    rubric = build_rubric()
    env = MobileUIEnv(task=task, rubric=rubric)
    try:
        result = env.run([])
        assert result["score"]["success"] is False
    except Exception as e:
        pytest.fail(f"env.run raised an exception: {e}")


def test_env_handles_invalid_json_string() -> None:
    """run() with completely broken JSON string must not crash."""
    task = get_task_by_id("task_001")
    assert task is not None
    rubric = build_rubric()
    env = MobileUIEnv(task=task, rubric=rubric)
    try:
        result = env.run("not json at all {{{")
        assert isinstance(result, dict)
    except Exception as e:
        pytest.fail(f"env.run raised an exception: {e}")


def test_env_max_steps_truncation() -> None:
    """Episode must stop at max_steps even if finish was never called."""
    task = get_task_by_id("task_001")
    assert task is not None
    rubric = build_rubric()
    env = MobileUIEnv(task=task, rubric=rubric)
    many_actions = [
        {"action": "tap", "target": "notes_button"},
        {"action": "back"},
    ] * 20
    result = env.run(many_actions)
    assert result["state"]["step_count"] <= task["max_steps"]
    assert result["truncated"] is True


def test_load_environment_structure() -> None:
    """load_environment() must return object with correct attrs."""
    env_config = load_environment()
    assert hasattr(env_config, "dataset")
    assert hasattr(env_config, "eval_dataset")
    assert hasattr(env_config, "rubric")
    assert hasattr(env_config, "make_env")
    assert len(env_config.dataset) == 22
    assert len(env_config.eval_dataset) == 12


def test_load_environment_make_env() -> None:
    """make_env() must return a working MobileUIEnv instance."""
    env_config = load_environment()
    task = env_config.eval_dataset[0]
    env = env_config.make_env(task)
    assert isinstance(env, MobileUIEnv)
    obs = env.reset()
    assert isinstance(obs, str)
