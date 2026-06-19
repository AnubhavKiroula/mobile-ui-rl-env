"""Unit tests for the state and action handling modules of mobile_ui_env.

Tests navigation, validation, parsing, safety penalties, toggles, and state modifications.
"""

import pytest
from mobile_ui_env.state import AppState
from mobile_ui_env.actions import (
    parse_actions,
    execute_action,
)
from mobile_ui_env.dataset import get_task_by_id
from mobile_ui_env.rubric import success_reward, safety_penalty


def test_valid_tap_changes_screen() -> None:
    """Valid tap on notes_button from home screen must change current_screen to 'notes'."""
    s = AppState()
    action = {"action": "tap", "target": "notes_button"}
    s, valid, _ = execute_action(action, s)
    assert valid is True
    assert s.current_screen == "notes"


def test_invalid_tap_does_not_crash() -> None:
    """Tapping a non-existent element must not raise any exception and must return valid=False."""
    s = AppState()
    action = {"action": "tap", "target": "nonexistent_button"}
    try:
        result = execute_action(action, s)
        assert result[1] is False
    except Exception as e:
        pytest.fail(f"execute_action raised an exception: {e}")


def test_creating_note_updates_state() -> None:
    """Full note creation flow must result in the note appearing in state.notes."""
    s = AppState()
    execute_action({"action": "tap", "target": "notes_button"}, s)
    execute_action({"action": "tap", "target": "add_note_button"}, s)
    execute_action(
        {"action": "type", "target": "note_input", "text": "Buy milk"}, s
    )
    execute_action({"action": "tap", "target": "save_note_button"}, s)
    assert "Buy milk" in s.notes


def test_correct_task_gets_success_reward() -> None:
    """Completing the Buy milk task correctly must yield success_reward == 1.0."""
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
    assert success_reward(s, task, actions) == 1.0


def test_logout_triggers_safety_penalty() -> None:
    """Tapping logout_button must increment safety_violations and safety_penalty must return 1.0."""
    task = get_task_by_id("task_010")
    assert task is not None
    s = AppState()
    actions = [
        {"action": "tap", "target": "profile_button"},
        {"action": "tap", "target": "logout_button"},
    ]
    for a in actions:
        execute_action(a, s)
    assert s.safety_violations > 0
    assert safety_penalty(s, task, actions) == 1.0


def test_back_from_home_is_invalid() -> None:
    """Back action from home screen must be invalid."""
    s = AppState()
    action = {"action": "back"}
    _, valid, _ = execute_action(action, s)
    assert valid is False


def test_back_from_notes_goes_home() -> None:
    """Back action from notes screen must return to home."""
    s = AppState()
    execute_action({"action": "tap", "target": "notes_button"}, s)
    assert s.current_screen == "notes"
    execute_action({"action": "back"}, s)
    assert s.current_screen == "home"


def test_type_on_wrong_screen_is_invalid() -> None:
    """Type action on settings screen must be invalid."""
    s = AppState()
    execute_action({"action": "tap", "target": "settings_button"}, s)
    action = {"action": "type", "target": "note_input", "text": "hello"}
    _, valid, _ = execute_action(action, s)
    assert valid is False


def test_focus_mode_toggles() -> None:
    """focus_mode_toggle must flip focus_mode boolean."""
    s = AppState()
    assert s.focus_mode is False
    execute_action({"action": "tap", "target": "settings_button"}, s)
    execute_action({"action": "tap", "target": "focus_mode_toggle"}, s)
    assert s.focus_mode is True
    execute_action({"action": "tap", "target": "focus_mode_toggle"}, s)
    assert s.focus_mode is False


def test_notifications_toggle() -> None:
    """notifications_toggle must flip notifications boolean."""
    s = AppState()
    assert s.notifications is True
    execute_action({"action": "tap", "target": "settings_button"}, s)
    execute_action({"action": "tap", "target": "notifications_toggle"}, s)
    assert s.notifications is False


def test_parse_actions_from_string() -> None:
    """parse_actions must handle JSON string input."""
    raw = '[{"action": "finish"}]'
    result = parse_actions(raw)
    assert isinstance(result, list)
    assert result[0]["action"] == "finish"


def test_parse_actions_from_markdown() -> None:
    """parse_actions must strip markdown code fences."""
    raw = '```json\n[{"action": "finish"}]\n```'
    result = parse_actions(raw)
    assert len(result) == 1
    assert result[0]["action"] == "finish"


def test_parse_actions_invalid_json_returns_empty() -> None:
    """parse_actions must return [] for completely invalid input."""
    result = parse_actions("this is not json at all {{{{")
    assert result == []


def test_invalid_action_count_increments() -> None:
    """invalid_action_count must increment for each bad action."""
    s = AppState()
    assert s.invalid_action_count == 0
    execute_action({"action": "tap", "target": "nonexistent"}, s)
    execute_action({"action": "tap", "target": "also_nonexistent"}, s)
    assert s.invalid_action_count == 2


def test_finish_sets_done() -> None:
    """finish action must set state.done to True."""
    s = AppState()
    execute_action({"action": "finish"}, s)
    assert s.done is True


def test_save_note_empty_draft_does_not_add() -> None:
    """Saving with empty draft must not add empty string to notes."""
    s = AppState()
    execute_action({"action": "tap", "target": "notes_button"}, s)
    execute_action({"action": "tap", "target": "save_note_button"}, s)
    assert len(s.notes) == 0


def test_observed_info_username() -> None:
    """Tapping username_label must store username in observed_info."""
    s = AppState()
    execute_action({"action": "tap", "target": "profile_button"}, s)
    execute_action({"action": "tap", "target": "username_label"}, s)
    assert "username" in s.observed_info
    assert s.observed_info["username"] == "john_doe"


def test_step_count_increments() -> None:
    """step_count must increment for every executed action."""
    s = AppState()
    assert s.step_count == 0
    execute_action({"action": "tap", "target": "notes_button"}, s)
    execute_action({"action": "back"}, s)
    assert s.step_count == 2
