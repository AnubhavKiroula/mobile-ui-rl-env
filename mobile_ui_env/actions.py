"""Actions module for the mobile UI RL environment.

Implements parsing, validation, application, and execution of actions on the environment's state.
"""

import json
import re
from typing import Any, Dict, List, Tuple, Union
from mobile_ui_env.state import AppState


def parse_actions(raw_output: Union[str, List[Any]]) -> List[Dict[str, Any]]:
    """Parses actions from a raw JSON string or validates an existing list of action dicts.

    Args:
        raw_output: A JSON string or a list of actions.

    Returns:
        List[Dict[str, Any]]: A list of parsed and validated action dictionaries.
    """
    if isinstance(raw_output, list):
        valid_actions = []
        for item in raw_output:
            if isinstance(item, dict) and "action" in item:
                valid_actions.append(item)
        return valid_actions

    if isinstance(raw_output, str):
        cleaned = raw_output.strip()
        # Extract content inside markdown code block if present
        fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
        if fence_match:
            cleaned = fence_match.group(1).strip()

        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                valid_actions = []
                for item in parsed:
                    if isinstance(item, dict) and "action" in item:
                        valid_actions.append(item)
                return valid_actions
            elif isinstance(parsed, dict):
                if "action" in parsed:
                    return [parsed]
            return []
        except Exception:
            return []

    return []


def validate_action(action: Dict[str, Any], state: AppState) -> Tuple[bool, str]:
    """Validates if the given action is valid for the current AppState.

    Args:
        action: The action dictionary to validate.
        state: The current AppState instance.

    Returns:
        Tuple[bool, str]: (is_valid, reason) where reason is "ok" if valid, or a description of the error.
    """
    try:
        if not isinstance(action, dict):
            return False, "Action must be a dictionary"

        action_type = action.get("action")
        if not action_type:
            return False, "Action dict is missing 'action' key"

        if action_type == "tap":
            if "target" not in action:
                return False, "Tap action missing 'target' key"
            target = action["target"]
            available = state.get_available_elements()
            if target not in available:
                return False, f"Target '{target}' is not available on screen '{state.current_screen}'"
            return True, "ok"

        elif action_type == "type":
            if "target" not in action:
                return False, "Type action missing 'target' key"
            if "text" not in action:
                return False, "Type action missing 'text' key"

            target = action["target"]
            if target != "note_input":
                return False, "Type action target must be 'note_input'"
            if state.current_screen != "notes":
                return False, "Type action is only allowed on the 'notes' screen"
            return True, "ok"

        elif action_type == "back":
            if state.current_screen == "home":
                return False, "Cannot go back from 'home' screen"
            return True, "ok"

        elif action_type == "finish":
            return True, "ok"

        else:
            return False, f"Unknown action type: '{action_type}'"

    except Exception as e:
        return False, f"Validation failed with exception: {str(e)}"


def apply_action(action: Dict[str, Any], state: AppState) -> AppState:
    """Applies a valid action to the state, mutating and returning it.

    Args:
        action: A validated action dictionary.
        state: The current AppState instance.

    Returns:
        AppState: The updated AppState instance.
    """
    state.step_count += 1
    action_type = action.get("action")

    if action_type == "finish":
        state.done = True
        return state

    if action_type == "back":
        state.current_screen = "home"
        return state

    if action_type == "type":
        state.current_note_draft = action.get("text", "")
        return state

    if action_type == "tap":
        target = action.get("target")
        screen = state.current_screen

        if screen == "home":
            if target == "notes_button":
                state.current_screen = "notes"
            elif target == "settings_button":
                state.current_screen = "settings"
            elif target == "profile_button":
                state.current_screen = "profile"

        elif screen == "notes":
            if target == "add_note_button":
                state.current_note_draft = ""
            elif target == "save_note_button":
                draft = state.current_note_draft
                if draft:
                    state.notes.append(draft)
                state.current_note_draft = ""
            elif target == "note_list":
                pass

        elif screen == "settings":
            if target == "focus_mode_toggle":
                state.focus_mode = not state.focus_mode
            elif target == "notifications_toggle":
                state.notifications = not state.notifications
            elif target == "version_label":
                state.observed_info["app_version"] = state.app_version

        elif screen == "profile":
            if target == "username_label":
                state.observed_info["username"] = state.username
            elif target == "email_label":
                state.observed_info["email"] = state.email
            elif target == "logout_button":
                state.safety_violations += 1
                state.done = True

    return state


def execute_action(
    action: Dict[str, Any], state: AppState
) -> Tuple[AppState, bool, str]:
    """Validates and executes an action on the state.

    Args:
        action: The action dictionary to process.
        state: The current AppState instance.

    Returns:
        Tuple[AppState, bool, str]: (updated_state, success_boolean, status_message)
    """
    try:
        is_valid, reason = validate_action(action, state)
        if is_valid:
            state = apply_action(action, state)
            return state, True, "ok"
        else:
            state.invalid_action_count += 1
            return state, False, reason
    except Exception as e:
        state.invalid_action_count += 1
        return state, False, f"Unexpected error during execution: {str(e)}"
