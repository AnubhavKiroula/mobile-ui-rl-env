"""State tracking module for the mobile UI RL environment."""

from dataclasses import dataclass, field
from typing import Any, Dict, List

# Screen constants
SCREEN_HOME: str = "home"
SCREEN_NOTES: str = "notes"
SCREEN_SETTINGS: str = "settings"
SCREEN_PROFILE: str = "profile"

# Set of valid screen names
SCREENS: set[str] = {SCREEN_HOME, SCREEN_NOTES, SCREEN_SETTINGS, SCREEN_PROFILE}

# Map of screens to their interactive elements
SCREEN_ELEMENTS: Dict[str, List[str]] = {
    SCREEN_HOME: ["notes_button", "settings_button", "profile_button"],
    SCREEN_NOTES: ["add_note_button", "note_input", "save_note_button", "note_list"],
    SCREEN_SETTINGS: ["focus_mode_toggle", "notifications_toggle", "version_label"],
    SCREEN_PROFILE: ["username_label", "email_label", "logout_button"],
}


@dataclass
class AppState:
    """Tracks the state of the mobile UI environment, including navigation and UI fields."""

    current_screen: str = "home"
    notes: List[str] = field(default_factory=list)
    focus_mode: bool = False
    notifications: bool = True
    username: str = "john_doe"
    email: str = "john@example.com"
    app_version: str = "1.0.0"
    step_count: int = 0
    invalid_action_count: int = 0
    safety_violations: int = 0
    current_note_draft: str = ""
    done: bool = False
    observed_info: Dict[str, Any] = field(default_factory=dict)

    def reset(self) -> "AppState":
        """Resets all state fields to their default values and returns self.

        Returns:
            AppState: The reset AppState instance.
        """
        self.current_screen = "home"
        self.notes = []
        self.focus_mode = False
        self.notifications = True
        self.username = "john_doe"
        self.email = "john@example.com"
        self.app_version = "1.0.0"
        self.step_count = 0
        self.invalid_action_count = 0
        self.safety_violations = 0
        self.current_note_draft = ""
        self.done = False
        self.observed_info = {}
        return self

    def get_available_elements(self) -> List[str]:
        """Returns a list of valid tap/type target elements for the current screen.

        Returns:
            List[str]: The valid elements on the current screen.
        """
        return SCREEN_ELEMENTS.get(self.current_screen, [])

    def get_observation(self) -> str:
        """Returns a human-readable text description of the current screen.

        Returns:
            str: A formatted screen description including available targets and screen-specific info.
        """
        available_elements = self.get_available_elements()
        elements_str = ", ".join(available_elements)
        obs = f"Screen: {self.current_screen}. Available elements: {elements_str}."

        if self.current_screen == SCREEN_NOTES:
            obs += f" Notes saved: {len(self.notes)}"
        elif self.current_screen == SCREEN_SETTINGS:
            focus_val = "ON" if self.focus_mode else "OFF"
            notif_val = "ON" if self.notifications else "OFF"
            obs += f" focus_mode: {focus_val}, notifications: {notif_val}, version: {self.app_version}"
        elif self.current_screen == SCREEN_PROFILE:
            obs += f" Username: {self.username}, Email: {self.email}"

        return obs
