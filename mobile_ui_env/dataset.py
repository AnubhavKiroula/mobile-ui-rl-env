"""Dataset module containing RL environment task specifications.

This module defines 30 tasks (20 for training, 10 for evaluation) with instructions,
split definitions, goal targets, step counts, and human-readable hints.
"""

from typing import Any, Dict, List, Optional

TASKS: List[Dict[str, Any]] = [
    # --- TRAIN TASKS (task_001 to task_020) ---
    {
        "task_id": "task_001",
        "instruction": 'Create a note titled "Buy milk"',
        "split": "train",
        "goal": {"type": "note_created", "title": "Buy milk"},
        "max_steps": 8,
        "optimal_steps": 5,
        "solution_hint": [
            "tap notes_button",
            "tap add_note_button",
            "type Buy milk",
            "tap save_note_button",
            "finish",
        ],
    },
    {
        "task_id": "task_002",
        "instruction": 'Create a note titled "Call doctor"',
        "split": "train",
        "goal": {"type": "note_created", "title": "Call doctor"},
        "max_steps": 8,
        "optimal_steps": 5,
        "solution_hint": [
            "tap notes_button",
            "tap add_note_button",
            "type Call doctor",
            "tap save_note_button",
            "finish",
        ],
    },
    {
        "task_id": "task_003",
        "instruction": 'Create a note titled "Team meeting at 3pm"',
        "split": "train",
        "goal": {"type": "note_created", "title": "Team meeting at 3pm"},
        "max_steps": 8,
        "optimal_steps": 5,
        "solution_hint": [
            "tap notes_button",
            "tap add_note_button",
            "type Team meeting at 3pm",
            "tap save_note_button",
            "finish",
        ],
    },
    {
        "task_id": "task_004",
        "instruction": "Enable focus mode",
        "split": "train",
        "goal": {"type": "focus_mode_on"},
        "max_steps": 6,
        "optimal_steps": 3,
        "solution_hint": [
            "tap settings_button",
            "tap focus_mode_toggle",
            "finish",
        ],
    },
    {
        "task_id": "task_005",
        "instruction": "Disable notifications",
        "split": "train",
        "goal": {"type": "notifications_off"},
        "max_steps": 6,
        "optimal_steps": 3,
        "solution_hint": [
            "tap settings_button",
            "tap notifications_toggle",
            "finish",
        ],
    },
    {
        "task_id": "task_006",
        "instruction": "Enable notifications",
        "split": "train",
        "goal": {"type": "notifications_on"},
        "max_steps": 8,
        "optimal_steps": 5,
        "solution_hint": [
            "tap settings_button",
            "tap notifications_toggle",
            "tap notifications_toggle",
            "finish",
        ],
    },
    {
        "task_id": "task_007",
        "instruction": "Find the username from the profile screen",
        "split": "train",
        "goal": {"type": "read_username", "value": "john_doe"},
        "max_steps": 6,
        "optimal_steps": 3,
        "solution_hint": ["tap profile_button", "tap username_label", "finish"],
    },
    {
        "task_id": "task_008",
        "instruction": "Find the email address from the profile screen",
        "split": "train",
        "goal": {"type": "read_email", "value": "john@example.com"},
        "max_steps": 6,
        "optimal_steps": 3,
        "solution_hint": ["tap profile_button", "tap email_label", "finish"],
    },
    {
        "task_id": "task_009",
        "instruction": "Open settings and check the app version",
        "split": "train",
        "goal": {"type": "read_version", "value": "1.0.0"},
        "max_steps": 6,
        "optimal_steps": 3,
        "solution_hint": ["tap settings_button", "tap version_label", "finish"],
    },
    {
        "task_id": "task_010",
        "instruction": "Go to the profile screen and do NOT logout",
        "split": "train",
        "goal": {"type": "navigate_profile_no_logout"},
        "max_steps": 6,
        "optimal_steps": 3,
        "solution_hint": ["tap profile_button", "finish"],
    },
    {
        "task_id": "task_011",
        "instruction": 'Create a note titled "Buy groceries"',
        "split": "train",
        "goal": {"type": "note_created", "title": "Buy groceries"},
        "max_steps": 8,
        "optimal_steps": 5,
        "solution_hint": [
            "tap notes_button",
            "tap add_note_button",
            "type Buy groceries",
            "tap save_note_button",
            "finish",
        ],
    },
    {
        "task_id": "task_012",
        "instruction": 'Create a note titled "Finish report"',
        "split": "train",
        "goal": {"type": "note_created", "title": "Finish report"},
        "max_steps": 8,
        "optimal_steps": 5,
        "solution_hint": [
            "tap notes_button",
            "tap add_note_button",
            "type Finish report",
            "tap save_note_button",
            "finish",
        ],
    },
    {
        "task_id": "task_013",
        "instruction": "Disable focus mode",
        "split": "train",
        "goal": {"type": "focus_mode_off"},
        "max_steps": 10,
        "optimal_steps": 6,
        "solution_hint": [
            "tap settings_button",
            "tap focus_mode_toggle",
            "tap focus_mode_toggle",
            "finish",
        ],
    },
    {
        "task_id": "task_014",
        "instruction": 'Create two notes: "Read book" and "Exercise"',
        "split": "train",
        "goal": {"type": "two_notes_created", "titles": ["Read book", "Exercise"]},
        "max_steps": 14,
        "optimal_steps": 10,
        "solution_hint": [
            "tap notes_button",
            "tap add_note_button",
            "type Read book",
            "tap save_note_button",
            "tap add_note_button",
            "type Exercise",
            "tap save_note_button",
            "finish",
        ],
    },
    {
        "task_id": "task_015",
        "instruction": "Navigate to the notes screen",
        "split": "train",
        "goal": {"type": "navigate_notes"},
        "max_steps": 4,
        "optimal_steps": 2,
        "solution_hint": ["tap notes_button", "finish"],
    },
    {
        "task_id": "task_016",
        "instruction": "Navigate to the settings screen",
        "split": "train",
        "goal": {"type": "navigate_settings"},
        "max_steps": 4,
        "optimal_steps": 2,
        "solution_hint": ["tap settings_button", "finish"],
    },
    {
        "task_id": "task_017",
        "instruction": 'Create a note titled "Water plants"',
        "split": "train",
        "goal": {"type": "note_created", "title": "Water plants"},
        "max_steps": 8,
        "optimal_steps": 5,
        "solution_hint": [
            "tap notes_button",
            "tap add_note_button",
            "type Water plants",
            "tap save_note_button",
            "finish",
        ],
    },
    {
        "task_id": "task_018",
        "instruction": 'Create a note titled "Pay electricity bill"',
        "split": "train",
        "goal": {"type": "note_created", "title": "Pay electricity bill"},
        "max_steps": 8,
        "optimal_steps": 5,
        "solution_hint": [
            "tap notes_button",
            "tap add_note_button",
            "type Pay electricity bill",
            "tap save_note_button",
            "finish",
        ],
    },
    {
        "task_id": "task_019",
        "instruction": 'Create two notes: "Morning run" and "Evening walk"',
        "split": "train",
        "goal": {
            "type": "two_notes_created",
            "titles": ["Morning run", "Evening walk"],
        },
        "max_steps": 14,
        "optimal_steps": 10,
        "solution_hint": [
            "tap notes_button",
            "tap add_note_button",
            "type Morning run",
            "tap save_note_button",
            "tap add_note_button",
            "type Evening walk",
            "tap save_note_button",
            "finish",
        ],
    },
    {
        "task_id": "task_020",
        "instruction": "Find both the username and email from the profile screen",
        "split": "train",
        "goal": {"type": "read_email", "value": "john@example.com"},
        "max_steps": 8,
        "optimal_steps": 4,
        "solution_hint": [
            "tap profile_button",
            "tap username_label",
            "tap email_label",
            "finish",
        ],
    },
    # --- EVAL TASKS (task_021 to task_030) ---
    {
        "task_id": "task_021",
        "instruction": 'Create a note titled "Schedule dentist appointment"',
        "split": "eval",
        "goal": {
            "type": "note_created",
            "title": "Schedule dentist appointment",
        },
        "max_steps": 8,
        "optimal_steps": 5,
        "solution_hint": [
            "tap notes_button",
            "tap add_note_button",
            "type Schedule dentist appointment",
            "tap save_note_button",
            "finish",
        ],
    },
    {
        "task_id": "task_022",
        "instruction": "Turn on focus mode",
        "split": "eval",
        "goal": {"type": "focus_mode_on"},
        "max_steps": 6,
        "optimal_steps": 3,
        "solution_hint": [
            "tap settings_button",
            "tap focus_mode_toggle",
            "finish",
        ],
    },
    {
        "task_id": "task_023",
        "instruction": "Turn off notifications",
        "split": "eval",
        "goal": {"type": "notifications_off"},
        "max_steps": 6,
        "optimal_steps": 3,
        "solution_hint": [
            "tap settings_button",
            "tap notifications_toggle",
            "finish",
        ],
    },
    {
        "task_id": "task_024",
        "instruction": "What is the username shown on the profile page?",
        "split": "eval",
        "goal": {"type": "read_username", "value": "john_doe"},
        "max_steps": 6,
        "optimal_steps": 3,
        "solution_hint": ["tap profile_button", "tap username_label", "finish"],
    },
    {
        "task_id": "task_025",
        "instruction": "Check the app version number in settings",
        "split": "eval",
        "goal": {"type": "read_version", "value": "1.0.0"},
        "max_steps": 6,
        "optimal_steps": 3,
        "solution_hint": ["tap settings_button", "tap version_label", "finish"],
    },
    {
        "task_id": "task_026",
        "instruction": 'Create a note titled "Renew gym membership"',
        "split": "eval",
        "goal": {"type": "note_created", "title": "Renew gym membership"},
        "max_steps": 8,
        "optimal_steps": 5,
        "solution_hint": [
            "tap notes_button",
            "tap add_note_button",
            "type Renew gym membership",
            "tap save_note_button",
            "finish",
        ],
    },
    {
        "task_id": "task_027",
        "instruction": "Visit the profile screen without pressing logout",
        "split": "eval",
        "goal": {"type": "navigate_profile_no_logout"},
        "max_steps": 6,
        "optimal_steps": 3,
        "solution_hint": ["tap profile_button", "finish"],
    },
    {
        "task_id": "task_028",
        "instruction": 'Create two notes: "Fix bug" and "Write tests"',
        "split": "eval",
        "goal": {"type": "two_notes_created", "titles": ["Fix bug", "Write tests"]},
        "max_steps": 14,
        "optimal_steps": 10,
        "solution_hint": [
            "tap notes_button",
            "tap add_note_button",
            "type Fix bug",
            "tap save_note_button",
            "tap add_note_button",
            "type Write tests",
            "tap save_note_button",
            "finish",
        ],
    },
    {
        "task_id": "task_029",
        "instruction": "Find the email address listed on the profile screen",
        "split": "eval",
        "goal": {"type": "read_email", "value": "john@example.com"},
        "max_steps": 6,
        "optimal_steps": 3,
        "solution_hint": ["tap profile_button", "tap email_label", "finish"],
    },
    {
        "task_id": "task_030",
        "instruction": 'Create a note titled "Prepare presentation"',
        "split": "eval",
        "goal": {"type": "note_created", "title": "Prepare presentation"},
        "max_steps": 8,
        "optimal_steps": 5,
        "solution_hint": [
            "tap notes_button",
            "tap add_note_button",
            "type Prepare presentation",
            "tap save_note_button",
            "finish",
        ],
    },
]


def build_dataset(split: str = "train") -> List[Dict[str, Any]]:
    """Builds a list of tasks for the specified split.

    Args:
        split: The dataset split to retrieve, either 'train' or 'eval'.

    Returns:
        List[Dict[str, Any]]: List of task dictionaries matching the split.

    Raises:
        ValueError: If split is not 'train' or 'eval'.
    """
    if split not in ("train", "eval"):
        raise ValueError(
            f"Unknown split: '{split}'. Must be either 'train' or 'eval'."
        )
    return [task for task in TASKS if task["split"] == split]


def get_task_by_id(task_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves a task by its task ID.

    Args:
        task_id: The ID of the task to retrieve (e.g., 'task_001').

    Returns:
        Optional[Dict[str, Any]]: The task dictionary if found, otherwise None.
    """
    for task in TASKS:
        if task["task_id"] == task_id:
            return task
    return None


def get_all_tasks() -> List[Dict[str, Any]]:
    """Retrieves all tasks in the dataset.

    Returns:
        List[Dict[str, Any]]: A list of all 30 task dictionaries.
    """
    return TASKS


def get_task_stats() -> Dict[str, Any]:
    """Computes summary statistics of the task dataset.

    Returns:
        Dict[str, Any]: Statistics including totals, splits, and goal type distributions.
    """
    total = len(TASKS)
    train_count = sum(1 for t in TASKS if t["split"] == "train")
    eval_count = sum(1 for t in TASKS if t["split"] == "eval")

    goal_dist: Dict[str, int] = {}
    for t in TASKS:
        goal_type = t["goal"]["type"]
        goal_dist[goal_type] = goal_dist.get(goal_type, 0) + 1

    return {
        "total": total,
        "train": train_count,
        "eval": eval_count,
        "goal_type_distribution": goal_dist,
    }
