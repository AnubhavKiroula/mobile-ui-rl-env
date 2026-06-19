"""Evaluation runner script for the mobile UI RL environment.

Runs a heuristic baseline agent or a dummy agent on the evaluation tasks
and prints summarized and per-task results.
"""

import argparse
import pprint
from typing import Any, Dict, List
from mobile_ui_env import load_environment


def heuristic_agent(task: Dict[str, Any]) -> List[Dict[str, Any]]:
    """A rule-based agent that produces correct action sequences for each goal type.

    Args:
        task: The task specification dictionary.

    Returns:
        List[Dict[str, Any]]: The sequence of action dictionaries.
    """
    goal = task.get("goal", {})
    goal_type = goal.get("type")

    if goal_type == "note_created":
        return [
            {"action": "tap", "target": "notes_button"},
            {"action": "tap", "target": "add_note_button"},
            {"action": "type", "target": "note_input", "text": goal.get("title", "")},
            {"action": "tap", "target": "save_note_button"},
            {"action": "finish"},
        ]

    elif goal_type == "two_notes_created":
        titles = goal.get("titles", ["", ""])
        return [
            {"action": "tap", "target": "notes_button"},
            {"action": "tap", "target": "add_note_button"},
            {"action": "type", "target": "note_input", "text": titles[0]},
            {"action": "tap", "target": "save_note_button"},
            {"action": "tap", "target": "add_note_button"},
            {"action": "type", "target": "note_input", "text": titles[1]},
            {"action": "tap", "target": "save_note_button"},
            {"action": "finish"},
        ]

    elif goal_type == "focus_mode_on":
        return [
            {"action": "tap", "target": "settings_button"},
            {"action": "tap", "target": "focus_mode_toggle"},
            {"action": "finish"},
        ]

    elif goal_type == "focus_mode_off":
        return [
            {"action": "tap", "target": "settings_button"},
            {"action": "tap", "target": "focus_mode_toggle"},
            {"action": "tap", "target": "focus_mode_toggle"},
            {"action": "finish"},
        ]

    elif goal_type == "notifications_off":
        return [
            {"action": "tap", "target": "settings_button"},
            {"action": "tap", "target": "notifications_toggle"},
            {"action": "finish"},
        ]

    elif goal_type == "notifications_on":
        return [
            {"action": "tap", "target": "settings_button"},
            {"action": "tap", "target": "notifications_toggle"},
            {"action": "tap", "target": "notifications_toggle"},
            {"action": "finish"},
        ]

    elif goal_type == "read_username":
        return [
            {"action": "tap", "target": "profile_button"},
            {"action": "tap", "target": "username_label"},
            {"action": "finish"},
        ]

    elif goal_type == "read_email":
        return [
            {"action": "tap", "target": "profile_button"},
            {"action": "tap", "target": "email_label"},
            {"action": "finish"},
        ]

    elif goal_type == "read_version":
        return [
            {"action": "tap", "target": "settings_button"},
            {"action": "tap", "target": "version_label"},
            {"action": "finish"},
        ]

    elif goal_type == "navigate_profile_no_logout":
        return [
            {"action": "tap", "target": "profile_button"},
            {"action": "finish"},
        ]

    elif goal_type == "navigate_settings":
        return [
            {"action": "tap", "target": "settings_button"},
            {"action": "finish"},
        ]

    elif goal_type == "navigate_notes":
        return [
            {"action": "tap", "target": "notes_button"},
            {"action": "finish"},
        ]

    return [{"action": "finish"}]


def dummy_agent(task: Dict[str, Any]) -> List[Dict[str, Any]]:
    """A dummy agent that immediately finishes the task.

    Args:
        task: The task specification dictionary.

    Returns:
        List[Dict[str, Any]]: A list containing only a finish action.
    """
    return [{"action": "finish"}]


def main() -> None:
    """Main execution function for run_eval.py."""
    parser = argparse.ArgumentParser(
        description="Run evaluation on Mobile UI Environment."
    )
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="Use the dummy agent instead of the heuristic baseline.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Run evaluation on a single specific task ID.",
    )
    args = parser.parse_args()

    env_config = load_environment()

    agent_fn = dummy_agent if args.dummy else heuristic_agent
    agent_name = "Dummy Agent" if args.dummy else "Heuristic Baseline Agent"

    if args.task:
        task = None
        for t in env_config.eval_dataset:
            if t["task_id"] == args.task:
                task = t
                break
        if not task:
            for t in env_config.dataset:
                if t["task_id"] == args.task:
                    task = t
                    break
        if not task:
            print(f"Error: Task '{args.task}' not found.")
            return

        env = env_config.make_env(task)
        actions = agent_fn(task)
        result = env.run(actions)
        print("============================================================")
        print(f"Single Task Debug: {args.task} ({agent_name})")
        print("============================================================")
        pprint.pprint(result)
        print("============================================================")
        return

    # Run full evaluation on all 10 eval tasks
    results = []
    for task in env_config.eval_dataset:
        env = env_config.make_env(task)
        actions = agent_fn(task)
        result = env.run(actions)
        results.append(result)

    total_tasks = len(results)
    successes = sum(1 for r in results if r["score"]["success"])
    success_rate = (
        (successes / total_tasks) * 100.0 if total_tasks > 0 else 0.0
    )

    avg_reward = (
        sum(r["score"]["total"] for r in results) / total_tasks
        if total_tasks > 0
        else 0.0
    )
    avg_steps = (
        sum(r["state"]["step_count"] for r in results) / total_tasks
        if total_tasks > 0
        else 0.0
    )

    total_invalid = sum(r["state"]["invalid_action_count"] for r in results)
    total_actions = sum(len(r["actions_parsed"]) for r in results)
    invalid_rate = (
        (total_invalid / total_actions) if total_actions > 0 else 0.0
    )

    safety_violations = sum(
        1 for r in results if r["score"]["safety_violation"]
    )

    # Component averages
    components = [
        "success_reward",
        "format_reward",
        "efficiency_reward",
        "invalid_action_penalty",
        "safety_penalty",
        "partial_progress_reward",
    ]
    comp_averages = {}
    for comp in components:
        comp_sum = sum(
            r["score"]["components"].get(comp, 0.0) for r in results
        )
        comp_averages[comp] = comp_sum / total_tasks if total_tasks > 0 else 0.0

    print("============================================================")
    print("Mobile UI Agent - Eval Results")
    print("============================================================")
    print(agent_name)
    print()
    print(f"Total eval tasks:       {total_tasks}")
    print(f"Success rate:           {success_rate:.1f}%")
    print(f"Average reward:         {avg_reward:.2f}")
    print(f"Average steps:          {avg_steps:.1f}")
    print(f"Invalid action rate:    {invalid_rate:.2f}")
    print(f"Safety violations:      {safety_violations}")
    print()
    print("--- Component Averages ---")
    for comp in components:
        print(f"  {comp}:".ljust(28) + f"{comp_averages[comp]:.2f}")
    print()
    print("--- Per-Task Results ---")
    for r in results:
        success_status = r["score"]["success"]
        reward_val = r["score"]["total"]
        steps_val = r["state"]["step_count"]
        print(
            f"  {r['task_id']} | success={success_status:<5} | "
            f"reward={reward_val:.2f} | steps={steps_val} | "
            f"instruction={r['instruction']}"
        )
    print("============================================================")


if __name__ == "__main__":
    main()
