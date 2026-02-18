#!/usr/bin/env python3
"""Add a new agent to the nanobot config and restart gateway."""

import json
import subprocess
import sys
import os
import signal
from pathlib import Path


def stop_gateway():
    """Stop any running nanobot gateway processes."""
    try:
        # Find and kill nanobot gateway processes
        result = subprocess.run(
            ["pkill", "-f", "nanobot gateway"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("🛑 Stopped existing gateway...")
            import time
            time.sleep(1)  # Wait for graceful shutdown
    except Exception as e:
        print(f"Warning: Could not stop gateway: {e}")


def start_gateway():
    """Start the nanobot gateway in the background."""
    # Get the directory where this script is located
    script_dir = Path(__file__).parent.parent
    venv_activate = script_dir / ".venv" / "bin" / "activate"

    if not venv_activate.exists():
        print("Error: Virtual environment not found at", venv_activate)
        print("Please run: python -m venv .venv && source .venv/bin/activate && pip install -e .")
        return False

    print("🚀 Starting gateway...")

    # Start gateway in background using nohup
    log_file = script_dir / "gateway.log"
    subprocess.Popen(
        f"source {venv_activate} && nohup nanobot gateway > {log_file} 2>&1 &",
        shell=True,
        executable="/bin/bash",
        cwd=script_dir
    )

    print(f"📋 Logs: {log_file}")
    return True


def main():
    if len(sys.argv) < 3:
        print("Usage: python add-agent.py <agent_name> <telegram_token>")
        print("Example: python add-agent.py mybot 123456789:ABCdefGHIjklMNOpqrSTUvwxYZ")
        sys.exit(1)

    agent_name = sys.argv[1]
    telegram_token = sys.argv[2]
    config_path = Path.home() / ".nanobot" / "config.json"
    workspace = Path.home() / ".nanobot" / f"workspace_{agent_name}"

    # Load config
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        config = json.load(f)

    # Check if agent already exists
    if agent_name in config.get("agents", {}).get("pool", {}):
        print(f"Error: Agent '{agent_name}' already exists in config")
        sys.exit(1)

    # Ensure pool exists
    if "agents" not in config:
        config["agents"] = {}
    if "pool" not in config["agents"]:
        config["agents"]["pool"] = {}

    # Add new agent
    config["agents"]["pool"][agent_name] = {
        "enabled": True,
        "name": agent_name.replace("-", " ").title(),
        "workspace": f"~/.nanobot/workspace_{agent_name}",
        "telegram": {
            "enabled": True,
            "token": telegram_token,
            "allowFrom": []
        }
    }

    # Save config
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # Create workspace directories
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "memory").mkdir(exist_ok=True)
    (workspace / "sessions").mkdir(exist_ok=True)
    (workspace / "skills").mkdir(exist_ok=True)

    # Create default memory files
    memory_file = workspace / "memory" / "MEMORY.md"
    if not memory_file.exists():
        memory_file.write_text("# Long-term Memory\n\n")

    history_file = workspace / "memory" / "HISTORY.md"
    if not history_file.exists():
        history_file.write_text("")

    print(f"✅ Agent '{agent_name}' added successfully!")
    print(f"   Workspace: {workspace}")

    # Restart gateway
    stop_gateway()
    start_gateway()

    print(f"\n✨ Done! Agent '{agent_name}' is now active.")


if __name__ == "__main__":
    main()
