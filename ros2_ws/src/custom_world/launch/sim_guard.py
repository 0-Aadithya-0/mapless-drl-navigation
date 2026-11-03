"""
Helpers shared by the custom_world launch files.

The guard refuses to start a Gazebo server while another one is still
running. Both servers would host a world named "default" and publish
/scan and /odom on the same Gazebo topics, so the bridge would merge two
robots' data into one stream without any error.
"""

import os
import time

# Ctrl-C reaches Gazebo, but its shutdown takes ~13-15 s, longer than
# launch waits before returning the prompt. A server still running after
# this long is a leftover, not one that is shutting down.
SHUTDOWN_GRACE = 20.0


def prepend_path(name: str, path: str) -> None:
    """Put path first in the search-path environment variable name."""
    existing = os.environ.get(name, "")
    os.environ[name] = path + os.pathsep + existing if existing else path


def gazebo_servers() -> list[tuple[int, str]]:
    """Return (pid, command) for each running `gz sim` that is not GUI-only."""
    servers = []

    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue

        try:
            with open(f"/proc/{entry}/cmdline", "rb") as f:
                cmdline = f.read()
        except OSError:
            # The process exited during the scan.
            continue

        tokens = cmdline.replace(b"\0", b" ").decode(errors="replace").split()

        # The gz tool is a Ruby script, "ruby .../gz sim ...", which renames
        # itself "gz sim ...". Shell wrappers ("sh -c ruby ...") are not
        # matched; the gz process they start is. Run with both server and
        # GUI, it forks children renamed "gz sim server" and "gz sim gui".
        if tokens and os.path.basename(tokens[0]) == "ruby":
            tokens = tokens[1:]

        gui_only = "-g" in tokens or tokens[2:3] == ["gui"]

        if (
            len(tokens) >= 2
            and os.path.basename(tokens[0]) == "gz"
            and tokens[1] == "sim"
            and not gui_only
        ):
            servers.append((int(entry), " ".join(tokens)))

    return servers


def refuse_if_gazebo_running() -> None:
    """Wait out a server that is shutting down, then fail if one remains."""
    deadline = time.monotonic() + SHUTDOWN_GRACE
    announced = False

    while servers := gazebo_servers():
        pids = " ".join(str(pid) for pid, _ in servers)

        if time.monotonic() >= deadline:
            listing = "\n".join(
                f"  pid {pid}: {command}" for pid, command in servers
            )

            raise RuntimeError(
                "Another Gazebo server is still running:\n"
                f"{listing}\n"
                f"Stop it before launching: kill {pids}"
            )

        if not announced:
            print(
                f"Waiting up to {SHUTDOWN_GRACE:.0f} s for a previous "
                f"Gazebo server (pid {pids}) to exit...",
                flush=True,
            )
            announced = True

        time.sleep(0.5)
