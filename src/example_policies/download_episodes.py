"""Download MCAP episodes from GCS using platform API filters.

Usage:
  python -m example_policies.download_episodes <output_dir> "<filter_str>" [--env=nonprod|prod]

Example:
  python -m example_policies.download_episodes ./episodes \
    "task=pick_red_brick&rating=excellent&date_from=2026-02-19&date_to=2026-02-19&hide_ignored=true"
"""

import os
import subprocess
import sys

from example_policies.platform_api_client import fetch_episode_paths


def main():
    env_name = "prod"
    positional = []

    for arg in sys.argv[1:]:
        if arg.startswith("--env="):
            env_name = arg.split("=", 1)[1]
        else:
            positional.append(arg)

    if len(positional) != 2:
        print("Usage: python -m example_policies.download_episodes <output_dir> <filter_str> [--env=prod|nonprod]",
              file=sys.stderr)
        sys.exit(1)

    output_dir, filter_str = positional
    os.makedirs(output_dir, exist_ok=True)

    print(f"Fetching episode list from API ({env_name})...", file=sys.stderr)
    episodes = fetch_episode_paths(filter_str, env=env_name)
    total = len(episodes)
    print(f"{total} episodes to download", file=sys.stderr)

    for i, ep in enumerate(episodes, 1):
        gs_path = ep["object_storage_path"]
        filename = gs_path.rsplit("/", 1)[-1]
        dest = os.path.join(output_dir, filename)

        if os.path.exists(dest):
            print(f"[{i}/{total}] skip (exists): {filename}", file=sys.stderr)
            continue

        print(f"[{i}/{total}] downloading: {filename}", file=sys.stderr)
        subprocess.run(["gcloud", "storage", "cp", gs_path, dest], check=True)

    print(f"Done. {total} episodes in {output_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
