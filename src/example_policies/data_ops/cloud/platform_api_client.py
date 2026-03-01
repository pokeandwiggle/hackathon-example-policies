"""Thin client for the platform API training endpoint.

Stdlib-only (no extra dependencies). Requires gcloud CLI for auth.
"""

import json
import subprocess
import urllib.parse
import urllib.request

ENVS = {
    "nonprod": {"project": "nonprod1-svc-vvps", "region": "europe-west3"},
    "prod": {"project": "prod1-svc-vvps", "region": "europe-west3"},
}


def _get_api_url(project_id: str, region: str) -> str:
    result = subprocess.run(
        ["gcloud", "run", "services", "describe", "platform-api",
         "--project", project_id, "--region", region, "--format", "value(status.url)"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def _get_auth_token() -> str:
    result = subprocess.run(
        ["gcloud", "auth", "print-identity-token"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def fetch_episode_paths(filter_str: str, env: str = "prod") -> list[dict]:
    """Call platform API and return [{"episode_id": int, "object_storage_path": "gs://..."}].

    Args:
        filter_str: URL query string copied from the platform UI
                    (e.g. "task=pick_red_brick&rating=excellent&hide_ignored=true").
        env: "prod" or "nonprod".
    """
    env_cfg = ENVS[env]
    params = urllib.parse.parse_qsl(filter_str.lstrip("?"), keep_blank_values=True)
    qs = urllib.parse.urlencode(params)

    base_url = _get_api_url(env_cfg["project"], env_cfg["region"])
    token = _get_auth_token()

    url = f"{base_url}/api/episodes/paths?{qs}"
    req = urllib.request.Request(url)
    req.add_header("Authorization", f"Bearer {token}")

    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read())

    return data["episodes"]
