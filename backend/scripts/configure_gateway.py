"""Render Caddy's hostname from the same Python config used by the app."""

import ipaddress
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


def render_gateway(template: str) -> str:
    hostname = config.TEXMIN_HOST.strip()
    try:
        ipaddress.ip_address(hostname.strip("[]"))
        valid = ":" not in hostname or (hostname.startswith("[") and hostname.endswith("]"))
    except ValueError:
        valid = len(hostname) <= 253 and all(
            re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?", label)
            for label in hostname.split(".")
        )
    if not valid:
        raise ValueError("TEXMIN_HOST must be one hostname or IP address, without a scheme, port, or path.")
    if "__TEXMIN_HOST__" not in template:
        raise ValueError("Caddy template is missing __TEXMIN_HOST__.")
    return template.replace("__TEXMIN_HOST__", hostname)


if __name__ == "__main__":
    template_path, output_path = map(Path, sys.argv[1:])
    output_path.write_text(render_gateway(template_path.read_text(encoding="utf-8")), encoding="utf-8")
    print("Gateway configuration generated from backend/config.py.")
