#!/usr/bin/env python3
"""Start the PathSnap API so that other machines on the network can reach it (Laptop 1).

Usage (from the repository root, with the virtual environment active):
    python scripts/run_api.py                 # listens on 0.0.0.0:8000
    python scripts/run_api.py --port 8080

Prints the LAN address(es) to put in the UI's VITE_API_URL on Laptop 2.
Environment: PATHSNAP_CORS_ORIGINS (see src/pathsnap/api/main.py) to restrict allowed UI origins.
"""

from __future__ import annotations

import argparse
import socket


def lan_addresses() -> list[str]:
    """IPv4 addresses of this machine that other devices on the network can use."""
    addrs = set()
    try:  # address of the interface used for outbound traffic (no packet is actually sent)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            addrs.add(s.getsockname()[0])
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            addrs.add(info[4][0])
    except OSError:
        pass
    return sorted(a for a in addrs if not a.startswith("127."))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="0.0.0.0", help="interface to bind (0.0.0.0 = all, reachable from the LAN)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reload", action="store_true", help="auto-reload on code changes (development only)")
    args = ap.parse_args()

    import uvicorn

    print("PathSnap API starting")
    print(f"  this laptop : http://127.0.0.1:{args.port}/api/health")
    for ip in lan_addresses():
        print(f"  from the LAN: http://{ip}:{args.port}/api/health   -> on Laptop 2 set VITE_API_URL=http://{ip}:{args.port}")
    print(f"  API docs    : http://127.0.0.1:{args.port}/docs\n")
    uvicorn.run("pathsnap.api.main:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
