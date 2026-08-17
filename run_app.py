"""Arranca Streamlit accesible desde el celular en la misma red (LAN)."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PORT = 8501
FIREWALL_RULE = "CrackExpertAI-Streamlit-8501"


def lan_ipv4_addresses() -> list[str]:
    preferred: str | None = None
    found: set[str] = set()
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.connect(("8.8.8.8", 80))
        preferred = probe.getsockname()[0]
        found.add(preferred)
        probe.close()
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            found.add(info[4][0])
    except OSError:
        pass

    def _usable(ip: str) -> bool:
        return not ip.startswith(("127.", "169.254."))

    usable = [ip for ip in found if _usable(ip)]
    if preferred and preferred in usable:
        return [preferred] + sorted(ip for ip in usable if ip != preferred)
    return sorted(usable)


def preferred_lan_ip() -> str:
    ips = lan_ipv4_addresses()
    return ips[0] if ips else "127.0.0.1"


def port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.4)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def listening_pid(port: int) -> int | None:
    if os.name != "nt":
        try:
            out = subprocess.check_output(["lsof", "-i", f"TCP:{port}", "-sTCP:LISTEN", "-t"], text=True)
        except (OSError, subprocess.CalledProcessError):
            return None
        for line in out.split():
            if line.strip().isdigit():
                return int(line.strip())
        return None
    try:
        out = subprocess.check_output(["netstat", "-ano", "-p", "tcp"], text=True, errors="replace")
    except (OSError, subprocess.CalledProcessError):
        return None
    needle = f":{port}"
    for line in out.splitlines():
        upper = line.upper()
        if "LISTENING" not in upper and "ESCUCHANDO" not in upper:
            continue
        if needle not in line:
            continue
        parts = line.split()
        if parts and parts[-1].isdigit():
            return int(parts[-1])
    return None


def process_name(pid: int) -> str:
    if os.name == "nt":
        try:
            out = subprocess.check_output(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                text=True,
                errors="replace",
            )
        except (OSError, subprocess.CalledProcessError):
            return ""
        line = out.strip().splitlines()[0] if out.strip() else ""
        if line.startswith('"'):
            return line.split('","')[0].strip('"').lower()
        return line.split()[0].lower() if line else ""
    try:
        out = subprocess.check_output(["ps", "-p", str(pid), "-o", "comm="], text=True)
        return out.strip().lower()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _is_our_server(name: str) -> bool:
    base = Path(name).name.lower()
    return base in {"python.exe", "pythonw.exe", "python", "streamlit.exe", "streamlit"}


def stop_pid(pid: int) -> bool:
    if os.name == "nt":
        done = subprocess.run(
            ["taskkill", "/PID", str(pid), "/F"],
            capture_output=True,
            check=False,
        )
        return done.returncode == 0
    done = subprocess.run(["kill", str(pid)], capture_output=True, check=False)
    return done.returncode == 0


def free_or_reuse_port(port: int) -> str:
    """Libera 8501 si lo ocupa un Python/Streamlit previo. Vacío = puerto libre."""
    if not port_in_use(port):
        return "libre"
    pid = listening_pid(port)
    if pid is None:
        return "ocupado"
    name = process_name(pid)
    if _is_our_server(name):
        print(f"El puerto {port} lo tenía un proceso anterior ({name} PID {pid}). Se detiene para reiniciar.")
        stop_pid(pid)
        for _ in range(20):
            time.sleep(0.25)
            if not port_in_use(port):
                return "liberado"
        return "ocupado"
    print(f"El puerto {port} está ocupado por {name or 'otro programa'} (PID {pid}).")
    return "ocupado"


def try_open_windows_firewall(port: int) -> str:
    if os.name != "nt":
        return "no aplica (no es Windows)"
    show = subprocess.run(
        ["netsh", "advfirewall", "firewall", "show", "rule", f"name={FIREWALL_RULE}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if show.returncode == 0 and FIREWALL_RULE in (show.stdout or ""):
        return "regla de firewall ya existe"
    add = subprocess.run(
        [
            "netsh",
            "advfirewall",
            "firewall",
            "add",
            "rule",
            f"name={FIREWALL_RULE}",
            "dir=in",
            "action=allow",
            "protocol=TCP",
            f"localport={port}",
            "profile=any",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if add.returncode == 0:
        return "regla de firewall creada (TCP 8501, todos los perfiles)"
    return (
        "no se pudo abrir el firewall (hace falta Administrador). "
        "PowerShell como admin:\n"
        f'  netsh advfirewall firewall add rule name="{FIREWALL_RULE}" '
        f"dir=in action=allow protocol=TCP localport={port} profile=any"
    )


def print_urls(ip: str, ips: list[str], port: int) -> None:
    print("=" * 64)
    print("CrackExpert AI")
    print("=" * 64)
    print(f"En este PC:     http://127.0.0.1:{port}")
    print(f"En el celular:  http://{ip}:{port}")
    if len(ips) > 1:
        print("Otras IPs de este PC:")
        for extra in ips:
            print(f"  http://{extra}:{port}")
    print()
    print("Misma WiFi. Si no abre, use un hotspot del portátil.")
    print("En el celular: Foto → Examinar → Cámara / Tomar foto.")
    print("=" * 64)
    print()


def main() -> int:
    os.chdir(ROOT)
    ip = preferred_lan_ip()
    ips = lan_ipv4_addresses()
    fw = try_open_windows_firewall(PORT)
    print_urls(ip, ips, PORT)
    print(f"Firewall: {fw}")
    print()

    status = free_or_reuse_port(PORT)
    if status == "ocupado" and port_in_use(PORT):
        print(f"Ya hay una app en el puerto {PORT}. Ábrala en el celular:")
        print(f"  http://{ip}:{PORT}")
        print("Si quiere reiniciar, cierre esa ventana y vuelva a ejecutar python run_app.py")
        return 0

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(ROOT / "app.py"),
        "--server.address=0.0.0.0",
        f"--server.port={PORT}",
        "--server.headless=true",
        "--server.enableCORS=false",
        "--server.enableXsrfProtection=false",
        "--server.enableWebsocketCompression=false",
        f"--browser.serverAddress={ip}",
        f"--browser.serverPort={PORT}",
        "--browser.gatherUsageStats=false",
    ]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
