"""Is a MetaTrader terminal already running? — without starting one.

`MetaTrader5.initialize()` launches the terminal when none is running.
That is convenient for a script and wrong for an application: the
Dashboard polls the connection status, so simply opening QuantEngine
opened MetaTrader, every time, whether or not the person wanted it.

Nothing in the app needs the terminal to be running. Installing an EA
writes files into its data directory, and reading the Pilot's reports
reads files out of one; both work on a closed terminal. The connection
is only for live spread (REQ-DATA-1.3) and it is only needed while a
campaign is being configured.

So the rule is: attach to a terminal that is already open, never open
one. Launching is a thing the user asks for, and there is a button.

ctypes rather than psutil because this is one call into a Windows API
that is already there, and a dependency that ships in the installer for
one process lookup is a dependency that has to be maintained forever.
"""

from __future__ import annotations

import ctypes
import sys
import time
from ctypes import wintypes

# Both the 64-bit terminal and the older 32-bit build.
TERMINAL_PROCESS_NAMES = ("terminal64.exe", "terminal.exe")

# The Dashboard polls; the answer does not change second to second, and
# enumerating processes on every poll would be a silly thing to spend a
# laptop's battery on.
_CACHE_SECONDS = 5.0
_cache: tuple[float, bool] | None = None

TH32CS_SNAPPROCESS = 0x00000002
MAX_PATH = 260


class PROCESSENTRY32(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD),
        ("cntUsage", wintypes.DWORD),
        ("th32ProcessID", wintypes.DWORD),
        ("th32DefaultHeapID", ctypes.POINTER(ctypes.c_ulong)),
        ("th32ModuleID", wintypes.DWORD),
        ("cntThreads", wintypes.DWORD),
        ("th32ParentProcessID", wintypes.DWORD),
        ("pcPriClassBase", ctypes.c_long),
        ("dwFlags", wintypes.DWORD),
        ("szExeFile", ctypes.c_char * MAX_PATH),
    ]


def _running_process_names() -> set[str]:
    kernel32 = ctypes.windll.kernel32
    snapshot = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    if snapshot == -1:
        return set()

    names: set[str] = set()
    entry = PROCESSENTRY32()
    entry.dwSize = ctypes.sizeof(PROCESSENTRY32)
    try:
        if not kernel32.Process32First(snapshot, ctypes.byref(entry)):
            return names
        while True:
            names.add(entry.szExeFile.decode(errors="ignore").lower())
            if not kernel32.Process32Next(snapshot, ctypes.byref(entry)):
                break
    finally:
        kernel32.CloseHandle(snapshot)
    return names


def terminal_is_running(use_cache: bool = True) -> bool:
    """True when a MetaTrader terminal is already open.

    Any failure answers False: the cost of being wrong that way is a
    Dashboard card saying "not connected" when it could have said
    otherwise. The cost of being wrong the other way is launching a
    trading terminal nobody asked for.
    """
    global _cache

    if sys.platform != "win32":
        return False

    now = time.monotonic()
    if use_cache and _cache is not None and now - _cache[0] < _CACHE_SECONDS:
        return _cache[1]

    try:
        running = any(name in _running_process_names() for name in TERMINAL_PROCESS_NAMES)
    except Exception:
        running = False

    _cache = (now, running)
    return running


def forget() -> None:
    """Drops the cache, so a launch is visible immediately rather than
    up to five seconds later."""
    global _cache
    _cache = None
