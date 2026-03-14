"""
Stage DLL loader & device initializer for Standa/xi-mc controllers (XIMC / pyximc).

Extracted from the original script so camera/logic code can simply import this
module and call `init_stage(...)` to get `(lib, device_id1, device_id2)`.

Notes
-----
- Keeps the original helper functions: `move`, `get_position`, `set_speed`.
- Works on Windows with manual wrapper/DLL paths (defaults match your current
  project layout). You can override paths via function args.
- If no real device is detected and `prefer_virtual=True`, a virtual device is
  opened using an xi-emu URI backed by a temp file.
- Errors raise Python exceptions instead of exiting the process.

"""
from __future__ import annotations

import os
import sys
import platform
import tempfile
import pathlib
from ctypes import *  # noqa: F401,F403 (needed for create_string_buffer, byref, cast)
from typing import Optional, Tuple

# Python 3 URI helpers (for virtual-device URI only)
if sys.version_info >= (3, 0):
    import urllib.parse  # noqa: F401

# -----------------------------
# Public API (what callers import)
# -----------------------------
__all__ = [
    "init_stage",
    "move",
    "get_position",
    "set_speed",
    "close_device_safely",
]

# Module-global handle to the imported pyximc module
_px = None


def _prepend_paths(wrapper_path: Optional[str], libdir: Optional[str]) -> None:
    """Put wrapper and DLL directories on sys.path / DLL search path for Windows.

    We mirror the behavior from the original script.
    """
    if wrapper_path:
        if wrapper_path not in sys.path:
            sys.path.append(wrapper_path)

    if platform.system() == "Windows" and libdir:
        if libdir not in sys.path:
            sys.path.append(libdir)
        if sys.version_info >= (3, 8):
            # Makes the DLLs discoverable without PATH pollution
            os.add_dll_directory(libdir)
        else:
            os.environ["Path"] = libdir + ";" + os.environ.get("Path", "")


def _import_pyximc() -> None:
    """Import pyximc as a module (allowed inside a function) and keep a handle.

    We avoid star-imports at function scope because Python forbids them.
    """
    global _px
    try:
        import pyximc as _px_mod  # type: ignore
        _px = _px_mod
    except ImportError as err:
        raise ImportError(
            "Can't import pyximc. Check the wrapper path and that pyximc.py and the DLLs "
            "(bindy.dll, libximc.dll, xiwrapper.dll) are present."
        ) from err
    except OSError as err:
        if platform.system() == "Windows":
            if getattr(err, "winerror", None) == 193:
                hint = "DLL architecture mismatch (bindy/libximc/xiwrapper vs OS)."
            elif getattr(err, "winerror", None) == 126:
                hint = (
                    "A required DLL is missing. Installing the VC++ redistributable from the "
                    "ximc/winXX folder may be required."
                )
            else:
                hint = str(err)
            raise OSError(f"Failed to load libximc dependencies: {hint}") from err
        raise OSError(
            "Can't load libximc shared libraries. Ensure libximc is installed and the "
            "interpreter architecture matches the system."
        ) from err


def _lib():
    """Access the `lib` object exported by pyximc via our module handle."""
    if _px is None:
        _import_pyximc()
    return _px.lib 


def _print_version(verbose: bool) -> None:
    if not verbose:
        return
    sbuf = create_string_buffer(64)
    _lib().ximc_version(sbuf)
    print("Library loaded")
    print("Library version:", sbuf.raw.decode().rstrip("\0"))


def _set_bindy_key_if_any(keyfile_path: Optional[str]) -> None:
    """Configure network key (Bindy). Optional; only needed for xi-net devices."""
    if keyfile_path and os.path.exists(keyfile_path):
        result = _lib().set_bindy_key(keyfile_path.encode("utf-8"))
        # If this fails, fall back to searching in the current directory
        if result != _px.Result.Ok:  
            _lib().set_bindy_key(b"keyfile.sqlite")
    else:
        # Fallback to looking in CWD
        _lib().set_bindy_key(b"keyfile.sqlite")


def _enumerate_devices(verbose: bool = True):
    probe_flags = _px.EnumerateFlags.ENUMERATE_PROBE + _px.EnumerateFlags.ENUMERATE_NETWORK  # type: ignore[attr-defined]
    devenum = _lib().enumerate_devices(probe_flags, b"addr=")

    dev_count = _lib().get_device_count(devenum)
    if verbose:
        print("Device enum handle:", repr(devenum))
        print("Device count:", dev_count)
        controller_name = _px.controller_name_t()  # type: ignore[attr-defined]
        for dev_ind in range(dev_count):
            enum_name = _lib().get_device_name(devenum, dev_ind)
            res = _lib().get_enumerate_device_controller_name(devenum, dev_ind, byref(controller_name))
            if res == _px.Result.Ok:  # type: ignore[attr-defined]
                print(
                    f"Enumerated device #{dev_ind} name (port): {enum_name!r}. "
                    f"Friendly name: {controller_name.ControllerName!r}."
                )
    return devenum, dev_count


def _open_virtual_device() -> bytes:
    """Create an xi-emu URI backed by a temp file and return it as bytes.

    Avoids backslashes in f-string expressions by converting the Windows path to
    POSIX form first.
    """
    temp_path = os.path.join(tempfile.gettempdir(), "virtual_controller.bin")
    if os.name == "nt":
        temp_posix = pathlib.PureWindowsPath(temp_path).as_posix()
    else:
        temp_posix = temp_path

    uri = f"xi-emu:///{temp_posix}"
    return uri.encode("utf-8")


def _open_devices(devenum, dev_count: int, prefer_virtual: bool, verbose: bool) -> Tuple[int, Optional[int], int]:
    """Open up to two devices. Returns (device_id1, device_id2, flag_virtual)."""
    flag_virtual = 0

    open_nameX: Optional[bytes] = None
    open_nameY: Optional[bytes] = None

    if dev_count > 0:
        open_nameX = _lib().get_device_name(devenum, 0)
        if dev_count > 1:
            open_nameY = _lib().get_device_name(devenum, 1)
    elif prefer_virtual:
        flag_virtual = 1
        open_nameX = _open_virtual_device()
        open_nameY = None
        if verbose:
            print("No real controllers detected. Opening virtual controller (xi-emu).")

    if not open_nameX:
        raise RuntimeError("No stage controller found and virtual device disabled.")

    if isinstance(open_nameX, str):  
        open_nameX = open_nameX.encode()
    if isinstance(open_nameY, str):
        open_nameY = open_nameY.encode()

    if verbose:
        print("\nOpen device", repr(open_nameX))
    device_id1 = _lib().open_device(open_nameX)
    if verbose:
        print("Device id (X):", repr(device_id1))

    device_id2 = None
    if open_nameY:
        if verbose:
            print("\nOpen device", repr(open_nameY))
        device_id2 = _lib().open_device(open_nameY)
        if verbose:
            print("Device id (Y):", repr(device_id2))

    if flag_virtual == 1 and verbose:
        print("\nThe real controller is not found or busy with another app.")
        print("The virtual controller is opened to check the operation of the library.")
        print("If you want to open a real controller, connect it or close the application that uses it.")

    return device_id1, device_id2, flag_virtual


# -----------------------------
# Public initializer
# -----------------------------

def init_stage(
    *,
    wrapper_path: Optional[str] = r"C:/lab/MBI/libximc_2.13.2/ximc-2.13.3/ximc/crossplatform/wrappers/python",
    libdir: Optional[str] = r"C:/lab/MBI/libximc_2.13.2/ximc-2.13.3/ximc/win64",
    keyfile_path: Optional[str] = r"C:/Users/lab/Desktop/LIA/Stage/integration/libximc_2.13.2/ximc-2.13.3/ximc/win32/keyfile.sqlite",
    prefer_virtual: bool = False,
    verbose: bool = True,
) -> Tuple[any, int, Optional[int]]:
    """Initialize pyximc library and open up to two devices.

    Returns
    -------
    (lib, device_id1, device_id2)
    """
    _prepend_paths(wrapper_path, libdir)
    _import_pyximc()
    _print_version(verbose)
    _set_bindy_key_if_any(keyfile_path)

    devenum, dev_count = _enumerate_devices(verbose=verbose)
    device_id1, device_id2, _flag_virtual = _open_devices(devenum, dev_count, prefer_virtual, verbose)

    # Return the ctypes lib object so caller can pass it into helper functions
    return _lib(), device_id1, device_id2


# -----------------------------
# Helper functions 
# -----------------------------

def move(lib, device_id, distance: int, udistance: int) -> None:
    print(f"\nGoing to {distance} steps, {udistance} microsteps")
    _ = lib.command_move(device_id, distance, udistance)


def get_position(lib, device_id):
    print("\nRead position")
    x_pos = _px.get_position_t()  # type: ignore[attr-defined]
    result = lib.get_position(device_id, byref(x_pos))
    if result == _px.Result.Ok:  # type: ignore[attr-defined]
        print(f"Position: {x_pos.Position} steps, {x_pos.uPosition} microsteps")
    return x_pos.Position, x_pos.uPosition


def set_speed(lib, device_id, speed: int) -> None:
    print("\nSet speed")
    mvst = _px.move_settings_t()  # type: ignore[attr-defined]
    result = lib.get_move_settings(device_id, byref(mvst))
    print("Read command result:", repr(result))
    print(f"The speed was equal to {mvst.Speed}. We will change it to {speed}")
    mvst.Speed = int(speed)
    result = lib.set_move_settings(device_id, byref(mvst))
    print("Write command result:", repr(result))


def close_device_safely(lib, device_id: Optional[int]) -> None:
    """Helper: close a device handle if valid."""
    if device_id:
        try:
            lib.close_device(byref(cast(c_int(device_id), POINTER(c_int))))
        except Exception:
            # Be tolerant on shutdown
            pass

def wait_for_stop(lib, device_id, interval):
    print("\nWaiting for stop")
    result = lib.command_wait_for_stop(device_id, interval)
    print("Result: " + repr(result))