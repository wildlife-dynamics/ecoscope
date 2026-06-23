"""Strip embedded RT_MANIFEST resources from Windows Python extension (.pyd) files.

Workaround for long-path import failures on Windows: setuptools' MSVCCompiler
embeds a side-by-side (SxS) activation manifest into every .pyd via mt.exe.
On deeply nested paths (e.g. ``~/.pixi/envs/<name>/Lib/site-packages/...``),
SxS activation-context resolution ignores both the longPathAware process
manifest and the LongPathsEnabled registry flag, and the import fails. The
embedded manifest's VC-runtime reference is inert on conda/pixi envs (the
runtime is loaded as a plain DLL), so stripping it is safe.

Usage (Windows only)::

    python strip_pyd_manifests.py <env-or-site-packages-dir> [--dry-run]

Examples::

    python strip_pyd_manifests.py C:\\Users\\me\\.pixi\\envs\\myenv
    python strip_pyd_manifests.py .pixi\\envs\\default\\Lib\\site-packages\\astropy --dry-run
"""

from __future__ import annotations

import argparse
import ctypes
import sys
from ctypes import wintypes
from pathlib import Path

RT_MANIFEST = 24
LOAD_LIBRARY_AS_DATAFILE = 0x00000002
ERROR_RESOURCE_DATA_NOT_FOUND = 1812
ERROR_RESOURCE_TYPE_NOT_FOUND = 1813
ERROR_RESOURCE_NAME_NOT_FOUND = 1814

_ENUMRESNAMEPROC = (
    ctypes.WINFUNCTYPE(
        wintypes.BOOL,
        wintypes.HMODULE,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    )
    if sys.platform == "win32"
    else None
)

_ENUMRESLANGPROC = (
    ctypes.WINFUNCTYPE(
        wintypes.BOOL,
        wintypes.HMODULE,
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.WORD,
        ctypes.c_void_p,
    )
    if sys.platform == "win32"
    else None
)


def _kernel32():
    k = ctypes.WinDLL("kernel32", use_last_error=True)

    k.LoadLibraryExW.argtypes = [wintypes.LPCWSTR, wintypes.HANDLE, wintypes.DWORD]
    k.LoadLibraryExW.restype = wintypes.HMODULE
    k.FreeLibrary.argtypes = [wintypes.HMODULE]
    k.FreeLibrary.restype = wintypes.BOOL

    k.EnumResourceNamesW.argtypes = [
        wintypes.HMODULE,
        ctypes.c_void_p,
        _ENUMRESNAMEPROC,
        ctypes.c_void_p,
    ]
    k.EnumResourceNamesW.restype = wintypes.BOOL
    k.EnumResourceLanguagesW.argtypes = [
        wintypes.HMODULE,
        ctypes.c_void_p,
        ctypes.c_void_p,
        _ENUMRESLANGPROC,
        ctypes.c_void_p,
    ]
    k.EnumResourceLanguagesW.restype = wintypes.BOOL

    k.BeginUpdateResourceW.argtypes = [wintypes.LPCWSTR, wintypes.BOOL]
    k.BeginUpdateResourceW.restype = wintypes.HANDLE
    k.UpdateResourceW.argtypes = [
        wintypes.HANDLE,
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.WORD,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    k.UpdateResourceW.restype = wintypes.BOOL
    k.EndUpdateResourceW.argtypes = [wintypes.HANDLE, wintypes.BOOL]
    k.EndUpdateResourceW.restype = wintypes.BOOL
    return k


def _ptr_as_int_id(p) -> int | None:
    """Return the integer ID if p is a MAKEINTRESOURCE pointer, else None."""
    val = ctypes.cast(p, ctypes.c_void_p).value or 0
    return val if val < 0x10000 else None


def _find_manifest_resources(k, path: Path) -> list[tuple[int, int]]:
    """Return [(resource_id, language), ...] for every RT_MANIFEST in path."""
    hmod = k.LoadLibraryExW(str(path), None, LOAD_LIBRARY_AS_DATAFILE)
    if not hmod:
        err = ctypes.get_last_error()
        raise OSError(err, f"LoadLibraryEx failed: {ctypes.FormatError(err)}")

    ids: list[int] = []
    found: list[tuple[int, int]] = []
    try:

        def on_name(_h, _t, lp_name, _lp):
            nid = _ptr_as_int_id(lp_name)
            if nid is not None:
                ids.append(nid)
            return True

        name_cb = _ENUMRESNAMEPROC(on_name)
        if not k.EnumResourceNamesW(hmod, RT_MANIFEST, name_cb, None):
            err = ctypes.get_last_error()
            if err not in (ERROR_RESOURCE_TYPE_NOT_FOUND, ERROR_RESOURCE_NAME_NOT_FOUND):
                raise OSError(err, f"EnumResourceNames: {ctypes.FormatError(err)}")
            return found

        for nid in ids:

            def on_lang(_h, _t, _n, lang, _lp, _nid=nid):
                found.append((_nid, lang))
                return True

            lang_cb = _ENUMRESLANGPROC(on_lang)
            if not k.EnumResourceLanguagesW(hmod, RT_MANIFEST, nid, lang_cb, None):
                err = ctypes.get_last_error()
                if err not in (
                    ERROR_RESOURCE_TYPE_NOT_FOUND,
                    ERROR_RESOURCE_NAME_NOT_FOUND,
                ):
                    raise OSError(err, f"EnumResourceLanguages: {ctypes.FormatError(err)}")
    finally:
        k.FreeLibrary(hmod)
    return found


def strip(path: Path, dry_run: bool) -> list[tuple[int, int]]:
    """Strip every RT_MANIFEST resource from ``path``. Returns what was found."""
    k = _kernel32()
    resources = _find_manifest_resources(k, path)
    if not resources or dry_run:
        return resources

    handle = k.BeginUpdateResourceW(str(path), False)
    if not handle:
        err = ctypes.get_last_error()
        raise OSError(err, f"BeginUpdateResource failed: {ctypes.FormatError(err)}")

    try:
        for rid, lang in resources:
            ok = k.UpdateResourceW(handle, RT_MANIFEST, rid, lang, None, 0)
            if not ok:
                err = ctypes.get_last_error()
                if err != ERROR_RESOURCE_DATA_NOT_FOUND:
                    raise OSError(
                        err,
                        f"UpdateResource(id={rid}, lang={lang}): " f"{ctypes.FormatError(err)}",
                    )
    except BaseException:
        k.EndUpdateResourceW(handle, True)
        raise

    if not k.EndUpdateResourceW(handle, False):
        err = ctypes.get_last_error()
        raise OSError(err, f"EndUpdateResource failed: {ctypes.FormatError(err)}")
    return resources


def main() -> int:
    if sys.platform != "win32":
        print("This script must run on Windows.", file=sys.stderr)
        return 2

    ap = argparse.ArgumentParser(description="Strip RT_MANIFEST from Windows .pyd files.")
    ap.add_argument("root", type=Path, help="env or site-packages dir to walk")
    ap.add_argument("--dry-run", action="store_true", help="report only")
    ap.add_argument("--ext", default=".pyd", help="file extension (default: .pyd)")
    args = ap.parse_args()

    if not args.root.exists():
        print(f"No such path: {args.root}", file=sys.stderr)
        return 2

    files = sorted(args.root.rglob(f"*{args.ext}"))
    if not files:
        print(f"No {args.ext} files under {args.root}")
        return 0

    modified = skipped = failed = 0
    for f in files:
        try:
            resources = strip(f, args.dry_run)
        except OSError as e:
            failed += 1
            print(f"FAILED {f}: {e}", file=sys.stderr)
            continue
        if not resources:
            skipped += 1
            continue
        modified += 1
        verb = "would strip" if args.dry_run else "stripped"
        print(f"{verb} {len(resources)} manifest resource(s) from {f}")

    print(f"\nDone. {modified} modified, {skipped} had no manifest, {failed} failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
