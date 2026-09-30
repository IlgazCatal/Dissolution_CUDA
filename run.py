# run.py  —  interactive terminal picker for the dissolution simulator
import sys
import numpy as np

from main import (
    SOLUTES, SOLVENTS, T_REF, main_driver,
)


# ------------------------------------------------------------------
#  Small terminal helpers
# ------------------------------------------------------------------

def _hr(char="-", width=64):
    print(char * width)


def _print_header(title):
    _hr("=")
    print(f"  {title}")
    _hr("=")


def _fits_float(s):
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


def _fits_int(s):
    try:
        int(s)
        return True
    except (TypeError, ValueError):
        return False


def _ask(prompt, default=None, validator=None, cast=str):
    """Prompt the user, re-ask on invalid input, return default on empty input."""
    suffix = f" [{default}]" if default is not None else ""
    while True:
        raw = input(f"{prompt}{suffix}: ").strip()
        if raw == "" and default is not None:
            return default
        if validator is not None and not validator(raw):
            print("  -> invalid input, please try again.")
            continue
        try:
            return cast(raw)
        except (TypeError, ValueError):
            print("  -> invalid input, please try again.")


# ------------------------------------------------------------------
#  Menu rendering + selection
# ------------------------------------------------------------------

def _list_entries(library, width_key=14):
    """Return an ordered list of (key, info) tuples for a library dict."""
    return list(library.items())


def _pick_from_library(library, title, extra_col=None, allow_back=False):
    """
    Show numbered menu; return the chosen key (str).
    If allow_back=True, entering 'b' returns None.
    """
    entries = _list_entries(library)
    _print_header(title)
    for i, (key, info) in enumerate(entries, start=1):
        line = f"  {i:2d}. {info['name']}"
        if extra_col is not None:
            try:
                line += extra_col(key, info)
            except Exception:
                pass
        print(line)
    _hr()
    if allow_back:
        print("  b.  back")
        _hr()

    keys_lower = [k.lower() for k, _ in entries]
    while True:
        raw = input("Your choice (number or name): ").strip().lower()
        if raw == "":
            print("  -> please enter a number or a name.")
            continue
        if allow_back and raw in ("b", "back"):
            return None

        # by number
        if _fits_int(raw):
            n = int(raw)
            if 1 <= n <= len(entries):
                return entries[n - 1][0]
            print(f"  -> enter a number between 1 and {len(entries)}.")
            continue

        # by name (exact key match)
        if raw in keys_lower:
            return entries[keys_lower.index(raw)][0]

        # by partial name match (info['name'])
        matches = [k for k, info in entries if raw in info["name"].lower()]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            print(f"  -> ambiguous; matches: {', '.join(matches)}")
            continue
        print("  -> not found. Try a number or the full key (e.g. 'nacl').")


def _solvent_extra(key, info):
    return (f"   (Tb = {info['Tb_K']:.1f} K, "
            f"D_scale = {info['D_scale']:.2f})")


def _solute_extra(key, info):
    return (f"   (prefers: {', '.join(info['preferred_solvents'])})")


# ------------------------------------------------------------------
#  Interactive configuration flow
# ------------------------------------------------------------------

def interactive_config():
    print()
    _print_header("Dissolution Simulator  —  Solute / Solvent / Temperature")
    print("  Choose a solute and a solvent from the lists below.")
    print("  You can enter the number, the short key (e.g. 'nacl'),")
    print("  or a fragment of the display name (e.g. 'chloride').")
    _hr()

    solute_key = None
    while solute_key is None:
        solute_key = _pick_from_library(
            SOLUTES, "Available solutes", extra_col=_solute_extra
        )

    print()
    solvent_key = None
    while solvent_key is None:
        solvent_key = _pick_from_library(
            SOLVENTS, "Available solvents", extra_col=_solvent_extra
        )

    # Temperature selection, with validation against the solvent liquid range
    solvent = SOLVENTS[solvent_key]
    Tm, Tb = solvent["Tm_K"], solvent["Tb_K"]
    print()
    _print_header("Temperature")
    print(f"  Solvent: {solvent['name']}")
    print(f"  Liquid range: {Tm:.2f} K  ...  {Tb:.2f} K")
    print(f"  Reference   : {T_REF:.2f} K  (default)")

    def _valid_T(s):
        return _fits_float(s) and float(s) > 0.0

    T_K = _ask("  Temperature (K)",
               default=T_REF,
               validator=_valid_T,
               cast=float)

    if not (Tm + 1.0 < T_K < Tb - 1.0):
        print(f"  [warning] T = {T_K} K is outside the normal liquid range "
              f"of {solvent['name']}.  The simulation will still run, "
              f"but interpret results with caution.")

    # Optional plot control
    print()
    show_plot_raw = _ask("  Show plot window at the end? (y/n)",
                         default="y").strip().lower()
    show_plot = show_plot_raw not in ("n", "no", "0", "false")

    save_name = _ask("  Save figure as",
                     default="dissolution_result.png").strip()
    if not save_name:
        save_name = "dissolution_result.png"

    # Confirm
    print()
    _hr("=")
    print("  Summary")
    _hr("=")
    print(f"  Solute    : {SOLUTES[solute_key]['name']}")
    print(f"  Solvent   : {solvent['name']}")
    print(f"  T         : {T_K:.2f} K")
    print(f"  Show plot : {show_plot}")
    print(f"  Save to   : {save_name}")
    _hr("=")

    confirm = _ask("  Start simulation? (y/n)", default="y").strip().lower()
    if confirm in ("n", "no", "0", "false"):
        print("Aborted.")
        sys.exit(0)

    return dict(
        solute_key=solute_key,
        solvent_key=solvent_key,
        T_K=T_K,
        show_plot=show_plot,
        save_path=save_name,
    )


# ------------------------------------------------------------------
#  Main
# ------------------------------------------------------------------

def main():
    opts = interactive_config()
    main_driver(**opts)


if __name__ == "__main__":
    main()