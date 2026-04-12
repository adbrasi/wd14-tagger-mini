"""Rich UI helpers for data_araknideo CLI.

Centralizes all user interaction: prompts, progress bars, panels, tables.
All other modules import from here instead of using print() directly.
"""
import sys
from typing import List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress,
    BarColumn,
    MofNCompleteColumn,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

console = Console()


def print_banner():
    banner = Text()
    banner.append("DATA ARAKNIDEO\n", style="bold blue")
    banner.append("dataset preprocessing & tagging pipeline", style="dark_blue")
    console.print(Panel(banner, border_style="blue", padding=(1, 4)))


def print_section(title: str):
    console.print(f"\n[bold blue]{'─' * 60}[/]")
    console.print(f"[bold dark_blue]  {title}[/]")
    console.print(f"[bold blue]{'─' * 60}[/]\n")


def print_success(msg: str):
    console.print(f"[bold green]✓[/] {msg}")


def print_warning(msg: str):
    console.print(f"[bold yellow]![/] {msg}")


def print_error(msg: str):
    console.print(f"[bold red]✗[/] {msg}")


def print_info(msg: str):
    console.print(f"[dim]→[/] {msg}")


def _fix_terminal():
    """Fix backspace (^?) on broken terminals like RunPod/VastAI."""
    import os as _os
    if not getattr(_fix_terminal, "_done", False):
        if not sys.stdin.isatty() or not sys.stdout.isatty():
            _fix_terminal._done = True
            return
        try:
            _os.system("stty erase ^? 2>/dev/null")
        except Exception:
            pass
        _fix_terminal._done = True


def _clean_input(raw: str) -> str:
    """Strip control characters (backspace/delete) that broken terminals leave in input."""
    # Simulate backspace: each \x7f or \x08 deletes the previous char
    result = []
    for ch in raw:
        if ch in ("\x7f", "\x08"):  # DEL and BS
            if result:
                result.pop()
        elif ord(ch) >= 32 or ch in ("\t",):  # printable + tab
            result.append(ch)
    return "".join(result).strip()


def _prompt(text: str):
    """Print prompt text and flush immediately for terminal compatibility."""
    import sys as _sys
    _fix_terminal()
    _sys.stdout.write(text)
    _sys.stdout.flush()


def ask_input(prompt: str, default: str = "") -> str:
    suffix = f" ({default})" if default else ""
    _prompt(f"{prompt}{suffix}: ")
    try:
        raw = _clean_input(input())
    except (EOFError, KeyboardInterrupt):
        raise KeyboardInterrupt
    return raw if raw else default


def ask_choice(prompt: str, options: List[str], default: int = 1) -> int:
    if not options:
        raise ValueError("ask_choice requires at least one option")
    console.print(f"\n[bold]{prompt}[/]")
    for i, opt in enumerate(options, 1):
        marker = " [blue]*[/]" if i == default else ""
        console.print(f"  [bold]{i})[/] {opt}{marker}")
    while True:
        _prompt(f"Choice ({default}): ")
        try:
            raw = _clean_input(input())
        except (EOFError, KeyboardInterrupt):
            raise KeyboardInterrupt
        if not raw:
            return default
        try:
            choice = int(raw)
            if 1 <= choice <= len(options):
                return choice
        except ValueError:
            pass
        console.print(f"  [red]Enter a number between 1 and {len(options)}[/]")


def ask_yes_no(prompt: str, default: bool = True) -> bool:
    hint = "Y/n" if default else "y/N"
    while True:
        _prompt(f"{prompt} [{hint}]: ")
        try:
            raw = _clean_input(input()).lower()
        except (EOFError, KeyboardInterrupt):
            raise KeyboardInterrupt
        if not raw:
            return default
        if raw in ("y", "yes", "s", "sim"):
            return True
        if raw in ("n", "no", "nao", "não"):
            return False
        console.print("  [red]Enter y/yes/sim or n/no[/]")


def ask_int(prompt: str, default: int = 1, minimum: int = 1) -> int:
    while True:
        _prompt(f"{prompt} ({default}): ")
        try:
            raw = _clean_input(input())
        except (EOFError, KeyboardInterrupt):
            raise KeyboardInterrupt
        if not raw:
            return default
        try:
            val = int(raw)
            if val >= minimum:
                return val
            console.print(f"  [red]Must be >= {minimum}[/]")
        except ValueError:
            console.print(f"  [red]Enter a number[/]")


def make_progress(**kwargs) -> Progress:
    """General-purpose progress bar."""
    disable = kwargs.pop("disable", not console.is_terminal)
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        disable=disable,
        **kwargs,
    )


def print_summary_table(title: str, rows: List[tuple]):
    """Print a key-value summary table."""
    table = Table(title=title, show_header=False, border_style="dim")
    table.add_column("Key", style="bold")
    table.add_column("Value")
    for key, value in rows:
        table.add_row(key, str(value))
    console.print(table)
