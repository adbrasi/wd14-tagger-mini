"""Rich UI helpers for data_araknideo CLI.

Centralizes all user interaction: prompts, progress bars, panels, tables.
All other modules import from here instead of using print() directly.
"""
from typing import List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress,
    BarColumn,
    DownloadColumn,
    MofNCompleteColumn,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table
from rich.text import Text

console = Console()


def print_banner():
    banner = Text()
    banner.append("DATA ARAKNIDEO\n", style="bold cyan")
    banner.append("dataset preprocessing & tagging pipeline", style="dim")
    console.print(Panel(banner, border_style="cyan", padding=(1, 4)))


def print_section(title: str):
    console.print(f"\n[bold yellow]{'─' * 60}[/]")
    console.print(f"[bold yellow]  {title}[/]")
    console.print(f"[bold yellow]{'─' * 60}[/]\n")


def print_success(msg: str):
    console.print(f"[bold green]✓[/] {msg}")


def print_warning(msg: str):
    console.print(f"[bold yellow]![/] {msg}")


def print_error(msg: str):
    console.print(f"[bold red]✗[/] {msg}")


def print_info(msg: str):
    console.print(f"[dim]→[/] {msg}")


def ask_input(prompt: str, default: str = "") -> str:
    return Prompt.ask(f"[bold]{prompt}[/]", default=default)


def ask_choice(prompt: str, options: List[str], default: int = 1) -> int:
    if not options:
        raise ValueError("ask_choice requires at least one option")
    console.print(f"\n[bold]{prompt}[/]")
    for i, opt in enumerate(options, 1):
        marker = " [cyan]*[/]" if i == default else ""
        console.print(f"  [bold]{i})[/] {opt}{marker}")
    while True:
        raw = Prompt.ask("Choice", default=str(default))
        try:
            choice = int(raw)
            if 1 <= choice <= len(options):
                return choice
        except ValueError:
            pass
        console.print(f"  [red]Enter a number between 1 and {len(options)}[/]")


def ask_yes_no(prompt: str, default: bool = True) -> bool:
    return Confirm.ask(f"[bold]{prompt}[/]", default=default)


def ask_int(prompt: str, default: int = 1, minimum: int = 1) -> int:
    while True:
        val = IntPrompt.ask(f"[bold]{prompt}[/]", default=default)
        if val >= minimum:
            return val
        console.print(f"  [red]Must be >= {minimum}[/]")


def make_progress(**kwargs) -> Progress:
    """General-purpose progress bar."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        **kwargs,
    )


def make_download_progress(**kwargs) -> Progress:
    """Download-specific progress bar with transfer speed."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
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
