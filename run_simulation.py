#!/usr/bin/env python3
"""
run_simulation.py — Full End-to-End Simulation
===============================================
Runs the complete semantic routing pipeline on your laptop.

Usage:
    python run_simulation.py                          # interactive demo
    python run_simulation.py --query "my question"   # single query
    python run_simulation.py --batch                 # run all test cases
    python run_simulation.py --production            # use real Ollama models
"""

import argparse
import asyncio
import logging
import os
import sys
import time

# Pretty terminal output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich import print as rprint
    RICH = True
except ImportError:
    RICH = False
    print("Note: install 'rich' for prettier output: pip install rich")

logging.basicConfig(
    level=logging.WARNING,  # suppress debug noise in demo
    format="%(levelname)s %(name)s: %(message)s"
)

console = Console() if RICH else None


# ── Demo Queries ───────────────────────────────────────────────────

DEMO_QUERIES = [
    # Personal only
    "What is my blood pressure medication?",
    "When is my next doctor appointment?",
    "Am I allergic to penicillin?",

    # Environmental only
    "What does this sign say?",
    "Is there a pharmacy nearby?",
    "Where is the nearest exit?",

    # Mixed - parallel
    "What medication am I holding and is there a pharmacy nearby?",
    "What food is this and am I allergic to anything on this menu?",

    # Mixed - sequential (implicit)
    "Where is my gate and how do I get there from security?",
    "Am I allergic to what I'm about to eat and should I avoid this restaurant?",

    # Follow-up (tests session injection)
    "What time does it close?",   # after a pharmacy query
]


def print_banner():
    if RICH:
        console.print(Panel(
            "[bold cyan]Semantic Query Routing System[/bold cyan]\n"
            "[dim]Privacy-Aware Two-Tier Edge–Fog Routing\n"
            "NSF IRES | HPCC Lab UNT | IMDEA Networks[/dim]",
            expand=False,
        ))
    else:
        print("=" * 60)
        print("  Semantic Query Routing System")
        print("  Privacy-Aware Two-Tier Edge-Fog Routing")
        print("=" * 60)


def print_result(result, query_num: int = None):
    clf = result.classification
    route_colors = {
        "edge_only": "green",
        "fog_only": "blue",
        "mixed_parallel": "yellow",
        "mixed_sequential": "magenta",
    }
    label_icons = {
        "Personal": "🔒",
        "Environmental": "🌍",
        "Mixed": "⚡",
    }

    if RICH:
        color = route_colors.get(result.route, "white")
        icon  = label_icons.get(clf.label, "?")

        header = f"{f'[{query_num}] ' if query_num else ''}{icon} [bold]{clf.label}[/bold]"
        header += f" → [{color}]{result.route}[/{color}]"
        header += f"  [dim]conf={clf.confidence:.2f} | {result.total_latency_ms:.0f}ms[/dim]"

        console.print(f"\n[bold white]Query:[/bold white] {result.query}")
        console.print(header)

        if result.decomposition:
            console.print(
                f"  [dim]↳ personal:  {result.decomposition.personal_subquery}[/dim]"
            )
            console.print(
                f"  [dim]↳ environ:   {result.decomposition.environmental_subquery}[/dim]"
            )
            if result.dependency:
                console.print(
                    f"  [dim]↳ dispatch:  {result.dependency.mode} ({result.dependency.reason[:60]}...)[/dim]"
                )

        if result.session_injected:
            console.print("  [dim yellow]↳ session context injected[/dim yellow]")

        console.print(Panel(result.final_response, title="Response", border_style=color))

    else:
        sep = "-" * 60
        print(f"\n{sep}")
        if query_num:
            print(f"[{query_num}] ", end="")
        print(f"Query:   {result.query}")
        print(f"Label:   {clf.label} (conf={clf.confidence:.2f}, via={clf.triggered_by})")
        print(f"Route:   {result.route} | {result.total_latency_ms:.0f}ms")
        if result.decomposition:
            print(f"  Personal sub: {result.decomposition.personal_subquery}")
            print(f"  Environ sub:  {result.decomposition.environmental_subquery}")
            if result.dependency:
                print(f"  Dependency:   {result.dependency.mode}")
        if result.session_injected:
            print("  [Session context injected]")
        print(f"\nResponse:\n{result.final_response}")
        print(sep)


def print_stats(results: list):
    total = len(results)
    by_route = {}
    for r in results:
        by_route[r.route] = by_route.get(r.route, 0) + 1
    avg_latency = sum(r.total_latency_ms for r in results) / total

    if RICH:
        table = Table(title="Simulation Summary", show_header=True, header_style="bold")
        table.add_column("Route", style="cyan")
        table.add_column("Count", justify="right")
        table.add_column("% of queries", justify="right")
        for route, count in sorted(by_route.items()):
            table.add_row(route, str(count), f"{count/total*100:.0f}%")
        table.add_row("─" * 20, "─" * 5, "─" * 12)
        table.add_row("[bold]Total[/bold]", str(total), "")
        console.print(table)
        console.print(f"[dim]Avg latency: {avg_latency:.0f}ms (simulation mode)[/dim]")
    else:
        print("\n" + "=" * 60)
        print("SUMMARY")
        for route, count in sorted(by_route.items()):
            print(f"  {route}: {count} ({count/total*100:.0f}%)")
        print(f"  Total: {total} | Avg latency: {avg_latency:.0f}ms")
        print("=" * 60)


# ── Main ──────────────────────────────────────────────────────────

async def main_async(args):
    print_banner()

    # Check seeds exist
    if not os.path.exists("data/edge_context.db"):
        print("\n[!] Edge store not seeded. Running seeder first...")
        import subprocess
        subprocess.run([sys.executable, "scripts/seed_edge_db.py"], check=True)

    print("\nLoading pipeline...")
    from edge.pipeline import RoutingPipeline

    config_override = {}
    if args.production:
        config_override["simulation_mode"] = False
        print("[!] Production mode: Ollama must be running on localhost:11434")
        print("[!] Fog server must be reachable at fog_server_url in config.yaml")

    pipeline = RoutingPipeline.from_config(config_override)
    print("✓ Pipeline ready.\n")

    if args.query:
        # Single query mode
        result = await pipeline.process(args.query)
        print_result(result)
        return

    if args.batch:
        queries = DEMO_QUERIES
    else:
        # Interactive mode
        queries = []
        if RICH:
            console.print("[bold]Interactive mode.[/bold] Type a query and press Enter. 'quit' to exit.")
            console.print("[dim]Or run with --batch to process all demo queries.[/dim]\n")
        else:
            print("Interactive mode. Type a query and press Enter. 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if query.lower() in ("quit", "exit", "q"):
                break
            if not query:
                continue
            result = await pipeline.process(query)
            print_result(result)
        return

    # Batch mode
    results = []
    for i, query in enumerate(queries, 1):
        if RICH:
            with console.status(f"Processing query {i}/{len(queries)}..."):
                result = await pipeline.process(query)
        else:
            result = await pipeline.process(query)
        print_result(result, query_num=i)
        results.append(result)

    print_stats(results)


def main():
    parser = argparse.ArgumentParser(description="Semantic Query Routing Simulation")
    parser.add_argument("--query", type=str, help="Process a single query")
    parser.add_argument("--batch", action="store_true", help="Run all demo queries")
    parser.add_argument("--production", action="store_true",
                        help="Use real Ollama models (simulation_mode=False)")
    parser.add_argument("--verbose", action="store_true", help="Show debug logs")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
