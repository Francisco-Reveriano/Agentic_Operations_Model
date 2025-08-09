#!/usr/bin/env python3
"""
Asynchronous + parallel Ops Overview runner with retries and CLI.
Default input path and sheet name are set from the original file.

Usage examples:
  ./ops_async_runner.py \
    --output /Data/Intermediate/Ops_Overview_Data_File_Processed.xlsx \
    --max-concurrency 10 --retries 3 --verbose
"""

import os
import sys
import argparse
import asyncio
import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

_weave_enabled = True
try:
    import weave  # type: ignore
    from weave.integrations.openai_agents.openai_agents import WeaveTracingProcessor  # type: ignore
    from agents import Runner, trace, set_trace_processors  # type: ignore
except Exception:
    weave = None  # type: ignore
    WeaveTracingProcessor = None  # type: ignore
    Runner = None  # type: ignore
    trace = None  # type: ignore
    set_trace_processors = None  # type: ignore
    _weave_enabled = False

try:
    from src.Agents.Master_Agent import q_a_agent  # type: ignore
except Exception:
    q_a_agent = None  # type: ignore


async def _async_retry(coro_factory, *, retries: int = 3, base_delay: float = 1.0, max_delay: float = 15.0, jitter: float = 0.25, logger: Optional[logging.Logger] = None):
    attempt = 0
    last_exc = None
    while attempt <= retries:
        try:
            return await coro_factory()
        except Exception as exc:
            last_exc = exc
            if attempt == retries:
                if logger:
                    logger.error("Exhausted retries after %s attempts: %s", attempt + 1, exc)
                break
            delay = min(max_delay, base_delay * (2 ** attempt))
            jitter_offset = (2 * np.random.random() - 1.0) * jitter * delay
            sleep_for = max(0.0, delay + jitter_offset)
            if logger:
                logger.warning("Attempt %s failed: %s — retrying in %.2fs", attempt + 1, exc, sleep_for)
            await asyncio.sleep(sleep_for)
            attempt += 1
    raise last_exc  # type: ignore[misc]


async def call_agent(prompt: str, *, max_turns: int, trace_name: Optional[str], retries: int, logger: logging.Logger) -> Any:
    if Runner is None or q_a_agent is None:
        raise RuntimeError("Runner/q_a_agent not available. Ensure your project deps are installed.")

    async def _do_call():
        if _weave_enabled and trace is not None and trace_name:
            from contextlib import asynccontextmanager

            @asynccontextmanager
            async def _tracer():
                try:
                    cm = trace(trace_name)
                except Exception:
                    cm = None
                try:
                    if cm:
                        cm.__enter__()
                    yield
                finally:
                    if cm:
                        cm.__exit__(None, None, None)

            async with _tracer():
                return await Runner.run(q_a_agent, prompt, max_turns=max_turns)
        else:
            return await Runner.run(q_a_agent, prompt, max_turns=max_turns)

    return await _async_retry(_do_call, retries=retries, base_delay=1.0, logger=logger)


async def process_row(row: pd.Series, sheet_name: str, *, retries: int, logger: logging.Logger, sem: asyncio.Semaphore) -> Dict[str, Any]:
    process = str(row.get("Function / Process Name", "")).strip()
    onshore = float(row.get("ONSHORE", 0) or 0)
    offshore = float(row.get("OFFSHORE", 0) or 0)

    onshore_query = (
        f"What is the 5-year cumulative percentage change that Generative AI will have "
        f"on the following process: {process} within the following Line of Business (LoB): "
        f"{sheet_name}? Keep in mind that the bank's current onshore capacity is {onshore} Full-Time Equivalent Employees."
    )
    offshore_query = (
        f"What is the 5-year cumulative percentage change that Generative AI will have "
        f"on the following process: {process} within the following Line of Business (LoB): "
        f"{sheet_name}? Keep in mind that the bank's current offshore capacity is {offshore} Full-Time Equivalent Employees."
    )

    async with sem:
        on_task = asyncio.create_task(call_agent(onshore_query, max_turns=100, trace_name=sheet_name, retries=retries, logger=logger))
        off_task = asyncio.create_task(call_agent(offshore_query, max_turns=100, trace_name=sheet_name, retries=retries, logger=logger))
        on, off = await asyncio.gather(on_task, off_task)

    def _safe(o: Any, attr: str, default: Any = None) -> Any:
        try:
            return getattr(o.final_output, attr)
        except Exception:
            return default

    return {
        "Onshore_High_Scenario_Vectors": _safe(on, "high_scenario"),
        "Onshore_Medium_Scenario_Vectors": _safe(on, "medium_scenario"),
        "Onshore_Low_Scenario_Vectors": _safe(on, "low_scenario"),
        "Onshore_High_Scenario_Reasoning": _safe(on, "high_scenario_reasoning"),
        "Onshore_Medium_Scenario_Reasoning": _safe(on, "medium_scenario_reasoning"),
        "Onshore_Low_Scenario_Reasoning": _safe(on, "low_scenario_reasoning"),
        "Onshore_CourseWork": _safe(on, "online_coursework"),
        "Offshore_High_Scenario_Vectors": _safe(off, "high_scenario"),
        "Offshore_Medium_Scenario_Vectors": _safe(off, "medium_scenario"),
        "Offshore_Low_Scenario_Vectors": _safe(off, "low_scenario"),
        "Offshore_High_Scenario_Reasoning": _safe(off, "high_scenario_reasoning"),
        "Offshore_Medium_Scenario_Reasoning": _safe(off, "medium_scenario_reasoning"),
        "Offshore_Low_Scenario_Reasoning": _safe(off, "low_scenario_reasoning"),
        "Offshore_CourseWork": _safe(off, "online_coursework"),
    }


async def amain(args) -> int:
    if load_dotenv is not None and not args.no_env:
        load_dotenv()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("ops_async_runner")

    if _weave_enabled and not args.no_weave:
        try:
            weave.init("Operation Model")  # type: ignore
            if set_trace_processors and WeaveTracingProcessor:
                set_trace_processors([WeaveTracingProcessor()])
            logger.info("Weave tracing enabled.")
        except Exception as exc:
            logger.warning("Weave init failed, continuing without tracing: %s", exc)

    if not os.path.exists(args.input):
        logger.error("Input file not found: %s", args.input)
        return 2

    logger.info("Loading Excel: %s (sheet=%s)", args.input, args.sheet)
    try:
        df = pd.read_excel(args.input, sheet_name=args.sheet)
    except Exception as exc:
        logger.error("Failed to read Excel: %s", exc)
        return 2

    df = df.copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    onshore_cols = [c for c in ["ONSHORE TEAMMATE", "ONSHORE CW", "Est. Size- ONSHORE "] if c in df.columns]
    df["ONSHORE"] = df[onshore_cols].sum(axis=1) if onshore_cols else 0
    df["OFFSHORE"] = df["Est. Size- OFFSHORE"] if "Est. Size- OFFSHORE" in df.columns else 0

    if args.limit is not None:
        df = df.head(args.limit)
        logger.info("Limiting to first %d rows", args.limit)

    max_conc = max(1, int(args.max_concurrency))
    sem = asyncio.Semaphore(max_conc)

    logger.info("Launching %d row tasks with max concurrency=%d", len(df), max_conc)

    tasks = [
        asyncio.create_task(
            process_row(row, args.sheet, retries=args.retries, logger=logger, sem=sem)
        )
        for _, row in df.iterrows()
    ]

    results: list[Dict[str, Any]] = []
    completed = 0
    total = len(tasks)

    for fut in asyncio.as_completed(tasks):
        try:
            res = await fut
        except Exception as exc:
            logger.exception("Row failed after retries: %s", exc)
            res = {k: None for k in [
                "Onshore_High_Scenario_Vectors", "Onshore_Medium_Scenario_Vectors", "Onshore_Low_Scenario_Vectors",
                "Onshore_High_Scenario_Reasoning", "Onshore_Medium_Scenario_Reasoning", "Onshore_Low_Scenario_Reasoning",
                "Onshore_CourseWork", "Offshore_High_Scenario_Vectors", "Offshore_Medium_Scenario_Vectors",
                "Offshore_Low_Scenario_Vectors", "Offshore_High_Scenario_Reasoning", "Offshore_Medium_Scenario_Reasoning",
                "Offshore_Low_Scenario_Reasoning", "Offshore_CourseWork"
            ]}
        results.append(res)
        completed += 1
        if total:
            pct = (completed / total) * 100
            logger.info("Progress: %d/%d (%.1f%%)", completed, total, pct)

    out = pd.concat([df.reset_index(drop=True), pd.DataFrame(results)], axis=1)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    try:
        logger.info("Writing Excel to %s", args.output)
        with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
            out.to_excel(writer, sheet_name=args.sheet, index=False)
    except Exception as exc:
        logger.error("Failed to write Excel: %s", exc)
        return 2

    logger.info("Done. Rows processed: %d", len(out))
    return 0


def parse_args(argv: Optional[list[str]] = None):
    parser = argparse.ArgumentParser(description="Async/parallel runner for Ops Overview prompts")
    parser.add_argument("--input", default="Data/Raw/Ops_Overview_Data_File.xlsx", help="Path to input Excel file")
    parser.add_argument("--sheet", default="Mortgage Servicing", help="Sheet name to process")
    parser.add_argument("--output", default="Data/Intermediate/Ops_Overview_Data_File_Processed.xlsx", help="Path to output Excel file")
    parser.add_argument("--max-concurrency", type=int, default=8, help="Max concurrent row tasks")
    parser.add_argument("--retries", type=int, default=3, help="Retries per agent call")
    parser.add_argument("--limit", type=int, default=None, help="Optional: limit number of rows for a quick run")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument("--no-weave", action="store_true", help="Disable Weave tracing even if available")
    parser.add_argument("--no-env", action="store_true", help="Do not load .env automatically")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    return asyncio.run(amain(args))


if __name__ == "__main__":
    sys.exit(main())
