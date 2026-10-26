"""Anthropic Messages API chat loop used by the standalone agent.

Uses the native Anthropic SDK with adaptive thinking — separation and
transcription are long, ordered pipelines, and letting Claude reason about
which stage can run next is worth the latency. Tools are authored once in
core.tools with Anthropic-style `input_schema` and dispatched via
core.tools.dispatch.
"""
from __future__ import annotations

import os
from typing import Callable

import anthropic

from config import MODEL


def make_client() -> anthropic.Anthropic:
    """Build an Anthropic client, or raise a clear error if unconfigured."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError(
            "No Anthropic API key — add ANTHROPIC_API_KEY=sk-ant-... to .env")
    return anthropic.Anthropic(api_key=key)


def run_loop(client: anthropic.Anthropic, tools: list,
             dispatch_fn: Callable[[str, dict], str],
             system: str, messages: list, log: Callable[[str], None],
             max_tokens: int = 16000) -> None:
    """Drive the model until it stops calling tools. `messages` is mutated in
    place so multi-turn history carries across calls. `dispatch_fn(name, args)`
    runs a tool and returns its text result."""
    anthropic_tools = [
        {
            "name": t["name"],
            "description": t.get("description", ""),
            "input_schema": t["input_schema"],
        }
        for t in tools
    ]

    while True:
        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=max_tokens,
                system=system,
                tools=anthropic_tools,
                thinking={"type": "adaptive"},
                output_config={"effort": "high"},
                messages=messages,
            )
        except anthropic.RateLimitError:
            log("Rate limited — wait a moment and try again.")
            return
        except anthropic.APIStatusError as e:
            log(f"API error {e.status_code}: {e.message}")
            return
        except anthropic.APIConnectionError:
            log("Network error reaching the API.")
            return

        if resp.stop_reason == "refusal":
            log("The model declined this request.")
            return

        for block in resp.content:
            if block.type == "text" and block.text.strip():
                log(block.text.strip())

        tool_calls = [b for b in resp.content if b.type == "tool_use"]
        if not tool_calls:
            return

        # Append the response verbatim — thinking blocks must survive round-trip.
        messages.append({"role": "assistant", "content": resp.content})

        tool_results = []
        for tc in tool_calls:
            args = tc.input if isinstance(tc.input, dict) else {}
            out = dispatch_fn(tc.name, args)
            log(f"  · {tc.name} -> {str(out)[:200]}")
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tc.id,
                "content": str(out),
            })

        messages.append({"role": "user", "content": tool_results})
