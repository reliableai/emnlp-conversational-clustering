"""Generic OpenRouter batch generation with structured parsing and retries."""

from __future__ import annotations

import json
import os
import random
from typing import Any, cast

from .progress import render_generation_progress, write_generation_progress
from .types import GeneratedTextBatch, OpenRouterConfig, Record, StructuredDatasetPreset


def response_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            else:
                text_attr = getattr(item, "text", None)
                if isinstance(text_attr, str):
                    parts.append(text_attr)
        combined = "".join(parts).strip()
        if combined:
            return combined
    raise TypeError("Expected OpenRouter response content to be a string or text blocks.")


def parse_generated_text_batch(
    payload: GeneratedTextBatch,
    expected_row_ids: set[int],
) -> dict[int, str]:
    parsed: dict[int, str] = {}
    for item in payload.items:
        if item.row_id in parsed:
            raise ValueError(f"Duplicate row_id in OpenRouter response: {item.row_id}")
        parsed[item.row_id] = item.text

    if set(parsed.keys()) != expected_row_ids:
        raise ValueError(
            "OpenRouter response row_ids do not match the requested batch. "
            f"expected={sorted(expected_row_ids)}, got={sorted(parsed.keys())}"
        )
    return parsed


def generated_text_response_schema() -> dict[str, Any]:
    return json.loads(json.dumps(GeneratedTextBatch.model_json_schema()))


def extract_structured_output_payload(response: object) -> GeneratedTextBatch:
    if isinstance(response, GeneratedTextBatch):
        return response
    if isinstance(response, dict):
        response_dict = cast(dict[str, object], response)
        parsing_error = response_dict.get("parsing_error")
        if parsing_error is not None:
            if isinstance(parsing_error, BaseException):
                raise ValueError("OpenRouter structured output parsing failed.") from parsing_error
            raise ValueError("OpenRouter structured output parsing failed.")
        parsed = response_dict.get("parsed")
        if isinstance(parsed, GeneratedTextBatch):
            return parsed
        raise TypeError(
            "OpenRouter structured output response did not include a parsed GeneratedTextBatch."
        )
    raise TypeError(
        "Expected OpenRouter structured output response to be "
        "a GeneratedTextBatch or response dict."
    )


def generate_texts_with_openrouter(
    preset: StructuredDatasetPreset,
    records: list[Record],
    row_payloads: list[dict[str, Any]],
    rng: random.Random,
    difficulty: str,
    *,
    config: OpenRouterConfig,
) -> list[str]:
    try:
        from dotenv import load_dotenv
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_openrouter import ChatOpenRouter
        from pydantic import SecretStr
    except ImportError as exc:
        raise RuntimeError(
            "OpenRouter text generation requires python-dotenv, langchain-core, "
            "langchain-openrouter, and pydantic."
        ) from exc

    load_dotenv()
    api_key_env = os.getenv("OPENROUTER_API_KEY")
    if not api_key_env:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")

    llm = ChatOpenRouter(
        openrouter_api_base="https://openrouter.ai/api/v1",
        openrouter_api_key=SecretStr(api_key_env),
        model=config.model,
        temperature=config.temperature,
        max_retries=0,
    )
    structured_llm = llm.with_structured_output(
        GeneratedTextBatch,
        strict=True,
        include_raw=True,
    )

    if not records:
        return []

    texts: list[str | None] = [None] * len(records)
    total_rows = len(records)
    batch_total = (total_rows + config.batch_size - 1) // config.batch_size
    completed_rows = 0
    write_generation_progress(
        render_generation_progress(completed_rows, total_rows, status="Starting generation")
    )

    for batch_number, batch_start in enumerate(range(0, len(records), config.batch_size), start=1):
        batch_end = min(batch_start + config.batch_size, len(records))
        expected_row_ids = set(range(batch_start, batch_end))
        payload_by_row_id = {
            row_id: row_payloads[row_id] for row_id in range(batch_start, batch_end)
        }
        batch_texts: dict[int, str] = {}
        pending_row_ids = set(expected_row_ids)
        prompt = preset.build_prompt(
            [payload_by_row_id[row_id] for row_id in sorted(expected_row_ids)],
            difficulty,
        )
        retry_prefix: str | None = None
        last_error: Exception | None = None
        last_parsed: dict[int, str] | None = None
        last_hard_failures: dict[int, ValueError] = {}
        last_soft_failures: dict[int, ValueError] = {}

        for attempt in range(config.max_retries + 1):
            status = "Generating" if attempt == 0 else f"Retrying ({attempt}/{config.max_retries})"
            write_generation_progress(
                render_generation_progress(
                    completed_rows,
                    total_rows,
                    status=status,
                    batch_index=batch_number,
                    batch_total=batch_total,
                )
            )
            attempt_prompt = prompt if retry_prefix is None else retry_prefix + prompt
            try:
                response = structured_llm.invoke(
                    [
                        SystemMessage(content=preset.system_prompt),
                        HumanMessage(content=attempt_prompt),
                    ]
                )
                payload = extract_structured_output_payload(response)
                parsed = parse_generated_text_batch(payload, pending_row_ids)
                last_parsed = parsed
            except Exception as exc:
                last_error = exc
                if retry_prefix is None:
                    retry_prefix = (
                        "The previous response did not match the required structured schema "
                        "exactly. Retry and return a response that matches the schema.\n\n"
                    )
                continue

            hard_failures, soft_failures = preset.collect_alignment_failures(records, parsed)
            last_hard_failures = hard_failures
            last_soft_failures = soft_failures
            failing_row_ids = set(hard_failures) | set(soft_failures)
            for row_id, text in parsed.items():
                if row_id not in failing_row_ids:
                    batch_texts[row_id] = text

            if failing_row_ids:
                failing_rows = {row_id: parsed[row_id] for row_id in sorted(failing_row_ids)}
                alignment_retry_prefix = preset.build_alignment_retry_prefix(records, failing_rows)
                if alignment_retry_prefix is None:
                    raise RuntimeError("Alignment failures were detected without retry guidance.")
                if attempt == config.max_retries:
                    last_error = ValueError(alignment_retry_prefix.strip())
                    continue
                pending_row_ids = failing_row_ids
                prompt = preset.build_prompt(
                    [payload_by_row_id[row_id] for row_id in sorted(failing_row_ids)],
                    difficulty,
                )
                last_error = ValueError(alignment_retry_prefix.strip())
                retry_prefix = alignment_retry_prefix + "\n"
                continue

            for row_id, text in batch_texts.items():
                texts[row_id] = text
            completed_rows = batch_end
            write_generation_progress(
                render_generation_progress(
                    completed_rows,
                    total_rows,
                    status="Generated",
                    batch_index=batch_number,
                    batch_total=batch_total,
                ),
                final=completed_rows == total_rows,
            )
            break
        else:
            if last_parsed is None:
                write_generation_progress(
                    render_generation_progress(
                        completed_rows,
                        total_rows,
                        status="Generation failed",
                        batch_index=batch_number,
                        batch_total=batch_total,
                    ),
                    final=True,
                )
                raise RuntimeError(
                    f"Failed to generate a valid OpenRouter response for rows "
                    f"{batch_start}-{batch_end - 1}."
                ) from last_error

            unresolved_row_ids = sorted({*last_hard_failures, *last_soft_failures})
            if unresolved_row_ids:
                for row_id in unresolved_row_ids:
                    source_text = last_parsed.get(row_id, "")
                    try:
                        batch_texts[row_id] = preset.repair_generated_text(
                            records[row_id],
                            source_text,
                        )
                    except Exception as exc:
                        regeneration_error: Exception | None = exc
                        regenerated_text: str | None = None
                        regen_prompt = preset.build_prompt([payload_by_row_id[row_id]], difficulty)
                        regen_retry_prefix = (
                            "The deterministic fallback repair failed for this row. "
                            "Regenerate only this row from metadata and return valid JSON only.\n"
                            "Keep every row_id exactly the same.\n\n"
                        )

                        for _regen_attempt in range(config.max_retries + 1):
                            regen_attempt_prompt = regen_retry_prefix + regen_prompt
                            try:
                                regen_response = structured_llm.invoke(
                                    [
                                        SystemMessage(content=preset.system_prompt),
                                        HumanMessage(content=regen_attempt_prompt),
                                    ]
                                )
                                regen_payload = extract_structured_output_payload(regen_response)
                                regen_parsed = parse_generated_text_batch(regen_payload, {row_id})
                            except Exception as regen_exc:
                                regeneration_error = regen_exc
                                regen_retry_prefix = (
                                    "The previous response did not match the required structured "
                                    "schema exactly. Retry and return a response that matches the "
                                    "schema.\n\n"
                                )
                                continue

                            regen_hard_failures, regen_soft_failures = (
                                preset.collect_alignment_failures(records, regen_parsed)
                            )
                            regen_failing_row_ids = set(regen_hard_failures) | set(
                                regen_soft_failures
                            )
                            if not regen_failing_row_ids:
                                regenerated_text = regen_parsed[row_id]
                                break

                            failing_rows = {
                                failing_row_id: regen_parsed[failing_row_id]
                                for failing_row_id in sorted(regen_failing_row_ids)
                            }
                            alignment_retry_prefix = preset.build_alignment_retry_prefix(
                                records,
                                failing_rows,
                            )
                            if alignment_retry_prefix is None:
                                regeneration_error = RuntimeError(
                                    "Alignment failures were detected without retry guidance."
                                )
                                continue
                            regeneration_error = ValueError(alignment_retry_prefix.strip())
                            regen_retry_prefix = alignment_retry_prefix + "\n"

                        if regenerated_text is None:
                            _ = regeneration_error
                            regenerated_text = preset.build_emergency_fallback_text(records[row_id])

                        batch_texts[row_id] = regenerated_text
                for row_id, text in batch_texts.items():
                    texts[row_id] = text
                completed_rows = batch_end
                write_generation_progress(
                    render_generation_progress(
                        completed_rows,
                        total_rows,
                        status="Generated with fallback",
                        batch_index=batch_number,
                        batch_total=batch_total,
                    ),
                    final=completed_rows == total_rows,
                )
                continue

            write_generation_progress(
                render_generation_progress(
                    completed_rows,
                    total_rows,
                    status="Generation failed",
                    batch_index=batch_number,
                    batch_total=batch_total,
                ),
                final=True,
            )
            raise RuntimeError(
                f"Failed to generate a valid OpenRouter response for rows "
                f"{batch_start}-{batch_end - 1}."
            ) from last_error

    final_texts = [text for text in texts if text is not None]
    if len(final_texts) == len(records):
        return final_texts
    missing_row_ids = [row_id for row_id, ticket_text in enumerate(texts) if ticket_text is None]
    raise RuntimeError(
        "OpenRouter generation completed with missing rows and cannot continue safely. "
        f"missing_row_ids={missing_row_ids}"
    )
