"""Generic schema decoding, prompt rendering, and record validation helpers."""

from __future__ import annotations

from typing import Any

from .types import Record, SchemaDefinition


def prompt_safe_schema(
    schema: SchemaDefinition,
    fields: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    prompt_schema: dict[str, Any] = {}
    field_names = fields or tuple(schema.keys())
    for field in field_names:
        config = schema[field]
        prompt_config: dict[str, Any] = {}
        if "values" in config:
            prompt_config["values"] = list(config["values"])
        if "parents" in config:
            prompt_config["parents"] = list(config["parents"])
        if "mapping" in config:
            mapping = config["mapping"]
            parents = config.get("parents", [])
            if parents and any(isinstance(key, tuple) for key in mapping):
                prompt_config["mapping"] = [
                    {
                        "parents": {parents[index]: key[index] for index in range(len(parents))},
                        "values": list(values),
                    }
                    for key, values in mapping.items()
                ]
            else:
                prompt_config["mapping"] = {
                    str(key): list(values) for key, values in mapping.items()
                }
        prompt_schema[field] = prompt_config
    return prompt_schema


def declared_field_values(schema: SchemaDefinition, field: str) -> list[str]:
    config = schema[field]
    if "values" in config:
        return list(config["values"])
    values: list[str] = []
    seen: set[str] = set()
    for mapping_values in config["mapping"].values():
        for value in mapping_values:
            if value not in seen:
                seen.add(value)
                values.append(value)
    return values


def validate_record_against_schema(schema: SchemaDefinition, record: Record) -> None:
    expected_fields = set(schema.keys())
    if set(record.keys()) != expected_fields:
        missing = sorted(expected_fields - set(record.keys()))
        extra = sorted(set(record.keys()) - expected_fields)
        raise ValueError(f"Record keys do not match schema. Missing={missing}, extra={extra}")

    for field, config in schema.items():
        value = record[field]
        if "values" in config and value not in config["values"]:
            raise ValueError(f"Invalid value for {field}: {value}")
        if "mapping" in config:
            parents = config["parents"]
            if len(parents) == 1:
                parent_value = record[parents[0]]
                allowed = config["mapping"][parent_value]
            else:
                parent_value = tuple(record[parent] for parent in parents)
                allowed = config["mapping"][parent_value]
            if value not in allowed:
                raise ValueError(
                    f"Invalid dependent value for {field}: {value} "
                    f"not in {allowed} for {parent_value}"
                )
