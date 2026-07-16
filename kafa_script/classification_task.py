"""Deterministic task and evidence primitives for causal classification.

This module is intentionally independent of the experiment runners and of
PyTorch.  It defines the small, auditable contract that a future generic
classification runner can consume without changing the frozen AG News code.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from string import Formatter
from typing import Any, Iterable, Mapping, Sequence


_TASK_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_SPEC_KEYS = {
    "schema_version",
    "task_id",
    "prompt_template",
    "verbalizers",
    "conflict_permutation",
    "text_field",
    "label_field",
}


def _normalize_json(value: Any) -> Any:
    """Return a strict JSON value with deterministic mapping-key order."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical JSON does not permit non-finite floats")
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("canonical JSON mapping keys must be strings")
        return {key: _normalize_json(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_normalize_json(item) for item in value]
    raise TypeError(f"value of type {type(value).__name__} is not JSON serializable")


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a value using the repository's content-addressing convention."""
    normalized = _normalize_json(value)
    return json.dumps(normalized, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":")).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def canonical_row(row: Mapping[str, Any], *, id_field: str = "_id") -> dict[str, Any]:
    """Normalize a source row while excluding a derived content-ID field."""
    if not isinstance(row, Mapping):
        raise TypeError("row must be a mapping")
    return _normalize_json({key: value for key, value in row.items()
                            if key != id_field})


def canonical_row_id(row: Mapping[str, Any], *, id_field: str = "_id") -> str:
    return canonical_sha256(canonical_row(row, id_field=id_field))


def canonical_rows_sha256(rows: Iterable[Mapping[str, Any]], *,
                          id_field: str = "_id") -> str:
    """Hash an ordered sequence of canonical source rows."""
    return canonical_sha256([canonical_row(row, id_field=id_field)
                             for row in rows])


@dataclass(frozen=True)
class ClassificationTaskSpec:
    """Canonical class ordering and prompt contract shared by all models."""

    schema_version: str
    task_id: str
    prompt_template: str
    verbalizers: tuple[str, ...]
    conflict_permutation: tuple[int, ...]
    text_field: str = "text"
    label_field: str = "label"

    def __post_init__(self) -> None:
        if self.schema_version != "1.0":
            raise ValueError("unsupported task-spec schema_version; expected '1.0'")
        if not _TASK_ID.fullmatch(self.task_id):
            raise ValueError("task_id must match [a-z0-9][a-z0-9._-]*")
        if len(self.verbalizers) < 2:
            raise ValueError("classification requires at least two verbalizers")
        if any(not isinstance(value, str) or not value.strip()
               for value in self.verbalizers):
            raise ValueError("verbalizers must be non-empty strings")
        if len(set(self.verbalizers)) != len(self.verbalizers):
            raise ValueError("verbalizers must be unique in canonical class order")
        fields = [field_name for _, field_name, _, _ in
                  Formatter().parse(self.prompt_template) if field_name is not None]
        if fields != ["text"]:
            raise ValueError("prompt_template must contain exactly one {text} field")
        if not self.text_field or not self.label_field:
            raise ValueError("text_field and label_field must be non-empty")
        if self.text_field == self.label_field:
            raise ValueError("text_field and label_field must differ")
        validate_conflict_permutation(self.conflict_permutation,
                                      len(self.verbalizers))

    @property
    def num_classes(self) -> int:
        return len(self.verbalizers)

    @property
    def normalized(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "prompt_template": self.prompt_template,
            "verbalizers": list(self.verbalizers),
            "conflict_permutation": list(self.conflict_permutation),
            "text_field": self.text_field,
            "label_field": self.label_field,
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.normalized)

    def prompt(self, text: str) -> str:
        if not isinstance(text, str):
            raise TypeError("classification text must be a string")
        return self.prompt_template.format(text=text)

    def verbalizer(self, label: int) -> str:
        label = _validated_label(label, self.num_classes)
        return self.verbalizers[label]

    def conflicting_label(self, label: int) -> int:
        label = _validated_label(label, self.num_classes)
        return self.conflict_permutation[label]


def validate_conflict_permutation(permutation: Sequence[int],
                                  num_classes: int) -> tuple[int, ...]:
    """Validate a deterministic label corruption as a fixed-point-free bijection."""
    if isinstance(permutation, (str, bytes)):
        raise TypeError("conflict_permutation must be a sequence of integers")
    values = tuple(permutation)
    if len(values) != num_classes:
        raise ValueError("conflict_permutation length must equal num_classes")
    if any(isinstance(value, bool) or not isinstance(value, int)
           for value in values):
        raise TypeError("conflict_permutation entries must be integers")
    if set(values) != set(range(num_classes)):
        raise ValueError("conflict_permutation must be a bijection over class IDs")
    if any(index == value for index, value in enumerate(values)):
        raise ValueError("conflict_permutation must have no fixed points")
    return values


def load_task_spec(source: Path | str | Mapping[str, Any]) -> ClassificationTaskSpec:
    """Load and normalize a strict task specification from JSON or a mapping."""
    if isinstance(source, Mapping):
        raw = dict(source)
    else:
        path = Path(source)
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise TypeError("task-spec JSON root must be an object")
    unknown = set(raw) - _SPEC_KEYS
    missing = {"schema_version", "task_id", "prompt_template", "verbalizers",
               "conflict_permutation"} - set(raw)
    if unknown:
        raise ValueError(f"unknown task-spec fields: {sorted(unknown)}")
    if missing:
        raise ValueError(f"missing task-spec fields: {sorted(missing)}")
    verbalizers = raw["verbalizers"]
    permutation = raw["conflict_permutation"]
    if isinstance(verbalizers, (str, bytes)) or not isinstance(verbalizers, Sequence):
        raise TypeError("verbalizers must be a sequence")
    if isinstance(permutation, (str, bytes)) or not isinstance(permutation, Sequence):
        raise TypeError("conflict_permutation must be a sequence")
    return ClassificationTaskSpec(
        schema_version=raw["schema_version"],
        task_id=raw["task_id"],
        prompt_template=raw["prompt_template"],
        verbalizers=tuple(verbalizers),
        conflict_permutation=tuple(permutation),
        text_field=raw.get("text_field", "text"),
        label_field=raw.get("label_field", "label"),
    )


def _validated_label(value: Any, num_classes: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("class labels must be integers")
    if not 0 <= value < num_classes:
        raise ValueError(f"class label {value} is outside [0, {num_classes})")
    return value


def balanced_split(rows: Sequence[Mapping[str, Any]], spec: ClassificationTaskSpec,
                   split_quotas: Mapping[str, int], namespace: str) \
        -> dict[str, list[Mapping[str, Any]]]:
    """Allocate deterministic, balanced, content-disjoint splits.

    ``split_quotas`` maps each split name to its required examples per class.
    Mapping insertion order defines allocation priority and is included in the
    deterministic contract.
    """
    if not namespace:
        raise ValueError("namespace must be non-empty")
    if not split_quotas:
        raise ValueError("at least one split quota is required")
    if len(set(split_quotas)) != len(split_quotas):
        raise ValueError("split names must be unique")
    for name, quota in split_quotas.items():
        if not isinstance(name, str) or not name:
            raise ValueError("split names must be non-empty strings")
        if isinstance(quota, bool) or not isinstance(quota, int) or quota < 0:
            raise ValueError("split quotas must be non-negative integers")

    grouped: dict[int, list[tuple[str, str, Mapping[str, Any]]]] = {
        label: [] for label in range(spec.num_classes)
    }
    observed_ids: set[str] = set()
    for row in rows:
        if spec.label_field not in row:
            raise KeyError(f"row is missing label field {spec.label_field!r}")
        if spec.text_field not in row:
            raise KeyError(f"row is missing text field {spec.text_field!r}")
        label = _validated_label(row[spec.label_field], spec.num_classes)
        row_id = canonical_row_id(row)
        if row_id in observed_ids:
            raise ValueError(f"duplicate canonical row ID: {row_id}")
        observed_ids.add(row_id)
        rank = hashlib.sha256(
            f"{namespace}|{row_id}".encode("utf-8")).hexdigest()
        grouped[label].append((rank, row_id, row))

    total_per_class = sum(split_quotas.values())
    for label, candidates in grouped.items():
        if len(candidates) < total_per_class:
            raise ValueError(
                f"class {label} has {len(candidates)} rows; "
                f"{total_per_class} required")
        candidates.sort(key=lambda item: (item[0], item[1]))

    result = {name: [] for name in split_quotas}
    cursors = {label: 0 for label in range(spec.num_classes)}
    for name, quota in split_quotas.items():
        for label in range(spec.num_classes):
            begin = cursors[label]
            result[name].extend(item[2] for item in
                                grouped[label][begin:begin + quota])
            cursors[label] += quota
        result[name].sort(key=canonical_row_id)
    assert_disjoint_splits(result)
    return result


def assert_disjoint_splits(splits: Mapping[str, Iterable[Mapping[str, Any]]]) -> None:
    """Raise when any canonical source row occurs in more than one split."""
    owners: dict[str, str] = {}
    for split_name, rows in splits.items():
        local: set[str] = set()
        for row in rows:
            row_id = canonical_row_id(row)
            if row_id in local:
                raise ValueError(f"duplicate row within split {split_name!r}: {row_id}")
            local.add(row_id)
            if row_id in owners:
                raise ValueError(
                    f"split overlap for {row_id}: {owners[row_id]!r} and "
                    f"{split_name!r}")
            owners[row_id] = split_name


_MIXTURE_FIELDS = (
    "kind",
    "source_id",
    "original_label",
    "training_label",
    "answer",
)


def mixture_assignment_sha256(items: Sequence[Mapping[str, Any]]) -> str:
    """Hash ordered contamination assignments without delimiter ambiguity."""
    records = []
    for item in items:
        if item.get("kind") not in {"clean", "conflict", "off_task"}:
            raise ValueError("mixture kind must be clean, conflict, or off_task")
        if not isinstance(item.get("source_id"), str) or not item["source_id"]:
            raise ValueError("mixture items require a non-empty source_id")
        record = {field: item[field] for field in _MIXTURE_FIELDS if field in item}
        records.append(record)
    return canonical_sha256(records)


def _probability_rows(probabilities: Any) -> list[list[float]]:
    if hasattr(probabilities, "tolist"):
        probabilities = probabilities.tolist()
    if isinstance(probabilities, (str, bytes)) or not isinstance(probabilities, Sequence):
        raise TypeError("probabilities must be a rank-2 sequence")
    rows = []
    for row in probabilities:
        if isinstance(row, (str, bytes)) or not isinstance(row, Sequence):
            raise TypeError("probabilities must be a rank-2 sequence")
        converted = []
        for value in row:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("probability entries must be real numbers")
            converted.append(float(value))
        rows.append(converted)
    return rows


def canonical_probability_payload(probabilities: Any) -> bytes:
    """Encode a matrix as shape-prefixed little-endian IEEE-754 float64."""
    rows = _probability_rows(probabilities)
    columns = len(rows[0]) if rows else 0
    if any(len(row) != columns for row in rows):
        raise ValueError("probability matrix rows must have equal length")
    header = b"KAFA-PROB-V1\0" + struct.pack("<QQ", len(rows), columns)
    return header + b"".join(struct.pack("<d", value)
                              for row in rows for value in row)


def audit_probability_payload(probabilities: Any, *, expected_rows: int | None = None,
                              num_classes: int | None = None,
                              tolerance: float = 1e-6) -> dict[str, Any]:
    """Validate a canonical probability matrix and return auditable metadata."""
    if tolerance < 0 or not math.isfinite(tolerance):
        raise ValueError("tolerance must be a finite non-negative number")
    rows = _probability_rows(probabilities)
    if expected_rows is not None and len(rows) != expected_rows:
        raise ValueError(f"expected {expected_rows} probability rows, got {len(rows)}")
    if not rows:
        raise ValueError("probability payload must contain at least one row")
    columns = len(rows[0])
    if columns < 2:
        raise ValueError("probability payload must contain at least two classes")
    if num_classes is not None and columns != num_classes:
        raise ValueError(f"expected {num_classes} classes, got {columns}")
    if any(len(row) != columns for row in rows):
        raise ValueError("probability matrix rows must have equal length")
    flat = [value for row in rows for value in row]
    if any(not math.isfinite(value) for value in flat):
        raise ValueError("probability payload contains non-finite values")
    if any(value < -tolerance or value > 1.0 + tolerance for value in flat):
        raise ValueError("probability payload contains values outside [0, 1]")
    errors = [abs(sum(row) - 1.0) for row in rows]
    if max(errors) > tolerance:
        raise ValueError("probability rows do not sum to one within tolerance")
    payload = canonical_probability_payload(rows)
    return {
        "encoding": "shape-prefixed little-endian-float64-v1",
        "shape": [len(rows), columns],
        "minimum": min(flat),
        "maximum": max(flat),
        "max_row_sum_error": max(errors),
        "payload_bytes": len(payload),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
    }
