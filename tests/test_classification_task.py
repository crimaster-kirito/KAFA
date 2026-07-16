import json

import pytest

from kafa_script.classification_task import (
    assert_disjoint_splits,
    audit_probability_payload,
    balanced_split,
    canonical_probability_payload,
    canonical_row_id,
    canonical_rows_sha256,
    load_task_spec,
    mixture_assignment_sha256,
    validate_conflict_permutation,
)


def task_mapping(classes=2):
    labels = (["Negative", "Positive"] if classes == 2 else
              ["World", "Sports", "Business", "Science"])
    return {
        "schema_version": "1.0",
        "task_id": f"fixture-{classes}",
        "prompt_template": "Text: {text}\nLabel:",
        "verbalizers": labels,
        "conflict_permutation": list(range(1, classes)) + [0],
        "text_field": "text",
        "label_field": "label",
    }


@pytest.mark.parametrize("classes", [2, 4])
def test_load_task_spec_supports_variable_k_and_stable_hash(tmp_path, classes):
    mapping = task_mapping(classes)
    path = tmp_path / "task.json"
    path.write_text(json.dumps(mapping, indent=4), encoding="utf-8")
    from_path = load_task_spec(path)
    from_mapping = load_task_spec(dict(reversed(list(mapping.items()))))

    assert from_path.num_classes == classes
    assert from_path.verbalizer(classes - 1) == mapping["verbalizers"][-1]
    assert from_path.prompt("hello") == "Text: hello\nLabel:"
    assert from_path.sha256 == from_mapping.sha256


@pytest.mark.parametrize("permutation", [[0, 1], [1, 1], [1, 2, 0, 3]])
def test_conflict_permutation_rejects_fixed_points_or_non_bijection(permutation):
    with pytest.raises(ValueError):
        validate_conflict_permutation(permutation, len(permutation))


def test_conflict_permutation_accepts_derangements_for_k2_and_k4():
    assert validate_conflict_permutation([1, 0], 2) == (1, 0)
    assert validate_conflict_permutation([1, 2, 3, 0], 4) == (1, 2, 3, 0)


def fixture_rows(classes=4, per_class=5):
    return [
        {"text": f"class {label} row {index}", "label": label,
         "metadata": {"index": index}}
        for label in range(classes)
        for index in range(per_class)
    ]


def test_canonical_row_ids_ignore_derived_id_and_are_order_independent():
    first = {"text": "x", "label": 1, "metadata": {"b": 2, "a": 1}}
    second = {"metadata": {"a": 1, "b": 2}, "label": 1, "text": "x",
              "_id": "untrusted-derived-value"}
    assert canonical_row_id(first) == canonical_row_id(second)
    assert canonical_rows_sha256([first, second]) == canonical_rows_sha256(
        [dict(first), dict(second)])


@pytest.mark.parametrize("classes", [2, 4])
def test_balanced_split_is_deterministic_balanced_and_disjoint(classes):
    spec = load_task_spec(task_mapping(classes))
    rows = fixture_rows(classes, 5)
    quotas = {"anchor": 2, "development": 2, "client": 1}
    first = balanced_split(rows, spec, quotas, "unit-test-v1")
    second = balanced_split(list(reversed(rows)), spec, quotas, "unit-test-v1")

    for split_name, per_class in quotas.items():
        assert [canonical_row_id(row) for row in first[split_name]] == [
            canonical_row_id(row) for row in second[split_name]]
        assert {label: sum(row["label"] == label for row in first[split_name])
                for label in range(classes)} == {
                    label: per_class for label in range(classes)}
    assert_disjoint_splits(first)


def test_balanced_split_rejects_insufficient_class_quota():
    spec = load_task_spec(task_mapping(2))
    rows = fixture_rows(2, 2)
    rows.pop()  # class 1 is now short
    with pytest.raises(ValueError, match="class 1"):
        balanced_split(rows, spec, {"anchor": 1, "development": 1}, "short")


def test_assert_disjoint_splits_rejects_overlap():
    row = {"text": "same", "label": 0}
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint_splits({"anchor": [row], "development": [dict(row)]})


def test_mixture_assignment_hash_is_deterministic_and_order_sensitive():
    items = [
        {"kind": "clean", "source_id": "a", "original_label": 0,
         "training_label": 0, "answer": "Negative", "ignored": 1},
        {"kind": "conflict", "source_id": "b", "original_label": 0,
         "training_label": 1, "answer": "Positive"},
        {"kind": "off_task", "source_id": "c", "answer": "World"},
    ]
    assert mixture_assignment_sha256(items) == mixture_assignment_sha256(
        [dict(reversed(list(item.items()))) for item in items])
    assert mixture_assignment_sha256(items) != mixture_assignment_sha256(
        list(reversed(items)))


@pytest.mark.parametrize("matrix,classes", [
    ([[0.25, 0.75], [1.0, 0.0]], 2),
    ([[0.1, 0.2, 0.3, 0.4]], 4),
])
def test_probability_audit_validates_simplex_shape_and_stable_hash(matrix, classes):
    first = audit_probability_payload(matrix, expected_rows=len(matrix),
                                      num_classes=classes)
    second = audit_probability_payload(tuple(tuple(row) for row in matrix),
                                       num_classes=classes)
    assert first["shape"] == [len(matrix), classes]
    assert first["payload_sha256"] == second["payload_sha256"]
    assert first["payload_bytes"] == len(canonical_probability_payload(matrix))


@pytest.mark.parametrize("matrix,error", [
    ([[0.2, 0.2]], "sum"),
    ([[1.1, -0.1]], "outside"),
    ([[float("nan"), float("nan")]], "non-finite"),
    ([[0.5, 0.5], [1.0]], "equal length"),
])
def test_probability_audit_rejects_invalid_payloads(matrix, error):
    with pytest.raises(ValueError, match=error):
        audit_probability_payload(matrix)
