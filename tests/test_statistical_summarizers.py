from pathlib import Path

from kafa_script.summarize_agnews_coteaching import paired_greater_p as ag_p
from kafa_script.summarize_agnews_post_observation_replication import (
    EXPECTED_PROTOCOL_SHA256,
    sha256,
)
from kafa_script.summarize_hausa_natural_noise import paired_greater_p as natural_p


def test_degenerate_positive_differences_do_not_claim_p_zero():
    baseline = [0.1, 0.2, 0.3]
    candidate = [0.2, 0.3, 0.4]
    assert ag_p(candidate, baseline) == 1.0
    assert natural_p(candidate, baseline) == 1.0


def test_degenerate_nonpositive_differences_are_conservative():
    baseline = [0.2, 0.3, 0.4]
    candidate = [0.1, 0.2, 0.3]
    assert ag_p(candidate, baseline) == 1.0
    assert natural_p(candidate, baseline) == 1.0


def test_nondegenerate_improvement_test_is_finite():
    baseline = [0.1, 0.2, 0.3]
    candidate = [0.12, 0.25, 0.39]
    for function in (ag_p, natural_p):
        value = function(candidate, baseline)
        assert 0.0 < value < 1.0


def test_replication_protocol_still_matches_pre_run_digest():
    root = Path(__file__).resolve().parents[1]
    protocol = root / "kafa_script/agnews_post_observation_seed_replication_protocol_v1.json"
    assert sha256(protocol) == EXPECTED_PROTOCOL_SHA256
