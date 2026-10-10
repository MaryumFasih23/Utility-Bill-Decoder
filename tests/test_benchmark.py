"""
test_benchmark.py - Unit tests for benchmark calculation logic.
"""

from benchmark import evaluate_level1, KESC_0008_GROUND_TRUTH


def test_perfect_benchmark():
    result = evaluate_level1(KESC_0008_GROUND_TRUTH, KESC_0008_GROUND_TRUTH)
    assert result["accuracy_percentage"] == 100.0
    assert result["earned_points"] == result["total_points"]
    assert len(result["mismatches"]) == 0
    assert len(result["charges_diffs"]) == 0
    assert len(result["taxes_diffs"]) == 0


def test_perturbed_benchmark():
    perturbed = dict(KESC_0008_GROUND_TRUTH)
    perturbed["units_consumed"] = 999
    perturbed["charges"] = list(KESC_0008_GROUND_TRUTH["charges"])[:-1]
    
    result = evaluate_level1(perturbed, KESC_0008_GROUND_TRUTH)
    assert result["accuracy_percentage"] < 100.0
    assert any(m["field"] == "units_consumed" for m in result["mismatches"])
    assert len(result["charges_diffs"]) > 0
