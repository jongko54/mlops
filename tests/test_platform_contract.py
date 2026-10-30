import numpy as np
import pytest

from mlops_lab.platform_lab import feature_matrix


def records():
    return {
        "sample_id": [2, 1],
        **{f"pixel_{i}": [0.25, 0.75] for i in range(64)},
    }


def test_features_align_to_entity_order_and_have_exact_model_shape():
    result = feature_matrix(records(), [1, 2])
    assert result.shape == (2, 64)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result[0], np.full(64, 0.75))
    np.testing.assert_array_equal(result[1], np.full(64, 0.25))


@pytest.mark.parametrize("case", ["missing", "null", "range", "duplicate", "unknown"])
def test_invalid_features_fail_before_inference(case):
    rows = records()
    if case == "missing":
        del rows["pixel_63"]
    elif case == "null":
        rows["pixel_0"][0] = None
    elif case == "range":
        rows["pixel_0"][0] = 1.1
    elif case == "duplicate":
        rows["sample_id"] = [1, 1]
    else:
        rows["sample_id"] = [1, 3]
    with pytest.raises(ValueError):
        feature_matrix(rows, [1, 2])
