from crimenet_ml.features import XGB_CORE_V1, XGB_HISTORY_V1


def test_feature_set_lengths() -> None:
    assert len(XGB_HISTORY_V1) == 40
    assert len(XGB_CORE_V1) == 77


def test_feature_sets_have_no_duplicates() -> None:
    assert len(XGB_HISTORY_V1) == len(set(XGB_HISTORY_V1))
    assert len(XGB_CORE_V1) == len(set(XGB_CORE_V1))


def test_history_is_subset_of_core() -> None:
    assert set(XGB_HISTORY_V1) <= set(XGB_CORE_V1)
