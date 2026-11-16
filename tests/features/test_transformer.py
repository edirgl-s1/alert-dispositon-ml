import dataclasses
import json

import pandas as pd
import pytest
from sklearn.base import clone

from alert_disposition.features.config import ChannelConfig, FeatureConfig
from alert_disposition.features.transformer import AlertFeatureTransformer
from tests.features.fixtures import ocsf_alert

CFG = dataclasses.replace(
    FeatureConfig.for_experiment("E3"),
    ocsf_full_path_key="path",
    path=ChannelConfig(min_df=2),
    command=ChannelConfig(min_df=2),
    text=ChannelConfig(min_df=2),
)


def _frame(**overrides):
    return pd.DataFrame(
        {"alert": [json.dumps(ocsf_alert(user=f"u{i}", **overrides)) for i in range(4)]}
    )


def test_json_strings_fit_transform_and_clone():
    t = AlertFeatureTransformer(CFG)
    X = t.fit_transform(_frame())
    assert X.shape[0] == 4
    assert X.shape[1] == len(t.get_feature_names_out())
    assert clone(t).get_params()["source_format"] == "ocsf"


def test_excluded_fields_do_not_change_features():
    a = AlertFeatureTransformer(CFG).fit(_frame())
    b = _frame(status="new", verdict_id=9, assignee={"email_addr": "x@y.test"})
    assert (a.transform(_frame()) != a.transform(b)).nnz == 0


def test_malformed_json_error_names_row_not_content():
    t = AlertFeatureTransformer(CFG)
    with pytest.raises(ValueError, match="row 0") as exc:
        t.fit(pd.DataFrame({"alert": ["{secret-not-json"]}))
    assert "secret" not in str(exc.value)
