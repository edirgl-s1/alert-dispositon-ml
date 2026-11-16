"""Build features from synthetic alerts, serialize, reload, and fit a minimal LightGBM model.

Run from the repo root::

    uv run python -m scripts.train_lightgbm_example

Labels here are synthetic placeholders (1 = needs investigation). Real labels
must be independently adjudicated investigation-necessity labels, not TP/FP.
"""

import dataclasses
import json
import tempfile
from pathlib import Path

from tests.features.fixtures import ocsf_alert

from alert_disposition.features.adapters import normalize_alert
from alert_disposition.features.config import ChannelConfig, FeatureConfig
from alert_disposition.features.extract import extract_feature_document
from alert_disposition.features.pipeline import FeaturePipeline
from alert_disposition.features.report import coverage_report

N_ALERTS = 200
TRAIN_FRACTION = 0.7


def main() -> None:
    config = dataclasses.replace(
        FeatureConfig.for_experiment("E3"),
        ocsf_full_path_key="path",
        path=ChannelConfig(min_df=5),
        command=ChannelConfig(min_df=5),
        text=ChannelConfig(min_df=5, max_features=1000),
    )
    payloads = [
        ocsf_alert(
            user=f"user{i}",
            n_evidence=i % 4,
            finding_info={"title": "Suspicious PowerShell" if i % 2 else "Scheduled task created"},
        )
        for i in range(N_ALERTS)
    ]
    labels = [i % 2 for i in range(N_ALERTS)]
    alerts = [normalize_alert(p, "ocsf", config=config) for p in payloads]
    docs = [extract_feature_document(a, config) for a in alerts]

    split = int(N_ALERTS * TRAIN_FRACTION)
    pipeline = FeaturePipeline(config)
    x_train = pipeline.fit_transform(docs[:split])
    x_test = pipeline.transform(docs[split:])
    print("train", x_train.shape, "test", x_test.shape)
    print("first features:", pipeline.get_feature_names_out()[:8])

    with tempfile.TemporaryDirectory() as tmp:
        artifact = Path(tmp) / "pipeline.joblib"
        pipeline.save(artifact, model_version_ref="example-0")
        reloaded, ref = FeaturePipeline.load(artifact)
        assert reloaded.get_feature_names_out() == pipeline.get_feature_names_out()
        print("reloaded", ref)

    report = coverage_report(pipeline, docs[:split], alerts[:split], held_out_docs=docs[split:])
    print(json.dumps({k: report[k] for k in ("shape", "density", "features_by_family")}))

    from lightgbm import LGBMClassifier

    model = LGBMClassifier(n_estimators=20, min_child_samples=5, verbose=-1)
    model.fit(x_train, labels[:split])
    p_needs_investigation = model.predict_proba(x_test)[:, 1]
    print("P(needs_investigation) head:", p_needs_investigation[:5].round(3))


if __name__ == "__main__":
    main()
