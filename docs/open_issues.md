# Open issues (TBD)

Known gaps we chose to defer. Revisit when real, labelled data is available.

## 1. The missed-investigation cap is a point estimate, not a guarantee
`select_threshold` (`evaluation/metrics.py`) picks the auto-dispose threshold so that *val*
misses at most `max_missed_investigation_rate` (default 1%) of should-investigate alerts. That
holds on val only. On new data the miss rate is a random variable: on synthetic data, test came
out at 2.65% against a 1% target. With a few hundred positives in val this is normal sampling
noise.

For an auto-dispose system the claim should be "at most X% missed". Options:
- choose the threshold against an upper confidence bound on the miss rate (e.g. Clopper-Pearson)
  instead of the empirical rate; costs some auto-dispose rate;
- report a confidence interval for test `missed_investigation_rate` next to the point value;
- make sure val has enough positives for the bound to be meaningful.

## 2. One bad row aborts the whole batch
A missing payload (NaN in a CSV) or malformed JSON raises in `AlertFeatureTransformer`
(`features/transformer.py`), failing the entire training or evaluation run. Errors name the row,
never the content.

Decision pending: keep failing loudly for training (current) vs. skip-and-count bad rows. For
inference the intended behaviour is per-alert: a payload that cannot be featurized returns
**no ML verdict** and the alert continues to the LLM stage. That needs an inference entry point,
which does not exist yet.

## 3. IPv4 redaction also matches version strings
See "Known issue (deferred)" in `docs/feature_extraction.md`.
