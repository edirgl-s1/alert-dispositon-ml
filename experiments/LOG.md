# Experiment log

<!-- date | run_name | config | change vs baseline | key metrics | decision -->
2026-11-16 | baseline | configs/default.yaml | Target changed from 3-class TP/FP/benign to binary needs_investigation; training uses FeaturePipeline (E3) on OCSF-shaped synthetic payloads; metric is auto-dispose rate at <=1% missed investigations (threshold fit on val). Splits remain chronological. | synthetic only, not comparable to earlier runs | new baseline
