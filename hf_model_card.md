---
license: mit
library_name: scikit-learn
tags:
- regression
- mlops
- cloud-infrastructure
- fastapi
---

# Cloud Node Saturation Risk Model

[Live API](https://node-saturation-api-fp6clzw5ga-od.a.run.app/docs) · [Interactive demo](https://huggingface.co/spaces/TechBsf/cloud-node-saturation-mlops-demo) · [Source code](https://github.com/TechBsf/cloud-node-saturation-mlops)

## Summary

This prototype uses a scikit-learn `LinearRegression` model to estimate a
bounded cloud-node saturation score from six infrastructure telemetry features.
The output is a regression score in `[0, 1]`, **not a calibrated probability**.

The model was trained on synthetic data. It is intended for demonstration and
MLOps practice, not production capacity planning or operational decisions.

## Model and artifact

- Selected model: `LinearRegression`
- Artifact: [`model/model.joblib`](https://huggingface.co/TechBsf/cloud-node-saturation-mlops/resolve/main/model/model.joblib)
- Framework: scikit-learn
- Training rows: 1,500 synthetic examples
- Random seed: 42
- Split: 70% train, 15% validation, 15% test

### Input features

| Feature | Meaning | Accepted range |
|---|---|---:|
| `cpu_utilization_pct` | CPU utilization | 0–100% |
| `memory_utilization_pct` | Memory utilization | 0–100% |
| `request_rate_rps` | Request rate | 0–10,000 requests/s |
| `p99_latency_ms` | P99 latency | 0–5,000 ms |
| `queue_depth` | Queue depth | 0–1,000 |
| `uptime_hours` | Uptime | 0–200,000 hours |

Supply features in the order shown above when using the serialized estimator
directly. The FastAPI endpoint accepts named fields and validates their bounds.

## Selection and evaluation

The XGBoost candidate is promoted only if its five-fold cross-validation RMSE
improves on the linear baseline by at least 1%. It did not meet that gate:

| Model | Mean five-fold CV RMSE | Decision |
|---|---:|---|
| LinearRegression | 0.029277 | Selected |
| XGBRegressor | 0.030894 | Rejected |

Held-out test metrics for the selected model:

| RMSE | MAE | R² |
|---:|---:|---:|
| 0.0304 | 0.0236 | 0.8737 |

These metrics describe performance on the synthetic test split and should not
be interpreted as evidence of real-world performance.

## Try it

- Use the [Cloud Run Swagger UI](https://node-saturation-api-fp6clzw5ga-od.a.run.app/docs)
  to send a request to the live FastAPI service.
- The [Hugging Face Space](https://huggingface.co/spaces/TechBsf/cloud-node-saturation-mlops-demo)
  runs a browser-side reproduction of the fitted linear calculation; it does
  not call the Cloud Run API.

The API also exposes `/metrics` and `/drift`. These demo diagnostics are
process-local and reset when the Cloud Run instance stops or restarts.

## Limitations

- Training and evaluation data are synthetic.
- The score is clipped to `[0, 1]` and is not probability-calibrated.
- The Cloud Run deployment is manual, scales to zero, and may incur usage or
  artifact-storage charges beyond Google Cloud free quotas.
- Monitoring is demonstrative; there is no durable metric store or alerting.

See the [GitHub repository](https://github.com/TechBsf/cloud-node-saturation-mlops)
for the training pipeline, API, tests, and deployment configuration.
