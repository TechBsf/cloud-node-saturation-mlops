---
title: Cloud Node Saturation Risk Demo
emoji: 📈
colorFrom: blue
colorTo: green
sdk: static
app_file: index.html
pinned: false
license: mit
tags:
- mlops
- regression
- cloud-infrastructure
---

# Cloud Node Saturation Risk Demo

Try sample cloud-node telemetry with a browser-side reproduction of the fitted
`LinearRegression` model. The score is clipped to `[0, 1]`; it is not a
calibrated probability. The model was trained on synthetic data, so this demo
is for exploration, not real operational decisions.

For real API inference, use the
[interactive Cloud Run API docs](https://node-saturation-api-fp6clzw5ga-od.a.run.app/docs).
The [model card and artifact](https://huggingface.co/TechBsf/cloud-node-saturation-mlops)
and [GitHub source](https://github.com/TechBsf/cloud-node-saturation-mlops)
describe how the project is trained, evaluated, and served.
