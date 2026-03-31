---
title: DermaScan+ Inference API
emoji: 🔬
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---


# DermaScan+ Inference API

FastAPI inference server for DermaScan+, a mobile skin condition detection app.

derm foundation model: 
- https://developers.google.com/health-ai-developer-foundations/derm-foundation
- https://huggingface.co/google/derm-foundation


## Endpoints

- `POST /analyze` — accepts raw image bytes, returns skin condition prediction and severity
- `GET /health` — health check

## Model Architecture

Two-stage pipeline using Google Derm Foundation as a frozen feature extractor with Logistic Regression classifier heads.
- Stage 1: Skin condition classification (14 classes)
- Stage 2: Severity classification (mild / moderate / severe) per condition

## Notes

This API is intended for use with the DermaScan+ mobile application only.
