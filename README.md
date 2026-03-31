---
title: DermaScan+ Inference API
emoji: 🔬
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
license: mit
---

# DermaScan+ Inference API

FastAPI inference server for DermaScan+, a mobile skin condition detection app developed as a capstone project.

> **Disclaimer**: This is a capstone research project and is not intended for clinical or commercial use. Predictions are not a substitute for professional medical diagnosis. Always consult a licensed dermatologist for accurate skin condition assessment.

## Derm Foundation

This project uses Google's Derm Foundation as a frozen feature extractor.

- https://developers.google.com/health-ai-developer-foundations/derm-foundation
- https://huggingface.co/google/derm-foundation

## Endpoints

- `POST /analyze` — accepts raw image bytes, returns skin condition prediction and severity
- `GET /health` — health check

## Model Architecture

Two-stage pipeline using Google Derm Foundation as a frozen feature extractor with Logistic Regression classifier heads.

- Stage 1: Skin condition classification (14 classes)
- Stage 2: Severity classification (mild / moderate / severe) per condition

## Known Limitations

Some conditions may exhibit low detection confidence due to limited publicly available training data. Conditions affected include acne whiteheads and enlarged pores. These limitations are documented and acknowledged as part of the capstone research.

## File Structure

- `embedder.py` — loads Derm Foundation and generates embeddings
- `app.py` — FastAPI inference server
- `preprocessing/preprocess_image.py` — resizes and converts images to RGB 448x448 for the encoder
- `trained_data_two_stage/` — trained Logistic Regression classifier heads (.pkl files)

## License

MIT
