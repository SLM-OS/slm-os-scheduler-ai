#!/usr/bin/env python3
"""Export trained models to ONNX and C deployment formats.

MLP/RL -> ONNX -> INT8 quantization -> C weight arrays.
XGBoost -> treelite -> C predictor.
See plan Section 9.2.
"""

from __future__ import annotations

# TODO: Implement model export per Section 9.2
