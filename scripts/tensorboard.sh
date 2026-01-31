#!/usr/bin/env bash
uv run tensorboard \
  --logdir runs \
  --host 0.0.0.0 \
  --port 6006