#!/usr/bin/env bash

echo "Worker Initiated"
ln -sf /runpod-volume /models
python -u /handle.py