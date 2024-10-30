#!/bin/bash
sudo docker run --runtime nvidia -it --rm --network=host --ipc=host --gpus all \
  -v ~/repo/signal-masters:/workspace \
  -w /workspace \
  --device=/dev/video0 \
  --device=/dev/ttyACM0 \
  signal-masters-image
