#!/bin/bash
7za a -t7z -m0=lzma -mx=9 -mfb=64 -md=32m -ms=on \
../ml_model_cube-backend.7z \
../backend \
-xr!node_modules \
-xr!__pycache__ \
-xr!.git \
-xr!cache \
-xr!unplugged \
-xr!install-state.gz \
-xr!.pnp.* \
-xr!*Zone.Identifier