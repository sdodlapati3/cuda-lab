#!/bin/bash
# Profile with Python backtraces enabled

nsys profile \
    --trace=cuda,nvtx \
    --python-backtrace=cuda \
    --python-sampling=true \
    --python-sampling-frequency=1000 \
    --force-overwrite=true \
    -o python_profile \
    python train_model.py --epochs 2

echo ""
echo "Profile with Python backtraces complete!"
echo ""
echo "View with: nsys-ui python_profile.nsys-rep"
echo "Get stats: nsys stats python_profile.nsys-rep"
echo ""
echo "Look for kernels triggered by HeavyLayer in model.py"
