#!/bin/bash
# Profile without Python backtraces (baseline)

nsys profile \
    --trace=cuda,nvtx \
    --force-overwrite=true \
    -o basic_profile \
    python train_model.py --epochs 2

echo ""
echo "Profile complete. View with: nsys-ui basic_profile.nsys-rep"
echo "Or get stats: nsys stats basic_profile.nsys-rep"
