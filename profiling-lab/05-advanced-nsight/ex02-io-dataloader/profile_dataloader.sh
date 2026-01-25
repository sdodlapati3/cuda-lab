#!/bin/bash
# Profile DataLoader performance

MODE=${1:-slow}

if [ "$MODE" = "slow" ]; then
    echo "Profiling SLOW DataLoader..."
    nsys profile \
        --trace=cuda,nvtx,osrt \
        --osrt-events=file_io \
        --sample=cpu \
        --force-overwrite=true \
        -o slow_profile \
        python slow_dataloader.py
    
    echo ""
    echo "Profile saved to slow_profile.nsys-rep"
    
elif [ "$MODE" = "fast" ]; then
    echo "Profiling FAST DataLoader..."
    nsys profile \
        --trace=cuda,nvtx,osrt \
        --sample=cpu \
        --force-overwrite=true \
        -o fast_profile \
        python fast_dataloader.py
    
    echo ""
    echo "Profile saved to fast_profile.nsys-rep"

elif [ "$MODE" = "compare" ]; then
    echo "Comparing profiles..."
    nsys stats slow_profile.nsys-rep > slow_stats.txt
    nsys stats fast_profile.nsys-rep > fast_stats.txt
    echo "Stats saved to slow_stats.txt and fast_stats.txt"
    
else
    echo "Usage: $0 [slow|fast|compare]"
    exit 1
fi
