#!/bin/bash
cd /Users/sanjeevadodlapati/Downloads/Repos/cuda-lab/cuda-programming-guide

# Define the document order
declare -a docs=(
  "01-introduction/cuda-platform.md"
  "01-introduction/programming-model.md"
  "01-introduction/hardware-implementation.md"
  "02-basics/intro-to-cuda-cpp.md"
  "02-basics/writing-cuda-kernels.md"
  "02-basics/cuda-memory.md"
  "02-basics/understanding-memory.md"
  "02-basics/asynchronous-execution.md"
  "02-basics/nvcc.md"
  "03-advanced/advanced-kernel-programming.md"
  "03-advanced/advanced-host-programming.md"
  "03-advanced/device-memory-access.md"
  "03-advanced/performance-optimization.md"
  "03-advanced/driver-api.md"
  "04-special-topics/unified-memory.md"
  "04-special-topics/multi-gpu-programming.md"
  "04-special-topics/cuda-graphs.md"
  "04-special-topics/dynamic-parallelism.md"
  "04-special-topics/cooperative-groups.md"
  "04-special-topics/stream-ordered-memory-allocation.md"
  "04-special-topics/virtual-memory-management.md"
  "04-special-topics/inter-process-communication.md"
  "04-special-topics/programmatic-dependent-launch.md"
  "04-special-topics/error-log-management.md"
  "04-special-topics/mig.md"
  "05-appendices/cpp-language-extensions.md"
  "05-appendices/compute-capabilities.md"
  "05-appendices/warp-matrix-functions.md"
  "05-appendices/texture-fetch.md"
  "05-appendices/environment-variables.md"
)

# Get the length
len=${#docs[@]}

for i in "${!docs[@]}"; do
  file="${docs[$i]}"
  
  # Calculate prev/next
  if [ $i -eq 0 ]; then
    prev=""
  else
    prev="${docs[$((i-1))]}"
  fi
  
  if [ $i -eq $((len-1)) ]; then
    next=""
  else
    next="${docs[$((i+1))]}"
  fi
  
  # Get relative path back to index
  depth=$(echo "$file" | tr -cd '/' | wc -c)
  home_path="../index.md"
  
  # Build navigation
  nav="\n\n---\n\n"
  
  if [ -n "$prev" ]; then
    # Get relative path
    prev_name=$(basename "$prev" .md | sed 's/-/ /g' | sed 's/\b\(.\)/\u\1/g')
    prev_dir=$(dirname "$prev")
    curr_dir=$(dirname "$file")
    if [ "$prev_dir" == "$curr_dir" ]; then
      prev_link=$(basename "$prev")
    else
      prev_link="../$prev"
    fi
    nav+="[← Previous: ${prev_name}](${prev_link})"
  fi
  
  nav+=" | [🏠 Home](${home_path})"
  
  if [ -n "$next" ]; then
    next_name=$(basename "$next" .md | sed 's/-/ /g' | sed 's/\b\(.\)/\u\1/g')
    next_dir=$(dirname "$next")
    curr_dir=$(dirname "$file")
    if [ "$next_dir" == "$curr_dir" ]; then
      next_link=$(basename "$next")
    else
      next_link="../$next"
    fi
    nav+=" | [Next: ${next_name} →](${next_link})"
  fi
  
  # Append navigation to file
  echo -e "$nav" >> "$file"
  echo "Added nav to: $file"
done
