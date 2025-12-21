# Phase 0 - Week 4: Project Templates

## Overview

Complete, production-ready project templates for common CUDA development scenarios.

| Day | Template | Use Case |
|-----|----------|----------|
| 1 | Single-File Project | Quick experiments |
| 2 | Library Project | Reusable CUDA libraries |
| 3 | Application Project | Full CUDA applications |
| 4 | Benchmark Project | Performance testing |
| 5 | Test Framework | Correctness testing |
| 6 | Complete Template | All-in-one starter |

## Prerequisites

- Weeks 1-3 complete (CMake, debugging, profiling)
- Understanding of project organization

## Using Templates

Each template is a complete, self-contained project:

```bash
# Copy template to your project
cp -r dayN-template-name ~/my-new-project

# Build and run
cd ~/my-new-project
./build.sh
```

## Template Features

### All Templates Include
- ✅ CMake build system
- ✅ CUDA error checking
- ✅ Multi-architecture support
- ✅ Debug/Release configurations
- ✅ README with usage instructions

### Day 6 Complete Template Also Includes
- ✅ Unit testing framework
- ✅ Benchmark harness
- ✅ Documentation structure
- ✅ CI/CD configuration
- ✅ Code formatting (clang-format)

## Directory Structure

```
week04/
├── README.md
├── day1-single-file/
├── day2-library/
├── day3-application/
├── day4-benchmark/
├── day5-testing/
└── day6-complete/
```

## Quick Reference

| Need | Template |
|------|----------|
| Quick test | day1-single-file |
| Reusable code | day2-library |
| Full app | day3-application |
| Performance comparison | day4-benchmark |
| Correctness testing | day5-testing |
| New serious project | day6-complete |
