# Advanced Nsight Systems Analysis

> **Master advanced profiling techniques for production-level performance engineering**

This module covers advanced Nsight Systems features often missed in basic tutorials but essential for NESAP/HPC work.

---

## 🎯 Learning Objectives

After completing these exercises, you will be able to:
- Correlate Python code with GPU activity using backtraces
- Profile I/O and data pipeline bottlenecks
- Perform scriptable analysis using `nsys stats` CLI
- Export to SQLite for custom analysis
- Use CPU sampling to find CPU-side bottlenecks
- Leverage OS runtime tracing for system-level insights
- Create comparison reports for optimization validation
- Use Nsight's expert systems for auto-analysis

---

## 📚 Exercises

| Exercise | Topic | Difficulty | Time |
|----------|-------|------------|------|
| [ex01-python-backtrace](ex01-python-backtrace/) | Python stack correlation | ⭐⭐⭐ | 1.5 hr |
| [ex02-io-dataloader](ex02-io-dataloader/) | Data pipeline profiling | ⭐⭐⭐ | 1.5 hr |
| [ex03-nsys-stats-cli](ex03-nsys-stats-cli/) | CLI-based analysis | ⭐⭐⭐ | 1 hr |
| [ex04-sqlite-analysis](ex04-sqlite-analysis/) | Custom scriptable analysis | ⭐⭐⭐⭐ | 2 hr |
| [ex05-cpu-sampling](ex05-cpu-sampling/) | CPU bottleneck detection | ⭐⭐⭐ | 1.5 hr |
| [ex06-osrt-tracing](ex06-osrt-tracing/) | OS runtime & syscalls | ⭐⭐⭐ | 1 hr |
| [ex07-comparison-reports](ex07-comparison-reports/) | Before/after analysis | ⭐⭐⭐ | 1 hr |
| [ex08-expert-systems](ex08-expert-systems/) | Auto-analysis & rules | ⭐⭐⭐⭐ | 1.5 hr |

---

## 🔧 Prerequisites

- Completed basic Nsight Systems exercises (01-nsight-systems/)
- Nsight Systems 2023.3+ (for all features)
- Python 3.8+ with PyTorch

```bash
# Verify Nsight Systems version
nsys --version

# Should be 2023.3 or newer for all features
```

---

## 🔑 Quick Reference

### Profile with All Traces
```bash
nsys profile \
    --trace=cuda,nvtx,osrt,cudnn,cublas \
    --python-backtrace=cuda \
    --python-sampling=true \
    --sample=cpu \
    --cpuctxsw=process-tree \
    -o full_profile \
    python train.py
```

### Common Analysis Commands
```bash
# Summary statistics
nsys stats report.nsys-rep

# Export to SQLite
nsys export --type=sqlite -o report.sqlite report.nsys-rep

# Compare two reports
nsys compare report1.nsys-rep report2.nsys-rep

# Generate rules-based analysis
nsys analyze report.nsys-rep
```
