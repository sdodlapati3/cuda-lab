# CUDA Curriculum Enhancement Plan

> **Created:** December 19, 2025  
> **Status:** Implementation In Progress  
> **Goal:** Make the 12-week curriculum comprehensive and blog-ready

---

## 📋 Executive Summary

This document outlines enhancements to the CUDA learning curriculum to:
1. Fill content gaps in existing weeks
2. Add missing checkpoint quizzes
3. Extend curriculum to Week 13-14 for modern GPU features
4. Create blog post infrastructure for content publishing

---

## Phase 1: Fix Existing Content Gaps

### 1.1 Missing Checkpoint Quizzes

| Week | Quiz Status | Action |
|------|-------------|--------|
| Week 1 | ✅ Exists | None |
| Week 2 | ✅ Exists | None |
| Week 3 | ❌ Missing | Create `checkpoint-quiz.md` |
| Week 4 | ❌ Missing | Create `checkpoint-quiz.md` |
| Week 5 | ❌ Missing | Create `checkpoint-quiz.md` |
| Week 6 | ❌ Missing | Create `checkpoint-quiz.md` |
| Week 7 | ❌ Missing | Create `checkpoint-quiz.md` |
| Week 8 | ❌ Missing | Create `checkpoint-quiz.md` |
| Week 9 | ❌ Missing | Create `checkpoint-quiz.md` |
| Week 10 | ❌ Missing | Create `checkpoint-quiz.md` |
| Week 11 | ❌ Missing | Create `checkpoint-quiz.md` |
| Week 12 | ❌ Missing | Create `checkpoint-quiz.md` |

### 1.2 Notebook Content Fixes

| File | Issue | Fix |
|------|-------|-----|
| `week-07/day-2-registers.ipynb` | Typo in cell 6 (`result[idx] = r1 * r2 o`) | Remove stray 'o' |

---

## Phase 2: New Weeks (13-14)

### Week 13: Tensor Cores & Mixed Precision

**Learning Goals:**
- Understand Tensor Core architecture
- Implement WMMA (Warp Matrix Multiply-Accumulate)
- Use mixed precision (FP16/TF32/BF16)
- Leverage cuBLAS with Tensor Cores

**Daily Schedule:**
| Day | Topic | Notebook |
|-----|-------|----------|
| 1 | Tensor Core Architecture | `day-1-tensor-core-basics.ipynb` |
| 2 | WMMA Programming | `day-2-wmma.ipynb` |
| 3 | Mixed Precision Training | `day-3-mixed-precision.ipynb` |
| 4 | cuBLAS Tensor Core Mode | `day-4-cublas-tensor.ipynb` |
| 5 | Practice & Quiz | `checkpoint-quiz.md` |

### Week 14: Real-World Applications

**Learning Goals:**
- Implement production-quality kernels
- Understand Flash Attention concepts
- Build custom PyTorch extensions
- Performance comparison methodology

**Daily Schedule:**
| Day | Topic | Notebook |
|-----|-------|----------|
| 1 | Softmax & LayerNorm Optimized | `day-1-fused-kernels.ipynb` |
| 2 | Attention Mechanisms | `day-2-attention.ipynb` |
| 3 | PyTorch CUDA Extensions | `day-3-pytorch-extensions.ipynb` |
| 4 | Benchmarking Methodology | `day-4-benchmarking.ipynb` |
| 5 | Practice & Quiz | `checkpoint-quiz.md` |

---

## Phase 3: Blog Post Infrastructure

### 3.1 Blog Template Structure

Create `/blog-templates/` directory with:
- `BLOG-POST-TEMPLATE.md` - Standard template for all posts
- `BLOG-SERIES-PLAN.md` - Planned blog series outline
- `README.md` - Instructions for converting notebooks to blogs

### 3.2 Recommended Blog Series

| # | Title | Source | Priority |
|---|-------|--------|----------|
| 1 | "Your First CUDA Kernel" | Week 1 Day 1 | 🔥 High |
| 2 | "Understanding GPU Memory Coalescing" | Week 2 Day 1 | 🔥 High |
| 3 | "Parallel Reduction: From Naive to Optimal" | Week 4 Day 1-2 | 🔥 High |
| 4 | "Tiled Matrix Multiply Step by Step" | Week 6 Day 2 | 🔥 High |
| 5 | "The Roofline Model for GPU Optimization" | Week 8 Day 2 | ⭐ Medium |
| 6 | "CUDA Streams: Overlapping Compute & Transfer" | Week 9 Day 1-2 | ⭐ Medium |
| 7 | "Tensor Cores: Practical Guide" | Week 13 Day 1-2 | ⭐ Medium |
| 8 | "Building PyTorch CUDA Extensions" | Week 14 Day 3 | ⭐ Medium |

---

## Phase 4: Documentation Updates

### 4.1 Update 12-week-curriculum.md
- Extend to 14 weeks
- Update progress tracker
- Add new week descriptions

### 4.2 Update README.md
- Add blog post section
- Update notebook count
- Add Week 13-14 to Colab links

---

## Implementation Checklist

- [ ] Phase 1.1: Create 10 missing checkpoint quizzes
- [ ] Phase 1.2: Fix typo in week-07/day-2-registers.ipynb
- [ ] Phase 2.1: Create Week 13 (4 notebooks + quiz)
- [ ] Phase 2.2: Create Week 14 (4 notebooks + quiz)
- [ ] Phase 3.1: Create blog template infrastructure
- [ ] Phase 4.1: Update curriculum documentation
- [ ] Phase 4.2: Update README with new content
- [ ] Commit and push all changes

---

## Timeline

| Phase | Estimated Time | Status |
|-------|---------------|--------|
| Phase 1 | 30 minutes | 🔄 In Progress |
| Phase 2 | 2 hours | ⬜ Not Started |
| Phase 3 | 30 minutes | ⬜ Not Started |
| Phase 4 | 15 minutes | ⬜ Not Started |

---

## Success Metrics

After implementation:
- 14 weeks of content (56+ notebooks)
- All weeks have checkpoint quizzes
- Blog infrastructure ready for publishing
- Curriculum covers modern GPU features (Tensor Cores)
