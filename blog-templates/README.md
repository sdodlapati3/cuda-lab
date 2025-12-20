# Blog Templates

This directory contains templates for converting CUDA learning content into blog posts.

## Contents

| File | Purpose |
|------|---------|
| [BLOG-POST-TEMPLATE.md](BLOG-POST-TEMPLATE.md) | Template for individual blog posts |
| [BLOG-SERIES-PLAN.md](BLOG-SERIES-PLAN.md) | Recommended blog series structure |

## Recommended Blog Series

Based on the 14-week curriculum, here are high-value blog series:

### Series 1: CUDA Fundamentals (Weeks 1-4)
- **Audience**: Beginners with Python/C++ background
- **Posts**: 4-6 articles
- **Key topics**: GPU architecture, thread indexing, memory hierarchy

### Series 2: Optimization Deep Dive (Weeks 5-8)
- **Audience**: Intermediate CUDA developers
- **Posts**: 4-5 articles
- **Key topics**: Memory coalescing, shared memory, occupancy

### Series 3: Tensor Cores & Mixed Precision (Week 13)
- **Audience**: ML/DL practitioners
- **Posts**: 3-4 articles
- **Key topics**: WMMA, FP16 training, cuBLAS integration

### Series 4: Production CUDA (Week 14)
- **Audience**: Engineers deploying CUDA code
- **Posts**: 3-4 articles
- **Key topics**: Kernel fusion, attention optimization, benchmarking

## Conversion Guidelines

1. **Start from notebook**: Each notebook can become 1-2 blog posts
2. **Add narrative**: Explain the "why" not just the "how"
3. **Include visuals**: Add diagrams for memory layouts, thread organization
4. **Provide runnable code**: Link to Colab notebooks
5. **SEO optimization**: Use descriptive titles and headers

## Quick Start

1. Copy `BLOG-POST-TEMPLATE.md` to your blog directory
2. Fill in sections based on a notebook
3. Add intro/outro content
4. Create visuals for complex concepts
5. Test all code snippets
