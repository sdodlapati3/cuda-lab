# CUDA Learning Notebook Quality Guide

This guide defines the standards for transforming CUDA notebooks from "lab manuals" into "teaching documents" suitable for standalone reading.

## Core Philosophy

> **Goal**: A reader should be able to understand the concepts WITHOUT running the code.
> The code demonstrates what they already understand conceptually.

```
❌ Current:  Header → Code dump → Header → Code dump
✅ Target:   Hook → Why → Analogy → Diagram → Snippet → Full code → Benchmark → Exercises
```

---

## 🎴 Concept Cards

Concept Cards are structured markdown sections inserted BEFORE major code blocks. They provide the "connective tissue" that transforms code repositories into teaching documents.

### Concept Card Template

```markdown
<details open>
<summary>💡 <b>Concept: [Title - What We're About to Learn]</b></summary>

### 🎯 The Problem
[1-2 sentences: What challenge are we solving? Why should the reader care?]

### 🚚 The Analogy
[A real-world metaphor that makes the concept intuitive. Use vivid imagery.]

### 🔧 Hardware Reality
[Technical explanation grounded in GPU architecture. Numbers and specifics.]

### ✅ The Pattern
```
Good: [Show the correct approach in pseudocode]
Bad:  [Show the anti-pattern]
```

### ⚠️ Common Gotchas
- [Mistake 1 developers make]
- [Mistake 2 developers make]

</details>
```

### Concept Card Types

| Type | Use When | Key Elements |
|------|----------|--------------|
| **Foundation** | Introducing a new concept | Analogy, hardware context, visual |
| **Mechanism** | Explaining HOW something works | Step-by-step breakdown, diagram |
| **Optimization** | Showing performance techniques | Before/after, benchmarks, trade-offs |
| **Gotcha** | Warning about common mistakes | Anti-pattern, why it fails, fix |
| **Comparison** | Contrasting approaches | Side-by-side, when to use each |

---

## 📊 Visual Standards

### Required Visualizations

Each notebook should include:

1. **Memory Access Pattern Diagram** (for memory-related topics)
   - Show thread-to-memory mapping
   - Color code: 🟢 Efficient, 🔴 Inefficient

2. **Performance Comparison Chart** (matplotlib)
   - Bar chart comparing approaches
   - Include actual numbers, not just "faster"

3. **Architecture Diagram** (for hardware concepts)
   - Use mermaid or ASCII with clear labels

### Visualization Code Pattern

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_performance_comparison(labels, values, title, ylabel="Time (ms)"):
    """Standard performance comparison chart."""
    colors = ['#2ecc71' if v == min(values) else '#e74c3c' if v == max(values) else '#3498db' 
              for v in values]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, values, color=colors)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel(ylabel)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.02,
                f'{val:.2f}', ha='center', fontsize=11)
    
    plt.tight_layout()
    plt.show()
```

---

## 📝 Code Presentation Standards

### The Snippet-First Approach

**Don't** dump 50+ lines of code at once. **Do** show the key insight first:

```markdown
The key insight is in the index calculation:

​```cuda
// Coalesced: Adjacent threads access adjacent memory
int idx = blockIdx.x * blockDim.x + threadIdx.x;
float val = input[idx];  // Threads 0,1,2,3... access addresses 0,1,2,3...

// Strided: Adjacent threads access scattered memory  
float val = input[idx * STRIDE];  // Threads 0,1,2,3... access addresses 0,32,64,96...
​```

<details>
<summary>📄 Full Implementation (click to expand)</summary>

[Full 50-line code here]

</details>
```

### Code Comment Standards

```cuda
// ============================================
// SECTION: Memory Access Pattern
// ============================================

__global__ void kernel(float* data) {
    // Calculate global thread index
    // Formula: block_offset + thread_offset
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // KEY INSIGHT: This access pattern is coalesced because
    // thread 0 accesses data[0], thread 1 accesses data[1], etc.
    // All 32 threads in a warp access a contiguous 128-byte region.
    float val = data[idx];
}
```

---

## 🎓 Pedagogical Structure

### Section Flow

Each major topic should follow this structure:

```
1. 🎣 HOOK (1-2 sentences)
   "Why is my GPU code 32x slower than expected?"

2. 💡 CONCEPT CARD
   [Analogy + Hardware Reality + Pattern]

3. 📊 VISUAL
   [Diagram showing the concept]

4. 🔍 CODE SNIPPET
   [Just the 2-5 lines that matter]

5. 📄 FULL CODE
   [Complete implementation, collapsible]

6. 📈 BENCHMARK
   [Show the performance difference with numbers]

7. 🔑 KEY TAKEAWAY
   [One sentence summary]
```

### Transition Phrases

Use these to create smooth narrative flow:

- "Now that we understand WHY this happens, let's see HOW to fix it..."
- "The theory is clear, but what does this look like in practice?"
- "Before we dive into the code, let's visualize what's happening..."
- "You might be wondering: why does the hardware work this way?"
- "Let's prove this with a benchmark..."

---

## ✅ Quality Checklist

Before a notebook is considered "blog-ready", verify:

### Content
- [ ] Every code block has a preceding concept card or explanation
- [ ] Hardware "why" is explained (not just "what")
- [ ] At least one real-world analogy per major concept
- [ ] Performance claims are backed by benchmarks with numbers

### Visuals
- [ ] At least one diagram per major concept
- [ ] Performance comparison chart included
- [ ] ASCII art replaced with proper visualizations (where possible)

### Code
- [ ] Key insights shown as snippets before full code
- [ ] Full implementations are collapsible
- [ ] Comments explain the "why", not just the "what"
- [ ] Index calculations are explained step-by-step

### Flow
- [ ] Smooth transitions between sections
- [ ] No abrupt jumps from text to code
- [ ] Hook at the beginning captures attention
- [ ] Summary/takeaways at the end

### Exercises
- [ ] C++ exercises are primary
- [ ] Python/Numba exercises are optional
- [ ] Clear connection between tutorial content and exercises

---

## 📚 Analogy Library

Reusable analogies for common CUDA concepts:

### Memory Coalescing
> **The Delivery Truck**: Think of the memory controller like a delivery truck with a minimum package size (128 bytes). Even if you only want 4 bytes, the truck delivers the full 128-byte package. If 32 threads request scattered addresses, you need 32 trucks instead of 1.

### Shared Memory
> **The Team Whiteboard**: Global memory is like a filing cabinet in another building - slow to access. Shared memory is like a whiteboard in your meeting room - everyone on the team (block) can read/write quickly, but other teams can't see it.

### Bank Conflicts
> **The Library Aisles**: Imagine 32 students trying to grab books simultaneously. If they each go to different aisles (banks), no problem. If 4 students need books from the same aisle, 3 must wait. That's a 4-way bank conflict.

### Warp Divergence
> **The Tour Group**: A warp is like a tour group that must stay together. If the guide says "those with red shirts go left, blue shirts go right", the ENTIRE group must first go left (red shirts active), then backtrack and go right (blue shirts active). Two trips instead of one.

### Occupancy
> **The Restaurant**: GPU SMs are like restaurant kitchens. Occupancy is how many chefs (warps) are cooking simultaneously. More chefs = better at hiding wait times (memory latency), but too many chefs with huge prep areas (registers) won't fit.

### Streams
> **Assembly Lines**: Streams are independent assembly lines. While one line waits for parts (memory transfer), another can be assembling (kernel execution). Multiple lines = overlapping work = faster throughput.

### CUDA Graphs
> **Recipe Cards**: Instead of telling the kitchen each step one at a time (launch overhead), you hand them a complete recipe card (graph). The kitchen knows the entire workflow upfront and can optimize execution.

---

## 🗂️ Notebook Transformation Tracking

| Week | Day | Notebook | Status | Concept Cards | Visuals | Reviewed |
|------|-----|----------|--------|---------------|---------|----------|
| 01 | 1 | gpu-basics | ⬜ Pending | 0/3 | 0/2 | No |
| 01 | 2 | thread-indexing | ⬜ Pending | 0/3 | 0/2 | No |
| 02 | 1 | memory-coalescing | 🔄 In Progress | 0/4 | 0/3 | No |
| ... | ... | ... | ... | ... | ... | ... |

**Status Legend**: ⬜ Pending | 🔄 In Progress | ✅ Complete | 🔍 In Review

---

## Getting Started

To transform a notebook:

1. Read through the entire notebook
2. Identify each major code block
3. For each code block, create a concept card
4. Add visualizations where helpful
5. Refactor code into snippet + full implementation
6. Add transitions between sections
7. Run through the quality checklist
8. Get peer review

Start with the flagship notebooks first, then use them as templates for the rest.
