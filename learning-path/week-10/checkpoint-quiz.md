# Week 10 Checkpoint Quiz: CUDA Graphs

## Conceptual Questions

### Q1: What Problem Do Graphs Solve?
Why would you use CUDA Graphs instead of regular stream-based launches?

**Your Answer:**

---

### Q2: Stream Capture
Explain the stream capture process for creating a graph.

**Your Answer:**

---

### Q3: Graph Instantiation
What happens during cudaGraphInstantiate? Why is it separate from execution?

**Your Answer:**

---

### Q4: Graph vs Stream
When would you NOT want to use a CUDA Graph?

**Your Answer:**

---

### Q5: Graph Updates
How can you modify a graph's parameters without rebuilding it?

**Your Answer:**

---

## Code Analysis

### Q6: Capture Issues
```cpp
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

kernel1<<<grid, block, 0, stream>>>(d_a, n);
cudaMemcpy(h_b, d_b, size, cudaMemcpyDeviceToHost);  // Problem?
kernel2<<<grid, block, 0, stream>>>(d_b, n);

cudaStreamEndCapture(stream, &graph);
```

What's wrong with this capture?

**Your Answer:**

---

### Q7: Graph Execution
```cpp
// Created graph once
cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);

for (int i = 0; i < 1000; i++) {
    // Want to change input data each iteration
    cudaMemcpy(d_input, h_inputs[i], size, H2D);  // Is this right?
    cudaGraphLaunch(graphExec, stream);
    cudaStreamSynchronize(stream);
}
```

Is this the most efficient approach? How could it be improved?

**Your Answer:**

---

## Practical Exercises

### Exercise 1: Basic Graph Capture
Capture a simple H2D → Kernel → D2H pipeline and measure speedup vs stream.

```cpp
// Your implementation:
```

### Exercise 2: Explicit Graph Construction
Build a graph with parallel kernel nodes that merge into a final node.

```cpp
// Your implementation:
```

### Exercise 3: Parameter Update
Create a graph and update kernel parameters between launches.

```cpp
// Your implementation:
```

---

## Performance Analysis

Compare these metrics for your graph vs stream implementation:
1. Average launch latency
2. Total time for 1000 iterations
3. CPU overhead

**Your Analysis:**

---

## Self-Assessment

Rate your understanding (1-5):
- [ ] Stream capture mechanism: ___
- [ ] Explicit graph construction: ___
- [ ] Graph instantiation vs execution: ___
- [ ] Parameter updates: ___
- [ ] Performance characteristics: ___

**Total Score: ___ / 25**

**Areas to Review:**
