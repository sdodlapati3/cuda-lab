# Exercise: CUDA Events

## Objective
Use CUDA events for accurate timing and inter-stream synchronization.

## Background

CUDA events mark points in streams for:
1. **Timing** - Accurate GPU timing without CPU overhead
2. **Synchronization** - Cross-stream dependencies

## API

```cpp
cudaEventCreate(&event);
cudaEventRecord(event, stream);      // Record event in stream
cudaEventSynchronize(event);         // Wait for event on CPU
cudaStreamWaitEvent(stream, event);  // Stream waits for event
cudaEventElapsedTime(&ms, start, end);
```

## Task

1. Implement accurate kernel timing with events
2. Implement producer-consumer pattern with event sync
3. Compare event timing vs CPU timing accuracy

## Files

- `events.cu` - Skeleton
- `solution.cu` - Reference
- `Makefile` - Build
- `test.sh` - Test
