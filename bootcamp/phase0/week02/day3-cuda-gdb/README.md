# Day 3: cuda-gdb Basics

## What You'll Learn

- Set breakpoints in GPU code
- Inspect thread and warp state
- Step through kernel execution
- Examine memory contents

## The Tool: cuda-gdb

`cuda-gdb` is GDB extended for CUDA. It can debug both host and device code.

```bash
cuda-gdb ./your_program
```

## Quick Start

```bash
./build.sh
cuda-gdb ./build/debug_example
```

## Essential Commands

### Basic GDB Commands
```
(cuda-gdb) break main              # Breakpoint at main
(cuda-gdb) run                     # Start program
(cuda-gdb) next                    # Step over
(cuda-gdb) step                    # Step into
(cuda-gdb) continue                # Continue to next breakpoint
(cuda-gdb) print variable          # Print value
(cuda-gdb) quit                    # Exit
```

### CUDA-Specific Commands
```
(cuda-gdb) break my_kernel         # Breakpoint in kernel
(cuda-gdb) cuda thread             # Show current thread
(cuda-gdb) cuda thread (1,0,0)     # Switch to thread
(cuda-gdb) cuda block              # Show current block
(cuda-gdb) cuda block (2,0,0)      # Switch to block
(cuda-gdb) cuda kernel             # Show active kernels
(cuda-gdb) info cuda threads       # List all CUDA threads
(cuda-gdb) info cuda lanes         # Show warp lanes
```

### Memory Inspection
```
(cuda-gdb) print @global data[0]   # Global memory
(cuda-gdb) print @shared sdata     # Shared memory
(cuda-gdb) print @local local_var  # Local memory
(cuda-gdb) x/10f data              # Examine 10 floats
```

## Debugging Session Example

```bash
$ cuda-gdb ./build/debug_example

(cuda-gdb) break simple_kernel
Breakpoint 1 at 0x... in simple_kernel

(cuda-gdb) run
Thread 1 hit Breakpoint 1, simple_kernel<<<(1,1,1),(32,1,1)>>> at debug_example.cu:15

(cuda-gdb) cuda thread
Current CUDA thread (0,0,0), block (0,0,0), device 0, SM 0, warp 0, lane 0

(cuda-gdb) print idx
$1 = 0

(cuda-gdb) cuda thread (5,0,0)
[Switching to CUDA thread (5,0,0)]

(cuda-gdb) print idx
$2 = 5

(cuda-gdb) info cuda threads
  Block (0,0,0), Thread (0,0,0) ...
  Block (0,0,0), Thread (1,0,0) ...
  ...
```

## Breakpoint Conditions

```
# Break only when thread index matches
(cuda-gdb) break kernel if threadIdx.x == 0 && blockIdx.x == 1

# Break on specific condition
(cuda-gdb) break kernel if data[idx] > 100
```

## Watchpoints

```
(cuda-gdb) watch @global data[42]  # Break when data[42] changes
```

## Common Issues

### "Cannot insert breakpoint"
Build with `-G` flag for device debugging:
```cmake
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")
```

### "No CUDA device"
Ensure you're on a machine with a GPU. cuda-gdb requires a real GPU.

### Slow debugging
`-G` disables all optimizations. Debugging large kernels is slow.

## TUI Mode

For a more visual experience:
```
(cuda-gdb) tui enable
```

Shows source code alongside commands.

## Exercises

1. Set a breakpoint in `simple_kernel`
2. Print the thread index for different threads
3. Examine shared memory contents
4. Use a conditional breakpoint to stop at thread 15
5. Watch a memory location and see when it changes
