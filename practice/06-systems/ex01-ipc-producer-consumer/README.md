# Exercise 1: IPC Producer-Consumer

## Objective

Implement a producer-consumer pattern using CUDA IPC to share GPU memory between processes.

## Task

Create two programs:
1. **Producer**: Allocates GPU memory, fills with data, exports IPC handle
2. **Consumer**: Imports IPC handle, reads data, verifies correctness

## Requirements

1. Producer must:
   - Allocate GPU memory with `cudaMalloc`
   - Fill with a known pattern (e.g., `data[i] = i * 2 + 1`)
   - Export handle with `cudaIpcGetMemHandle`
   - Write handle to file or shared memory
   - Wait for consumer to finish (use a signal file)

2. Consumer must:
   - Read IPC handle from file
   - Open memory with `cudaIpcOpenMemHandle`
   - Verify data correctness
   - Close handle with `cudaIpcCloseMemHandle`

## Files to Complete

- `producer.cu` - Producer process
- `consumer.cu` - Consumer process

## Testing

```bash
# Terminal 1
./producer

# Terminal 2 (while producer is waiting)
./consumer
```

## Expected Output

Producer:
```
Producer: Allocated 4096 bytes
Producer: Filled with pattern (data[i] = i * 2 + 1)
Producer: IPC handle written to ipc_handle.bin
Producer: Waiting for consumer...
Producer: Consumer finished, cleaning up
```

Consumer:
```
Consumer: Read IPC handle from ipc_handle.bin
Consumer: Opened shared memory
Consumer: Verifying data...
Consumer: SUCCESS - All 1024 values correct!
```

## Hints

- The IPC handle is a 64-byte opaque structure
- Both processes must use the same GPU device
- Producer must not free memory until consumer is done
- Use file locks or signal files for synchronization
