# Exercise 08: VMM Growable Buffer

## Objective
Implement a growable GPU buffer using Virtual Memory Management APIs.

## Background
Traditional CUDA memory allocation:
- `cudaMalloc` allocates fixed-size memory
- Growing requires allocate + copy + free

VMM approach:
- Reserve large virtual address range upfront
- Map physical memory on demand
- No copying when growing!

## VMM Workflow
1. `cuMemAddressReserve` - Reserve virtual address space
2. `cuMemCreate` - Create physical memory handle
3. `cuMemMap` - Map physical to virtual
4. `cuMemSetAccess` - Set access permissions
5. Growing: Just map more physical memory to reserved range

## Requirements
- CUDA Driver API
- Compute Capability 6.0+

## Testing
```bash
make
./test.sh
```
