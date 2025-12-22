/**
 * Week 45, Day 3: NCCL API
 */
#include <cstdio>

int main() {
    printf("Week 45 Day 3: NCCL C API\n\n");
    
    printf("Basic NCCL Usage:\n");
    printf("```cpp\n");
    printf("#include <nccl.h>\n");
    printf("\n");
    printf("// 1. Initialize\n");
    printf("ncclComm_t comm;\n");
    printf("ncclUniqueId id;\n");
    printf("ncclGetUniqueId(&id);  // Root generates ID\n");
    printf("// Broadcast id to all ranks (MPI, etc.)\n");
    printf("ncclCommInitRank(&comm, nranks, id, rank);\n");
    printf("\n");
    printf("// 2. Collective operation\n");
    printf("ncclAllReduce(sendbuff, recvbuff, count, ncclFloat,\n");
    printf("              ncclSum, comm, stream);\n");
    printf("\n");
    printf("// 3. Cleanup\n");
    printf("ncclCommDestroy(comm);\n");
    printf("```\n\n");
    
    printf("Available Operations:\n");
    printf("┌─────────────────────────────────────────────────────────────────────┐\n");
    printf("│ ncclAllReduce(send, recv, count, dtype, op, comm, stream)          │\n");
    printf("│ ncclBroadcast(send, recv, count, dtype, root, comm, stream)        │\n");
    printf("│ ncclReduce(send, recv, count, dtype, op, root, comm, stream)       │\n");
    printf("│ ncclAllGather(send, recv, count, dtype, comm, stream)              │\n");
    printf("│ ncclReduceScatter(send, recv, count, dtype, op, comm, stream)      │\n");
    printf("│ ncclSend(send, count, dtype, peer, comm, stream)                   │\n");
    printf("│ ncclRecv(recv, count, dtype, peer, comm, stream)                   │\n");
    printf("└─────────────────────────────────────────────────────────────────────┘\n\n");
    
    printf("Grouping Operations:\n");
    printf("```cpp\n");
    printf("// Multiple collectives in single launch\n");
    printf("ncclGroupStart();\n");
    printf("ncclAllReduce(...);\n");
    printf("ncclAllReduce(...);\n");
    printf("ncclGroupEnd();\n");
    printf("```\n");
    
    return 0;
}
