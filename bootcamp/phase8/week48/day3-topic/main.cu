/**
 * Week 48, Day 3: Checkpointing Distributed
 */
#include <cstdio>

int main() {
    printf("Week 48 Day 3: Checkpointing in Distributed Training\n\n");
    
    printf("DDP Checkpointing:\n");
    printf("```python\n");
    printf("# Save on rank 0 only to avoid duplicates\n");
    printf("if dist.get_rank() == 0:\n");
    printf("    torch.save({\n");
    printf("        'epoch': epoch,\n");
    printf("        'model_state_dict': model.module.state_dict(),  # Note: .module!\n");
    printf("        'optimizer_state_dict': optimizer.state_dict(),\n");
    printf("        'loss': loss,\n");
    printf("    }, 'checkpoint.pth')\n");
    printf("\n");
    printf("# All ranks wait for save to complete\n");
    printf("dist.barrier()\n");
    printf("```\n\n");
    
    printf("FSDP Checkpointing:\n");
    printf("```python\n");
    printf("from torch.distributed.fsdp import StateDictType\n");
    printf("from torch.distributed.checkpoint import save, load\n");
    printf("\n");
    printf("# FSDP-specific: gather full state or save sharded\n");
    printf("with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):\n");
    printf("    state_dict = model.state_dict()\n");
    printf("    if rank == 0:\n");
    printf("        torch.save(state_dict, 'model.pth')\n");
    printf("\n");
    printf("# Or save sharded (more efficient for huge models)\n");
    printf("with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):\n");
    printf("    save({'model': model}, checkpoint_dir='./ckpt')\n");
    printf("```\n\n");
    
    printf("Resume Training:\n");
    printf("```python\n");
    printf("# Load checkpoint\n");
    printf("map_location = {'cuda:0': f'cuda:{rank}'}\n");
    printf("checkpoint = torch.load('checkpoint.pth', map_location=map_location)\n");
    printf("model.module.load_state_dict(checkpoint['model_state_dict'])\n");
    printf("optimizer.load_state_dict(checkpoint['optimizer_state_dict'])\n");
    printf("```\n");
    
    return 0;
}
