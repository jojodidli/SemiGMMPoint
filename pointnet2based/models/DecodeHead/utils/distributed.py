import torch
import torch.distributed as dist
# import torch.distributed as dist



# utils
@torch.no_grad()
def concat_all_gather_wo_grad(tensor):
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

@torch.no_grad()
def concat_all_gather_wo_grad_single(tensor):

    return tensor