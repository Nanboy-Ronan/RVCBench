# Lightweight pure-PyTorch fallback for monotonic alignment.
# The original package ships a Cython extension that fails to build on Python 3.12,
# so we replicate the small DP used for maximum_path here.
import torch


def _maximum_path_single(value: torch.Tensor, mask: torch.Tensor, max_neg_val: float) -> torch.Tensor:
    """Compute maximum monotonic path for one sample."""
    t_y, t_x = mask.shape
    max_neg = value.new_tensor(max_neg_val)

    # clone to avoid mutating caller tensors
    work = value.clone()
    work = work.masked_fill(~mask, max_neg)

    for y in range(t_y):
        x_start = max(0, t_x + y - t_y)
        x_end = min(t_x, y + 1)
        for x in range(x_start, x_end):
            v_cur = max_neg if x == y else work[y - 1, x]
            if x == 0:
                v_prev = work.new_tensor(0.0) if y == 0 else max_neg
            else:
                v_prev = work[y - 1, x - 1]
            work[y, x] = work[y, x] + torch.maximum(v_prev, v_cur)

    path = work.new_zeros((t_y, t_x), dtype=torch.int32)
    index = t_x - 1
    for y in range(t_y - 1, -1, -1):
        path[y, index] = 1
        if index != 0:
            if index == y:
                index -= 1
            elif y > 0 and work[y - 1, index] < work[y - 1, index - 1]:
                index -= 1
    return path


def maximum_path(value: torch.Tensor, mask: torch.Tensor, max_neg_val: float = -1e9) -> torch.Tensor:
    """
    Torch implementation compatible with monotonic_align.maximum_path.

    Args:
        value: Tensor [b, t_y, t_x]
        mask: Tensor [b, t_y, t_x] with True for valid positions
        max_neg_val: fill value for masked positions
    """
    if value.dim() != 3 or mask.dim() != 3:
        raise ValueError("value and mask must be 3-D tensors")

    batch_paths = []
    mask_bool = mask.bool()
    for b in range(value.shape[0]):
        batch_paths.append(_maximum_path_single(value[b], mask_bool[b], max_neg_val))
    return torch.stack(batch_paths, dim=0)
