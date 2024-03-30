from megatron_mini.core import parallel_state
import megatron_mini.core.tensor_parallel
import megatron_mini.core.utils

# Alias parallel_state as mpu, its legacy name
mpu = parallel_state

__all__ = [
    "parallel_state",
    "tensor_parallel",
    "utils",
]
