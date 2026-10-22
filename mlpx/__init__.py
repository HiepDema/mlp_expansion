"""MLP expansion: parameter-efficient fine-tuning by adding neurons to a gated MLP.

    from mlpx import apply_expansion, fold_expansion

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B")
    apply_expansion(model, rank=14)      # 3 * hidden * rank new params per layer
    # ... train ...
    fold_expansion(model)                # plain MLP with intermediate_size + rank
"""
from mlpx.adapters import ExpandedMLP, LoRALinear
from mlpx import official
from mlpx.surgery import (
    ATTN_LINEARS,
    MLP_LINEARS,
    apply_expansion,
    apply_lora,
    count_params,
    expansion_params_per_layer,
    expansion_rank_matching_lora,
    fold_expansion,
    get_blocks,
    linear_shapes,
    lora_params_per_layer,
    merge_all,
    merge_lora,
    resolve,
)
