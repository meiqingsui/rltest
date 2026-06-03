# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""
Script to extract logp values from Megatron-LM training side to debug Train-Inference Mismatch.

export ASCEND_RT_VISIBLE_DEVICES=12,13,14,15
export CUDA_DEVICE_MAX_CONNECTIONS=1
torchrun --nproc_per_node=2 test_train_logp.py \
    --hf-checkpoint /mnt/sfs_turbo/models/Qwen3.5-9B/ \
    --tensor-model-parallel-size 2 \
    --micro-batch-size 1 --seq-length 2048 \
    --test-tokens "12, 134, 45, 10, 89" --qkv-format bshd
"""

import dataclasses
import os
import sys
import traceback
from argparse import Namespace
from contextlib import contextmanager

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

import logging

try:
    import mindspeed.megatron_adaptor  # noqa: F401
except ImportError:
    raise ValueError("mindspeed.megatron_adaptor not found")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 0. Device abstraction (GPU / Ascend NPU)
# ---------------------------------------------------------------------------

def _is_npu_available() -> bool:
    try:
        import torch_npu  # noqa: F401
        return hasattr(torch, "npu") and torch.npu.is_available()
    except ImportError:
        return False

def get_accelerator_type() -> str:
    return "npu" if _is_npu_available() else "cuda"

def get_dist_backend() -> str:
    return "hccl" if get_accelerator_type() == "npu" else "nccl"

def set_device(local_rank: int) -> torch.device:
    acc = get_accelerator_type()
    device_str = f"{acc}:{local_rank}"
    if acc == "npu":
        import torch_npu
        torch.npu.set_device(device_str)
    else:
        torch.cuda.set_device(device_str)
    return torch.device(device_str)

def current_device() -> torch.device:
    acc = get_accelerator_type()
    if acc == "npu":
        import torch_npu
        return torch.device(f"npu:{torch.npu.current_device()}")
    return torch.device(f"cuda:{torch.cuda.current_device()}")

def to_device(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to(current_device())

def _init_npu_environment() -> None:
    if get_accelerator_type() != "npu":
        return
    import torch_npu  # noqa: F401
    try:
        import mindspeed.megatron_adaptor  # noqa: F401
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# 1. Argument parsing
# ---------------------------------------------------------------------------

def parse_test_args() -> Namespace:
    from megatron.training.arguments import parse_args as _megatron_parse_args
    from megatron.training.arguments import validate_args as _megatron_validate_args
    from megatron.training.tokenizer.tokenizer import _vocab_size_with_padding

    defaults = [
        "--tokenizer-type", "HuggingFaceTokenizer",
        "--use-distributed-optimizer",
        "--overlap-grad-reduce",
        "--bf16",
        "--lr", "1e-5",
        "--min-lr", "1e-6",
        "--num-layers", "1",
        "--hidden-size", "1",
        "--num-attention-heads", "1",
        "--no-rope-fusion",
    ]

    existing = set(sys.argv)
    inject = []
    i = 0
    while i < len(defaults):
        flag = defaults[i]
        if flag.startswith("--") and flag not in existing:
            inject.append(flag)
            if i + 1 < len(defaults) and not defaults[i + 1].startswith("--"):
                inject.append(defaults[i + 1])
                i += 2
            else:
                i += 1
        else:
            i += 1 if not (i + 1 < len(defaults) and not defaults[i + 1].startswith("--")) else 2

    sys.argv.extend(inject)
    args = _megatron_parse_args(extra_args_provider=_add_test_args, ignore_unknown_args=True)

    args.rank = int(os.environ.get("RANK", 0))
    args.world_size = int(os.environ.get("WORLD_SIZE", 1))
    args.use_distributed_optimizer = True
    args.variable_seq_lengths = True

    if not args.bf16 and not args.fp16:
        args.bf16 = True

    if args.seq_length and not args.max_position_embeddings:
        args.max_position_embeddings = args.seq_length

    if not args.tokenizer_model:
        args.tokenizer_model = args.hf_checkpoint

    if args.vocab_size and not args.padded_vocab_size:
        args.padded_vocab_size = _vocab_size_with_padding(args.vocab_size, args)

    if not hasattr(args, "num_rollout") or args.num_rollout is None:
        args.num_rollout = 10
    if not hasattr(args, "rollout_batch_size") or args.rollout_batch_size is None:
        args.rollout_batch_size = 1
    if not hasattr(args, "n_samples_per_prompt") or args.n_samples_per_prompt is None:
        args.n_samples_per_prompt = 1

    _megatron_validate_args(args)
    args.variable_seq_lengths = True

    return args


def _add_test_args(parser):
    group = parser.add_argument_group("Logp Test")
    group.add_argument("--hf-checkpoint", type=str, required=True)
    group.add_argument("--megatron-to-hf-mode", type=str, default="bridge")
    group.add_argument("--test-tokens", type=str, default="10,11,12,13,14,15,16,17", 
                       help="Comma separated list of token ids for input")
    return parser


# ---------------------------------------------------------------------------
# 2. Distributed initialization
# ---------------------------------------------------------------------------

def init_distributed(args: Namespace) -> None:
    from datetime import timedelta
    _init_npu_environment()
    acc_type = get_accelerator_type()
    backend = get_dist_backend()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    set_device(local_rank)

    dist.init_process_group(backend=backend, timeout=timedelta(minutes=10))
    args.rank = dist.get_rank()
    args.world_size = dist.get_world_size()

    from megatron.core import mpu
    from megatron.core.num_microbatches_calculator import init_num_microbatches_calculator
    from megatron.training.global_vars import set_args

    set_args(args)
    mpu.initialize_model_parallel(
        args.tensor_model_parallel_size,
        args.pipeline_model_parallel_size,
        args.virtual_pipeline_model_parallel_size,
        context_parallel_size=args.context_parallel_size,
        expert_model_parallel_size=args.expert_model_parallel_size,
        expert_tensor_parallel_size=getattr(args, "expert_tensor_parallel_size", 1),
    )

    import random
    from megatron.core import tensor_parallel
    seed = args.seed + 100 * mpu.get_pipeline_model_parallel_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if acc_type == "npu":
        import torch_npu
        torch.npu.manual_seed(seed)
    tensor_parallel.model_parallel_cuda_manual_seed(seed)

    init_num_microbatches_calculator(
        args.rank, args.rampup_batch_size, args.global_batch_size,
        args.micro_batch_size, args.data_parallel_size, False,
    )


# ---------------------------------------------------------------------------
# 3. Model building via AutoBridge
# ---------------------------------------------------------------------------

def build_model_with_bridge(args: Namespace):
    from megatron.bridge import AutoBridge
    from megatron.core.enums import ModelType
    from megatron.training.training import get_model

    bridge = AutoBridge.from_hf_pretrained(args.hf_checkpoint, trust_remote_code=True)
    provider = bridge.to_megatron_provider(load_weights=True)

    args.vocab_size = provider.vocab_size
    if not getattr(args, "padded_vocab_size", None):
        if getattr(provider, "should_pad_vocab", False):
            from megatron.bridge.utils.vocab_utils import calculate_padded_vocab_size
            args.padded_vocab_size = calculate_padded_vocab_size(
                provider.vocab_size, provider.make_vocab_size_divisible_by, provider.tensor_model_parallel_size
            )
        else:
            args.padded_vocab_size = provider.vocab_size

    bridge_keys = [
        "attention_backend", "tensor_model_parallel_size", "sequence_parallel",
        "pipeline_model_parallel_size", "context_parallel_size",
        "expert_model_parallel_size", "expert_tensor_parallel_size",
        "variable_seq_lengths", "attention_softmax_in_fp32",
        "bias_dropout_fusion", "apply_rope_fusion",
        "recompute_granularity", "recompute_method", "recompute_num_layers",
        "distribute_saved_activations",
        "moe_router_load_balancing_type", "moe_router_dtype",
        "moe_aux_loss_coeff", "moe_token_dispatcher_type",
        "micro_batch_size","seq_length"
    ]

    # fix: provider all set
    args_dict = vars(args)
    for attr in vars(provider):
        if attr in args_dict and attr in bridge_keys:
            setattr(provider, attr, args_dict[attr])
    
    for attr in args_dict:
        if attr not in vars(provider):
            setattr(provider, attr, args_dict[attr])
    

    if args.fp16:
        provider.fp16, provider.bf16 = True, False
        provider.params_dtype = torch.float16
    elif args.bf16:
        provider.fp16, provider.bf16 = False, True
        provider.params_dtype = torch.bfloat16

    provider.finalize()

    def wrap_model_provider(original_provider):
        def wrapped_provider(pre_process=True, post_process=True, vp_stage=None, **kwargs):
            import inspect
            sig = inspect.signature(original_provider)
            if "vp_stage" in sig.parameters:
                return original_provider(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
            return original_provider(pre_process=pre_process, post_process=post_process)
        return wrapped_provider

    model = get_model(wrap_model_provider(provider.provide), ModelType.encoder_or_decoder, wrap_with_ddp=True)
    return model, bridge


# ---------------------------------------------------------------------------
# 4. Weight loading
# ---------------------------------------------------------------------------

@contextmanager
def _patch_megatron_model(model):
    from megatron.core.utils import unwrap_model
    unwrapped_model = unwrap_model(model)[0]
    model_config = unwrapped_model.config
    attribute_was_added = False
    if not hasattr(model_config, "share_embeddings_and_output_weights"):
        model_config.share_embeddings_and_output_weights = unwrapped_model.share_embeddings_and_output_weights
        attribute_was_added = True
    try:
        yield
    finally:
        if attribute_was_added:
            delattr(model_config, "share_embeddings_and_output_weights")

@contextmanager
def _patch_scatter_dtype_cast():
    original_scatter = dist.scatter
    def _scatter_with_cast(output, scatter_list=None, **kwargs):
        if scatter_list is not None and output is not None:
            target_dtype = output.dtype
            scatter_list = [t.to(dtype=target_dtype) if t.dtype != target_dtype else t for t in scatter_list]
        return original_scatter(output, scatter_list=scatter_list, **kwargs)
    dist.scatter = _scatter_with_cast
    try:
        yield
    finally:
        dist.scatter = original_scatter

def load_hf_weights(model, args: Namespace, bridge) -> None:
    logger.info(f"[Rank {args.rank}] Loading HF weights from {args.hf_checkpoint}")
    with _patch_megatron_model(model):
        with _patch_scatter_dtype_cast():
            bridge.load_hf_weights(model)
    logger.info(f"[Rank {args.rank}] HF weights loaded successfully")


# ---------------------------------------------------------------------------
# 5. Forward Step & Logp Extraction
# ---------------------------------------------------------------------------

def extract_logp_loss_function(target_tokens):
    def loss_function(output_tensor, non_loss_data=False):
        if non_loss_data:
            return output_tensor, {}
        
        logits = output_tensor.float()
        log_probs_full = F.log_softmax(logits, dim=-1)
        
        # Depending on format: THD (packed) or BSHD (batch, seq, hidden)
        # Let's handle generic format. Target tokens usually shift by 1.
        # Suppose input is [seq_len, batch_size] or [seq_len * batch_size]
        
        logger.info(f"Logits shape: {logits.shape}, target tokens shape: {target_tokens.shape}")
        
        # Save logits to file for debug
        rank = dist.get_rank()
        torch.save(logits.mean().detach().cpu(), f"logits_rank_{rank}.pt")
        
        # Extract logp for the given target tokens
        # We'll just return it in a dict
        return logits.mean(), {"logits": logits.mean().detach().cpu(), "log_probs_full": log_probs_full.mean().detach().cpu()}
    return loss_function

def create_forward_step(args, tokens):
    from megatron.core import mpu
    from megatron.core.packed_seq_params import PackedSeqParams

    def forward_step_thd(data_iterator, model, return_schedule_plan=False):
        assert not return_schedule_plan
        
        pad_token_id = 0
        pad_size = mpu.get_tensor_model_parallel_world_size() * 128

        cu_seqlens = [0, tokens.size(0)]
        pad = (pad_size - tokens.size(0) % pad_size) % pad_size
        if pad > 0:
            padded_tokens = F.pad(tokens, (0, pad), value=pad_token_id)
            cu_seqlens.append(cu_seqlens[-1] + pad)
        else:
            padded_tokens = tokens

        cu_seqlens_t = to_device(torch.tensor(cu_seqlens, dtype=torch.int))
        max_seqlen = (cu_seqlens_t[1:] - cu_seqlens_t[:-1]).max().item()

        packed_seq_params = PackedSeqParams(
            cu_seqlens_q=cu_seqlens_t, cu_seqlens_kv=cu_seqlens_t,
            max_seqlen_q=max_seqlen, max_seqlen_kv=max_seqlen,
            qkv_format="thd",
        )

        input_ids = to_device(padded_tokens.unsqueeze(0))
        
        # Target tokens for logp extraction
        target_tokens = tokens.clone()

        output_tensor = model(
            input_ids=input_ids, position_ids=None, attention_mask=None,
            labels=None, packed_seq_params=packed_seq_params,
        )

        return output_tensor, extract_logp_loss_function(target_tokens)

    def forward_step_bshd(data_iterator, model, return_schedule_plan=False):
        assert not return_schedule_plan
        
        pad_token_id = 0
        max_seqlen = args.seq_length
        pad = max_seqlen - tokens.size(0)
        
        if pad > 0:
            padded_tokens = F.pad(tokens, (0, pad), value=pad_token_id)
        else:
            padded_tokens = tokens[:max_seqlen]

        # Shape [batch=1, seq_len]
        input_ids = to_device(padded_tokens.unsqueeze(0))
        
        # position_ids [batch=1, seq_len]
        position_ids = torch.arange(max_seqlen, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0)
        
        # Target tokens for logp extraction
        target_tokens = tokens.clone()

        output_tensor = model(
            input_ids=input_ids, position_ids=position_ids, attention_mask=None,
            labels=None, packed_seq_params=None,
        )

        return output_tensor, extract_logp_loss_function(target_tokens)
    return forward_step_thd if getattr(args, "qkv_format", "bshd") == "thd" else forward_step_bshd


def run_test():
    args = parse_test_args()
    try:
        # NPU patch
        from mindspeed.megatron_adaptor import repatch
    except ImportError:
        repatch = None

    if repatch is not None:
        repatch(args)
    init_distributed(args)
    
    model, bridge = build_model_with_bridge(args)
    load_hf_weights(model, args, bridge)
    
    for m in model:
        m.eval()
    
    # Parse test tokens
    token_ids = [int(x.strip()) for x in args.test_tokens.split(",")]
    tokens = torch.tensor(token_ids, dtype=torch.long)
    logger.info(f"[Rank {args.rank}] Input tokens: {tokens}")
    
    from megatron.core.pipeline_parallel import get_forward_backward_func
    forward_backward_func = get_forward_backward_func()
    forward_step = create_forward_step(args, tokens)
    
    with torch.no_grad():
        output = forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=[None],
            model=model,
            num_microbatches=1,
            seq_length=args.seq_length,
            micro_batch_size=int(args.micro_batch_size),
            forward_only=True,
        )
    
    logger.info(f"[Rank {args.rank}] Forward pass completed. Output: {output}")
    return 0

if __name__ == "__main__":
    sys.exit(run_test())
