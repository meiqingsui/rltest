export ASCEND_RT_VISIBLE_DEVICES=12,13,14,15
export SGLANG_SET_CPU_AFFINITY=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export HCCL_BUFFSIZE=1536
export HCCL_OP_EXPANSION_MODE=AIV


python -m sglang.launch_server \
   --device npu \
   --enable-multimodal \
   --attention-backend ascend \
   --mm-attention-backend ascend_attn \
   --trust-remote-code \
   --tp-size 4 \
   --model-path /mnt/sfs_turbo/models/Qwen3.5-9B \
   --port 30000 \
   --mem-fraction-static 0.8 \
   --max-total-tokens 10240 \
   --disable-radix-cache