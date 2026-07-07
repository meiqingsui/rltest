#export ASDOPS_HOME_PATH=/storage/hw/lanzeshun/Ascend/nnal/atb/latest/atb/cxx_abi_1
export VLLM_USE_V1=1
export VLLM_VERSION=0.18.0
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=380
export NET_CARD_NAME="eth0"
export VLLM_ASCEND_ENABLE_NZ=0
export HCCL_BUFFSIZE=300
export ASCEND_BUFFER_POOL="4:8"
export USE_MULTI_GROUPS_KV_CACHE=1
export USE_MULTI_BLOCK_POOL=1
env
python ./offline.py
