# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm import LLM, SamplingParams
from vllm.inputs.data import TokensPrompt
import os

# Sample prompts.
prompts = [
    # "Hello",
    "Hello, who are you?"
]
# from msprobe.pytorch import seed_all, PrecisionDebugger
# seed_all()
# debugger = PrecisionDebugger(config_path="/sfs_turbo/hw/hym/deepseekV4/20260525/dump")
sampling_params = SamplingParams(temperature=0.0, max_tokens=512)
def main():
    llm = LLM(
        model="/storage/models/DeepSeek-V4-Flash-BF16_hym_cut_layer",
        trust_remote_code=True,
        enable_prefix_caching=False,
        enforce_eager=True,
        tensor_parallel_size=4,
        mamba_cache_dtype='float32',
        max_num_seqs=16,
        gpu_memory_utilization=0.65,
        enable_return_routed_experts=True)

    outputs = llm.generate(prompts, sampling_params)
    print("\nGenerated Outputs:\n" + "-" * 60)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt:    {prompt!r}")
        print(f"Output:    {generated_text!r}")
        print("-" * 60)


if __name__ == "__main__":
    main()
