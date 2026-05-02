"""
Thin VLLM wrapper. Loaded once, called as a pure function from rollout.

Input per call: a list of chat-message lists (each a list of {role, content}).
Output: a list of completion strings.

Optional LoRA support: pass ``enable_lora=True`` and ``max_lora_rank`` at
construction; then call ``set_lora(path)`` to point subsequent generates at a
specific LoRA adapter directory.  Re-pointing to a new path is cheap (no
model reload).
"""
from __future__ import annotations

from typing import List, Dict, Optional

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


class ChatLLM:
    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-7B-Instruct",
        max_model_len: int = 16384,
        gpu_memory_utilization: float = 0.85,
        enable_prefix_caching: bool = True,
        dtype: str = "bfloat16",
        enable_lora: bool = False,
        max_lora_rank: int = 16,
        max_loras: int = 1,
    ):
        self.model = model
        self.enable_lora = enable_lora
        kwargs = dict(
            model=model,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_prefix_caching=enable_prefix_caching,
            max_model_len=max_model_len,
            tensor_parallel_size=1,
        )
        if enable_lora:
            kwargs.update(enable_lora=True, max_loras=max_loras,
                          max_lora_rank=max_lora_rank)
        self.llm = LLM(**kwargs)
        self.tokenizer = self.llm.get_tokenizer()
        self._lora_request: Optional[LoRARequest] = None
        self._lora_counter = 0

    def set_lora(self, path: Optional[str]) -> None:
        """Point all future generate() calls at this LoRA adapter directory.
        Pass None or empty to revert to the base model.

        Each set_lora() call increments a counter so VLLM treats it as a
        distinct adapter (this is required by VLLM's caching).
        """
        if not self.enable_lora:
            raise RuntimeError("ChatLLM was constructed with enable_lora=False")
        if not path:
            self._lora_request = None
            return
        self._lora_counter += 1
        self._lora_request = LoRARequest(
            lora_name=f"adapter_{self._lora_counter}",
            lora_int_id=self._lora_counter,
            lora_path=path,
        )

    def generate(
        self,
        chats: List[List[Dict[str, str]]],
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_new_tokens: int = 128,
        stop: Optional[List[str]] = None,
        seed: Optional[int] = None,
    ) -> List[str]:
        """Each element of `chats` becomes its own request, batched through VLLM."""
        prompts = [
            self.tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            )
            for chat in chats
        ]
        params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            stop=stop or ["\nObservation:", "\nObservation ", "<|im_end|>"],
            seed=seed,
        )
        kwargs = {}
        if self._lora_request is not None:
            kwargs["lora_request"] = self._lora_request
        outs = self.llm.generate(prompts, params, use_tqdm=False, **kwargs)
        return [o.outputs[0].text for o in outs]
