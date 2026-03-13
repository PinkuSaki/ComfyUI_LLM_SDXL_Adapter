import torch
import logging

from .prompt_parser import build_weighted_character_map

logger = logging.getLogger("LLM-SDXL-Adapter")


class T5GEMMATextEncoder:
    """
    ComfyUI node that encodes text using a loaded Language Model
    Supports various LLM architectures with chat templates
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_model": ("LLM_MODEL",),
                "llm_tokenizer": ("LLM_TOKENIZER",),
                "text": ("STRING", {"multiline": True, "default": "masterpiece, best quality, 1girl, anime style"}),
                "max_length": ("INT", {"default": 512, "min": 8, "max": 4096}),
                "device": (["cpu", "cuda"], {"default": "cuda"}),
                "dtype": (["float32", "bfloat16"], {"default": "bfloat16"}),
                "emphasis": (["disabled", "scale", "lerp"], {"default": "scale"}),
            }
        }

    RETURN_TYPES = ("LLM_HIDDEN_STATES", "LLM_ATTENTION_MASK", "STRING", "LLM_TOKEN_WEIGHTS")
    RETURN_NAMES = ("hidden_states", "attention_mask", "info", "token_weights")
    FUNCTION = "encode_text"
    CATEGORY = "llm_sdxl"

    def _build_token_weights_from_offsets(self, offsets, attention_mask, char_weights, device):
        """根据 tokenizer 的 offset mapping，把字符权重映射到 token 权重。"""
        token_weights = torch.ones((1, len(offsets)), dtype=torch.float32, device=device)

        for token_index, (start, end) in enumerate(offsets):
            if attention_mask[token_index].item() == 0:
                continue

            if end <= start or start >= len(char_weights):
                continue

            bounded_end = min(end, len(char_weights))
            token_char_weights = char_weights[start:bounded_end]
            if token_char_weights:
                token_weights[0, token_index] = sum(token_char_weights) / len(token_char_weights)

        return token_weights

    def _build_weighted_tokens(self, tokenizer, text, max_length, device):
        """
        解析 prompt 权重语法后，保持整串 tokenize，不破坏原始 token 序列。
        """
        plain_text, char_weights = build_weighted_character_map(text)
        encoded_text = plain_text + "<eos>"
        char_weights.extend([1.0] * len("<eos>"))

        try:
            inputs = tokenizer(
                encoded_text,
                return_tensors="pt",
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_offsets_mapping=True,
            )
        except (NotImplementedError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "当前 tokenizer 不支持 offset mapping，无法在保持原始 tokenization 的前提下启用 emphasis。"
            ) from exc

        offsets = inputs.pop("offset_mapping")[0].tolist()
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        token_weights = self._build_token_weights_from_offsets(
            offsets, attention_mask[0], char_weights, device
        )

        return input_ids, attention_mask, token_weights

    def encode_text(self, llm_model, llm_tokenizer, text, max_length, device, dtype, emphasis="scale"):
        """
        Encode text using Language Model and return hidden states
        """
        try:
            if emphasis == "disabled":
                # 原始路径：hidden states 不做 emphasis，仅输出全 1 token 权重。
                inputs = llm_tokenizer(
                    text + "<eos>",
                    return_tensors="pt",
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                )
                input_ids = inputs.input_ids.to(device)
                attention_mask = inputs.attention_mask.to(device)
                token_weights = torch.ones_like(input_ids, dtype=torch.float32, device=device)

                with torch.no_grad():
                    outputs = llm_model(input_ids=input_ids, attention_mask=attention_mask)
                    hidden_states = outputs.last_hidden_state.to(torch.float32)
            else:
                # 兼容旧工作流里的 scale / lerp 选项，但实际权重交给 adapter attention 处理。
                input_ids, attention_mask, token_weights = self._build_weighted_tokens(
                    llm_tokenizer, text, max_length, device
                )

                with torch.no_grad():
                    outputs = llm_model(input_ids=input_ids, attention_mask=attention_mask)
                    hidden_states = outputs.last_hidden_state.to(torch.float32)

            info = f"Text: {text[:50]}...\nEncoded: {hidden_states.shape[1]}\nShape: {hidden_states.shape}"
            if emphasis != "disabled":
                info += f"\nEmphasis: adapter_attention ({emphasis})"

            logger.info(f"Encoded text with shape: {hidden_states.shape}")

            return (hidden_states, attention_mask, info, token_weights)
        except Exception as e:
            logger.error(f"Failed to encode text: {str(e)}")
            raise Exception(f"Text encoding failed: {str(e)}")


# Node mapping for ComfyUI registration
NODE_CLASS_MAPPINGS = {
    "T5GEMMATextEncoder": T5GEMMATextEncoder
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "T5GEMMATextEncoder": "T5Gemma Text Encoder"
}
