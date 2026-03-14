import gc
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
                "enable_weighted_prompt": ("BOOLEAN", {"default": True}),
                "max_length": ("INT", {"default": 512, "min": 8, "max": 4096}),
                "device": (["cpu", "cuda"], {"default": "cuda"}),
                "dtype": (["float32", "bfloat16"], {"default": "bfloat16"}),
                "offload_after_encode": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = (
        "LLM_HIDDEN_STATES",
        "LLM_ATTENTION_MASK",
        "STRING",
        "LLM_TOKEN_WEIGHTS",
        "LLM_EMPTY_HIDDEN_STATES",
        "LLM_EMPTY_ATTENTION_MASK",
    )
    RETURN_NAMES = (
        "hidden_states",
        "attention_mask",
        "info",
        "token_weights",
        "empty_prompt_hidden_states",
        "empty_prompt_attention_mask",
    )
    FUNCTION = "encode_text"
    CATEGORY = "llm_sdxl"

    def _model_is_on_device(self, model, target_device):
        try:
            model_device = next(model.parameters()).device
        except StopIteration:
            return False

        target = torch.device(target_device)
        if target.type == "cuda":
            return model_device.type == "cuda" and (
                target.index is None or model_device.index == target.index
            )

        return model_device == target

    def _move_model_to_device(self, model, target_device):
        if self._model_is_on_device(model, target_device):
            return

        model.to(target_device)

    def _offload_model_to_cpu(self, model):
        if self._model_is_on_device(model, "cpu"):
            return

        model.to("cpu")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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

    def _has_nontrivial_weights(self, weights, attention_mask=None, atol=1e-6):
        """仅在有效 token 上存在非 1.0 权重时，才启用加权路径。"""
        if weights is None:
            return False

        if attention_mask is not None:
            valid_weights = weights[attention_mask > 0]
        else:
            valid_weights = weights.reshape(-1)

        if valid_weights.numel() == 0:
            return False

        return not torch.allclose(
            valid_weights,
            torch.ones_like(valid_weights),
            atol=atol,
            rtol=0.0,
        )

    def _tokenize_text(self, tokenizer, encoded_text, max_length, device, return_offsets_mapping=False):
        tokenizer_kwargs = {
            "return_tensors": "pt",
            "padding": "max_length",
            "max_length": max_length,
            "truncation": True,
        }
        if return_offsets_mapping:
            tokenizer_kwargs["return_offsets_mapping"] = True

        inputs = tokenizer(encoded_text, **tokenizer_kwargs)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        return inputs, input_ids, attention_mask

    def _build_raw_tokens(self, tokenizer, text, max_length, device):
        _, input_ids, attention_mask = self._tokenize_text(
            tokenizer,
            text + "<eos>",
            max_length,
            device,
        )
        return input_ids, attention_mask

    def _build_weighted_tokens(self, tokenizer, text, max_length, device):
        """
        解析 prompt 权重语法后，保持整串 tokenize，不破坏原始 token 序列。
        """
        plain_text, char_weights = build_weighted_character_map(text)
        encoded_text = plain_text + "<eos>"

        if not self._has_nontrivial_weights(
            torch.tensor(char_weights, dtype=torch.float32)
        ):
            _, input_ids, attention_mask = self._tokenize_text(
                tokenizer,
                encoded_text,
                max_length,
                device,
            )
            return input_ids, attention_mask, None, False

        char_weights = char_weights + [1.0] * len("<eos>")

        try:
            inputs, input_ids, attention_mask = self._tokenize_text(
                tokenizer,
                encoded_text,
                max_length,
                device,
                return_offsets_mapping=True,
            )
        except (NotImplementedError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "当前 tokenizer 不支持 offset mapping，无法在保持原始 tokenization 的前提下处理加权 prompt。"
            ) from exc

        offsets = inputs.pop("offset_mapping")[0].tolist()
        token_weights = self._build_token_weights_from_offsets(
            offsets, attention_mask[0], char_weights, device
        )

        return input_ids, attention_mask, token_weights, True

    def _build_empty_prompt_inputs(self, tokenizer, max_length, device):
        """构造 empty-prompt 基线编码输入。"""
        inputs = tokenizer(
            "<eos>",
            return_tensors="pt",
            padding="max_length",
            max_length=max_length,
            truncation=True,
        )

        return inputs.input_ids.to(device), inputs.attention_mask.to(device)

    def _encode_hidden_states(self, model, input_ids, attention_mask):
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            return outputs.last_hidden_state.to(torch.float32)

    def encode_text(self, llm_model, llm_tokenizer, text, enable_weighted_prompt, max_length, device, dtype, offload_after_encode):
        """
        Encode text using Language Model and return hidden states
        """
        try:
            self._move_model_to_device(llm_model, device)

            if enable_weighted_prompt:
                input_ids, attention_mask, token_weights, use_weighted_prompt = self._build_weighted_tokens(
                    llm_tokenizer, text, max_length, device
                )
            else:
                input_ids, attention_mask = self._build_raw_tokens(
                    llm_tokenizer, text, max_length, device
                )
                token_weights = None
                use_weighted_prompt = False

            hidden_states = self._encode_hidden_states(llm_model, input_ids, attention_mask)

            empty_hidden_states = None
            empty_attention_mask = None
            if use_weighted_prompt:
                empty_input_ids, empty_attention_mask = self._build_empty_prompt_inputs(
                    llm_tokenizer, max_length, device
                )
                empty_hidden_states = self._encode_hidden_states(
                    llm_model,
                    empty_input_ids,
                    empty_attention_mask,
                )

            info = f"Text: {text[:50]}...\nEncoded: {hidden_states.shape[1]}\nShape: {hidden_states.shape}"
            if not enable_weighted_prompt:
                info += "\nWeighted prompt: disabled (raw tokenizer path)"
            elif use_weighted_prompt:
                info += "\nWeighted prompt: compressed_sequence_lerp (empty_prompt baseline)"
            else:
                info += "\nWeighted prompt: passthrough (single encoder pass)"
            if offload_after_encode:
                self._offload_model_to_cpu(llm_model)
                info += "\nModel offload: moved to CPU after encode"

            logger.info(f"Encoded text with shape: {hidden_states.shape}")

            return (
                hidden_states,
                attention_mask,
                info,
                token_weights,
                empty_hidden_states,
                empty_attention_mask,
            )
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
