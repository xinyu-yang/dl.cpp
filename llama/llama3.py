import os, sys
from pathlib import Path
import tiktoken
from tiktoken.load import load_tiktoken_bpe
import torch
import json
import matplotlib.pyplot as plt
import logging
sys.path.append(os.path.join(sys.path[0], ".."))
from utils import logger

Logger = logger.get_logger(__name__, __file__, level=logging.DEBUG)


class Llama3Inference:
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.count = 0
        # Load config first
        self._load_config(self.base_path.joinpath("params.json").as_posix())
        self._load_model(self.base_path.joinpath("consolidated.00.pth").as_posix())
        self._load_tokenizer(self.base_path.joinpath("tokenizer.model").as_posix())
        self._load_embedding_layer()
        self._precompute_RPE()


    def _load_tokenizer(self, tokenizer_path):
        Logger.info(f"Loading Tokenizer from {tokenizer_path} ...")
        special_tokens = [
                    "<|begin_of_text|>",
                    "<|end_of_text|>",
                    "<|reserved_special_token_0|>",
                    "<|reserved_special_token_1|>",
                    "<|reserved_special_token_2|>",
                    "<|reserved_special_token_3|>",
                    "<|start_header_id|>",
                    "<|end_header_id|>",
                    "<|reserved_special_token_4|>",
                    "<|eot_id|>",  # end of turn
                ] + [f"<|reserved_special_token_{i}|>" for i in range(5, 256 - 5)]

        mergeable_ranks = load_tiktoken_bpe(tokenizer_path)

        self.tokenizer = tiktoken.Encoding(
                    name=Path(tokenizer_path).name,
                    pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
                    mergeable_ranks=mergeable_ranks,
                    special_tokens={token: len(mergeable_ranks) + i for i, token in enumerate(special_tokens)},
                )

        self.idx_of_first_special_token = len(mergeable_ranks)


    def _load_config(self, path):
        Logger.info(f"Loading configs from {path}")
        with open(path, mode="r") as config_file:
            self.config = json.load(config_file)
            Logger.debug(f"Config: {self.config}")


    def _load_model(self, model_path):
        Logger.info(f"Loading model from {model_path}")
        self.model = torch.load(model_path)
        self.qk_dim = self.model["layers.0.attention.wq.weight"].shape[0] // self.config["n_heads"]
        self.v_dim = self.model["layers.0.attention.wv.weight"].shape[0] // self.config["n_kv_heads"]
        self.dtype = self.model["layers.0.attention_norm.weight"].dtype
        Logger.debug(f"qk_dim: {self.qk_dim}, v_dim: {self.v_dim}, dtype: {self.dtype}")
        # Grouped Query Attention
        self.GQA_ratio = self.config["n_heads"] // self.config["n_kv_heads"]
        for k in list(self.model.keys())[:]:
            value = self.model.get(k)
            Logger.debug(f"{k}: {value.shape}")


    # Currently, the embedding layer uses torch.nn.Embedding and requires init
    def _load_embedding_layer(self):
        Logger.info("Loading embedding layer ...")
        self.embedding_layer = torch.nn.Embedding(self.config["vocab_size"], self.config["dim"])
        self.embedding_layer.weight.data.copy_(self.model["tok_embeddings.weight"])
        return


    # Precompute 10000 position encodings
    def _precompute_RPE(self):
        Logger.info("Precomputing Rotaryt Positional Encoding ...")
        zero_to_one_splits = torch.tensor(range(self.qk_dim // 2)) / (self.qk_dim // 2)
        rope_theta = torch.tensor(self.config["rope_theta"])
        freqs = 1.0 / (rope_theta ** zero_to_one_splits)
        freqs = torch.outer(torch.arange(10000), freqs)
        self.freq_cis = torch.polar(torch.ones_like(freqs), freqs)
        Logger.debug(f"The shape of freqs_cis: {self.freq_cis.shape}")
        return


    def tokenization_layer(self, inputs):
        Logger.debug(f"Tokenizing '{inputs}'")
        tokens = [self.idx_of_first_special_token] + self.tokenizer.encode(inputs)
        Logger.debug(f"Tokenized to '{tokens}'")
        return torch.tensor(tokens, dtype=torch.int32) # The input of embedding should be Int/Long


    # Root Mean Square layer
    def RMS_layer(self, embeddings, norm_weights):
        Logger.debug(f"RMS Layer input size: {embeddings.shape}")
        norm_eps = self.config["norm_eps"]
        rms = torch.rsqrt(embeddings.pow(2).mean(-1, keepdim=True) + norm_eps)
        normed_embeds = embeddings * rms * norm_weights
        Logger.debug(f"RMS Layer output size: {normed_embeds.shape}")
        return normed_embeds


    # Rotary Positional Embedding layer
    def RPE_layer(self, inputs: torch.Tensor):
        Logger.debug(f"RPE Layer input size: {inputs.shape}")
        input_pairs = inputs.float().view(inputs.shape[0], -1, 2)
        assert input_pairs.shape[1] == self.qk_dim // 2, f"input pairs shape {input_pairs.shape} does not comply qk_dim {self.qk_dim}"
        input_complex = torch.view_as_complex(input_pairs)
        input_rotated = input_complex * self.freq_cis[:input_pairs.shape[0]]
        outputs = torch.view_as_real(input_rotated)
        outputs = outputs.view(inputs.shape)
        Logger.debug(f"RPE Layer output size: {outputs.shape}")
        return outputs


    def attention_layer(self, normed_inputs, q_layer, k_layer, v_layer, o_layer):
        q_layer = q_layer.view(self.config["n_heads"], self.qk_dim, self.config["dim"])
        Logger.debug(f"q_layer shape: {q_layer.shape}")

        k_layer = k_layer.view(self.config["n_kv_heads"], self.qk_dim, self.config["dim"])
        Logger.debug(f"k_layer shape: {k_layer.shape}")

        v_layer = v_layer.view(self.config["n_kv_heads"], self.v_dim, self.config["dim"])
        Logger.debug(f"v_layer shape: {v_layer.shape}")

        qkv_outputs = []
        for idx in range(self.config["n_heads"]):
            q_layer_head = q_layer[idx]
            k_layer_head = k_layer[idx // self.GQA_ratio]
            v_layer_head = v_layer[idx // self.GQA_ratio]

            q_value = torch.matmul(normed_inputs, q_layer_head.T)
            k_value = torch.matmul(normed_inputs, k_layer_head.T)
            v_value = torch.matmul(normed_inputs, v_layer_head.T)

            rotated_q_value = self.RPE_layer(q_value)
            rotated_k_value = self.RPE_layer(k_value)

            qk_value = torch.matmul(rotated_q_value, rotated_k_value.T) / (self.qk_dim ** 0.5)
            Logger.debug(f"qk_value shape: {qk_value.shape}")

            mask = torch.full(qk_value.shape, float("-inf"))
            mask = torch.triu(mask, diagonal=1) # Upper diagonal

            masked_qk_value = qk_value + mask

            softmax_qk_value = torch.nn.functional.softmax(masked_qk_value, dim=1).to(self.dtype)

            qkv_value = torch.matmul(softmax_qk_value, v_value)
            Logger.debug(f"qkv_value shape: {qkv_value.shape}")

            qkv_outputs.append(qkv_value)

        multihead_output = torch.cat(qkv_outputs, dim=-1)
        Logger.debug(f"The shape of multi-head layer outputs: {multihead_output.shape}")

        # wo layer
        Logger.debug(f"o layer shape: {o_layer.shape}")
        attn_output = torch.matmul(multihead_output, o_layer.T)
        Logger.debug(f"attention layer output: {attn_output.shape}")

        return attn_output


    # Feed Forward layer
    def FFN_layer(self, inputs, w1, w2, w3):
        ffn_outputs = torch.matmul(
                torch.functional.F.silu(torch.matmul(inputs, w1.T))
                * torch.matmul(inputs, w3.T), w2.T)
        Logger.debug(f"Outputs of ffn layer: {ffn_outputs.shape}")
        return ffn_outputs


    def predict(self, inputs):
        # Tokenize
        tokens = self.tokenization_layer(inputs)
        # Embedding
        embeds = self.embedding_layer(tokens).to(self.dtype)
        for layer_idx in range(self.config["n_layers"]):
            Logger.debug(f"The {layer_idx} layer:")
            # RMS layer
            attn_norm_weights = self.model[f"layers.{layer_idx}.attention_norm.weight"]
            attn_norm = self.RMS_layer(embeds, attn_norm_weights)
            # Grouped multi-head attention
            q_layer = self.model[f"layers.{layer_idx}.attention.wq.weight"]
            k_layer = self.model[f"layers.{layer_idx}.attention.wk.weight"]
            v_layer = self.model[f"layers.{layer_idx}.attention.wv.weight"]
            o_layer = self.model[f"layers.{layer_idx}.attention.wo.weight"]
            attn_output = self.attention_layer(attn_norm, q_layer, k_layer, v_layer, o_layer)
            # Add attention outputs with embeddings
            combined_attn_embeds = embeds + attn_output
            # RMS layer
            ffn_norm_weights = self.model[f"layers.{layer_idx}.ffn_norm.weight"]
            ffn_norm = self.RMS_layer(combined_attn_embeds, ffn_norm_weights)
            # FFN layer
            w1 = self.model[f"layers.{layer_idx}.feed_forward.w1.weight"]
            w2 = self.model[f"layers.{layer_idx}.feed_forward.w2.weight"]
            w3 = self.model[f"layers.{layer_idx}.feed_forward.w3.weight"]
            ffn_outputs = self.FFN_layer(ffn_norm, w1, w2, w3)
            # Add attnetion outputs with embeddings and ffn outputs
            combined_outputs = ffn_outputs + combined_attn_embeds
            # For looping
            embeds = combined_outputs

        # Output RMS layer
        output_norm = self.RMS_layer(embeds, self.model["norm.weight"])
        # Output logits
        logits = torch.matmul(output_norm[-1], self.model["output.weight"].T)
        output_token = torch.argmax(logits, dim=-1)
        Logger.debug(f"Output token {output_token}")
        output_word = self.tokenizer.decode([output_token.item()])
        return output_word

def main(base_path):
    inputs = "The answer to the ultimate question of life, the universe, and everything is "
    BASE_PATH = Path(base_path)
    infer = Llama3Inference(BASE_PATH)
    output = infer.predict(inputs)
    print(f"Output is: '{output}'")
    print(f"{inputs}{output}")

if __name__ == "__main__":
    main("../../llama3/Meta-Llama-3-8B")
