import glob
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from huggingface_hub import snapshot_download

from .language import LanguageModel, TextConfig
from .vision import VisionConfig, VisionModel


@dataclass
class ModelConfig:
    text_config: TextConfig
    vision_config: VisionConfig
    model_type: str
    ignore_index: int = -100
    image_token_id: int = 151655
    video_token_id: int = 151656
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    vision_token_id: int = 151654
    vision_feature_select_strategy: str = "default"
    vision_feature_layer: int = -2
    vocab_size: int = 32000
    eos_token_id: Optional[List[int]] = None

    @classmethod
    def from_dict(cls, params):
        # Copy text config parameters from root level
        excluded_keys = {"vision_config"}
        params["text_config"] = dict(
            filter(lambda x: x[0] not in excluded_keys, params.items())
        )

        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vision_tower = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config)

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        image_grid_thw: Optional[mx.array] = None,
    ):
        if pixel_values is None:
            return self.language_model.model.embed_tokens(input_ids)

        dtype = self.vision_tower.patch_embed.proj.weight.dtype
        pixel_values = pixel_values.astype(dtype)

        # Get the input embeddings from the language model
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)

        # Get the ouptut hidden states from the vision model
        hidden_states = self.vision_tower(
            pixel_values, image_grid_thw, output_hidden_states=False
        )

        if hidden_states.ndim == 2:
            hidden_states = hidden_states[None, :, :]

        # Insert special image tokens in the input_ids
        final_inputs_embeds = self.merge_input_ids_with_image_features(
            self.config.image_token_id,
            self.config.video_token_id,
            hidden_states,
            inputs_embeds,
            input_ids,
        )
        return final_inputs_embeds

    @staticmethod
    def merge_input_ids_with_image_features(
        image_token_id, video_token_id, image_features, inputs_embeds, input_ids
    ):
        image_token_id = image_token_id
        video_token_id = video_token_id
        # Positions of <image> tokens in input_ids, assuming batch size is 1
        image_positions = input_ids == image_token_id
        if mx.sum(image_positions) == 0:
            image_positions = input_ids == video_token_id

        image_indices = np.where(image_positions)[1].tolist()
        inputs_embeds[:, image_indices, :] = image_features
        return inputs_embeds

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: mx.array,
        mask: mx.array,
        cache=None,
        **kwargs,
    ):
        image_grid_thw = kwargs.pop("image_grid_thw", None)
        second_per_grid_ts = kwargs.pop("second_per_grid_ts", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)
        position_ids = kwargs.pop("position_ids", None)
        grid_thw = image_grid_thw if image_grid_thw is not None else video_grid_thw

        inputs_embeds = self.get_input_embeddings(input_ids, pixel_values, grid_thw)

        logits = self.language_model(None, cache=cache, inputs_embeds=inputs_embeds)
        return logits

    @staticmethod
    def from_pretrained(path_or_hf_repo: str):
        path = Path(path_or_hf_repo)
        if not path.exists():
            path = Path(
                snapshot_download(
                    repo_id=path_or_hf_repo,
                    allow_patterns=[
                        "*.json",
                        "*.safetensors",
                        "*.py",
                        "tokenizer.model",
                        "*.tiktoken",
                    ],
                )
            )

        with open(path / "config.json", "r") as f:
            model_config = json.load(f)

        model_config = ModelConfig.from_dict(model_config)

        model_config.vision_config = VisionConfig.from_dict(model_config.vision_config)
        model_config.text_config = TextConfig.from_dict(model_config)

        model = Model(model_config)
        weight_files = glob.glob(str(path / "*.safetensors"))
        if not weight_files:
            raise FileNotFoundError(f"No safetensors found in {path}")

        weights = {}
        for wf in weight_files:
            weights.update(mx.load(wf))

        weights = VisionModel.sanitize(weights)
        weights = LanguageModel.sanitize(weights)

        model.load_weights(list(weights.items()))
        return model

    def sanitize(self, weights):
        def transform_key(key):
            if "vision_tower" not in key:
                key = key.replace("visual", "vision_tower")
            if "language_model" not in key:
                if "model" in key:
                    key = key.replace("model", "language_model.model")
                elif "lm_head" in key:
                    key = key.replace("lm_head", "language_model.lm_head")
            return key

        return {transform_key(k): v for k, v in weights.items()}
