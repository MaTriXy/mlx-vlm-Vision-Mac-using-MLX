import unittest

import mlx.core as mx
from mlx.utils import tree_map


class TestModels(unittest.TestCase):

    def language_test_runner(self, model, model_type, vocab_size, num_layers):
        self.assertEqual(model.model_type, model_type)
        self.assertEqual(len(model.layers), num_layers)

        batch_size = 1

        for t in [mx.float32, mx.float16]:
            model.update(tree_map(lambda p: p.astype(t), model.parameters()))

            inputs = mx.array([[0, 1]])
            outputs = model(inputs)
            self.assertEqual(outputs.shape, (batch_size, 2, vocab_size))
            self.assertEqual(outputs.dtype, t)

            outputs = model(mx.argmax(outputs[0, -1:, :], keepdims=True), cache=None)
            self.assertEqual(outputs.shape, (batch_size, 1, vocab_size))
            self.assertEqual(outputs.dtype, t)

    def mm_projector_test_runner(
        self, mm_projector, vision_hidden_size, text_hidden_size
    ):

        batch_size = 1

        for t in [mx.float32, mx.float16]:
            mm_projector.update(
                tree_map(lambda p: p.astype(t), mm_projector.parameters())
            )

            vision_features = mx.random.uniform(
                shape=(batch_size, vision_hidden_size), dtype=t
            )
            input_tensor = mx.array(vision_features)

            outputs = mm_projector(input_tensor)
            self.assertEqual(outputs.shape, (batch_size, text_hidden_size))
            self.assertEqual(outputs.dtype, t)

    def vision_test_runner(
        self,
        vision_tower,
        model_type,
        vision_hidden_size,
        num_channels,
        image_size: tuple,
        vision_feature_layer=-2,
        **kwargs,
    ):
        self.assertEqual(vision_tower.model_type, model_type)

        batch_size = 1

        input_tensor = mx.random.uniform(
            shape=(batch_size, image_size[0], image_size[1], num_channels)
        )

        hidden_states = vision_tower(input_tensor, output_hidden_states=True, **kwargs)

        # Check vision hidden feature layer's shape matches the expected hidden size
        self.assertEqual(
            hidden_states[vision_feature_layer].shape[-1], vision_hidden_size
        )

    def test_llava_bunny(self):
        from mlx_vlm.models import llava_bunny

        text_config = llava_bunny.TextConfig(
            model_type="qwen2",
            hidden_size=4096,
            num_hidden_layers=32,
            intermediate_size=11008,
            num_attention_heads=32,
            rms_norm_eps=1e-6,
            vocab_size=32000,
            num_key_value_heads=32,
            rope_theta=10000.0,
            rope_traditional=False,
            rope_scaling=None,
        )

        vision_config = llava_bunny.VisionConfig(
            model_type="siglip_vision_model",
            num_hidden_layers=27,
            hidden_size=1152,
            intermediate_size=4304,
            num_attention_heads=16,
            image_size=384,
            patch_size=14,
            projection_dim=768,
            vocab_size=32000,
            num_channels=3,
            layer_norm_eps=1e-6,
        )

        config = llava_bunny.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            model_type="llava-qwen2",
            auto_map={
                "AutoConfig": "configuration_llava_qwen2.LlavaQwen2Config",
                "AutoModelForCausalLM": "modeling_llava_qwen2.LlavaQwen2ForCausalLM",
            },
            hidden_size=1024,
            mm_hidden_size=1152,
            mm_vision_tower="google/siglip-so400m-patch14-384",
            mm_projector_type="mlp2x_gelu",
            ignore_index=-100,
            image_token_index=-200,
            vocab_size=151936,
        )

        model = llava_bunny.Model(config)

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

        self.mm_projector_test_runner(
            model.mm_projector,
            config.vision_config.hidden_size,
            config.text_config.hidden_size,
        )

        self.vision_test_runner(
            model.vision_tower.vision_tower,
            config.vision_config.model_type,
            config.vision_config.hidden_size,
            config.vision_config.num_channels,
            (config.vision_config.image_size, config.vision_config.image_size),
        )

    def test_llava_next(self):
        from mlx_vlm.models import llava_next

        text_config = llava_next.TextConfig(
            model_type="llama",
            hidden_size=4096,
            num_hidden_layers=32,
            intermediate_size=11008,
            num_attention_heads=32,
            rms_norm_eps=1e-5,
            vocab_size=32000,
            num_key_value_heads=32,
            rope_theta=10000.0,
            rope_traditional=False,
            rope_scaling=None,
        )

        vision_config = llava_next.VisionConfig(
            model_type="clip_vision_model",
            num_hidden_layers=23,
            hidden_size=1024,
            intermediate_size=4096,
            num_attention_heads=16,
            image_size=336,
            patch_size=14,
            projection_dim=768,
            vocab_size=32000,
            num_channels=3,
            layer_norm_eps=1e-6,
        )

        config = llava_next.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            model_type="llava",
            ignore_index=-100,
            image_token_index=32000,
            vocab_size=32000,
            vision_feature_layer=-2,
            vision_feature_select_strategy="default",
        )

        model = llava_next.Model(config)

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

        self.mm_projector_test_runner(
            model.multi_modal_projector,
            config.vision_config.hidden_size,
            config.text_config.hidden_size,
        )

        self.vision_test_runner(
            model.vision_tower,
            config.vision_config.model_type,
            config.vision_config.hidden_size,
            config.vision_config.num_channels,
            (config.vision_config.image_size, config.vision_config.image_size),
        )

    def test_llava(self):
        from mlx_vlm.models import llava

        text_config = llava.TextConfig(
            model_type="llama",
            hidden_size=4096,
            num_hidden_layers=32,
            intermediate_size=11008,
            num_attention_heads=32,
            rms_norm_eps=1e-5,
            vocab_size=32000,
            num_key_value_heads=32,
            rope_theta=10000.0,
            rope_traditional=False,
            rope_scaling=None,
        )

        vision_config = llava.VisionConfig(
            model_type="clip_vision_model",
            num_hidden_layers=23,
            hidden_size=1024,
            intermediate_size=4096,
            num_attention_heads=16,
            image_size=336,
            patch_size=14,
            projection_dim=768,
            vocab_size=32000,
            num_channels=3,
            layer_norm_eps=1e-6,
        )

        config = llava.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            model_type="llava",
            ignore_index=-100,
            image_token_index=32000,
            vocab_size=32000,
            vision_feature_layer=-2,
            vision_feature_select_strategy="default",
        )

        model = llava.Model(config)

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

        self.mm_projector_test_runner(
            model.multi_modal_projector,
            config.vision_config.hidden_size,
            config.text_config.hidden_size,
        )

        self.vision_test_runner(
            model.vision_tower,
            config.vision_config.model_type,
            config.vision_config.hidden_size,
            config.vision_config.num_channels,
            (config.vision_config.image_size, config.vision_config.image_size),
        )

    def test_idefics2(self):
        from mlx_vlm.models import idefics2

        text_config = idefics2.TextConfig(
            model_type="mistral",
            hidden_size=4096,
            num_hidden_layers=32,
            intermediate_size=14336,
            num_attention_heads=32,
            rms_norm_eps=1e-5,
            vocab_size=32000,
            num_key_value_heads=8,
            rope_theta=10000.0,
            rope_traditional=False,
        )

        vision_config = idefics2.VisionConfig(
            model_type="idefics2",
            num_hidden_layers=27,
            hidden_size=1152,
            intermediate_size=4304,
            num_attention_heads=16,
            image_size=980,
            patch_size=14,
            num_channels=3,
            layer_norm_eps=1e-6,
        )

        perceiver_config = idefics2.PerceiverConfig(
            model_type="idefics2Perceiver",
            resampler_n_latents=64,
            resampler_depth=3,
            resampler_n_heads=16,
            resampler_head_dim=96,
            num_key_value_heads=4,
        )

        config = idefics2.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            perceiver_config=perceiver_config,
            model_type="idefics2",
            ignore_index=-100,
            image_token_index=32001,
        )

        model = idefics2.Model(config)

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

        self.vision_test_runner(
            model.vision_model,
            config.vision_config.model_type,
            config.vision_config.hidden_size,
            config.vision_config.num_channels,
            (config.vision_config.image_size, config.vision_config.image_size),
        )

    def test_paligemma(self):
        from mlx_vlm.models import paligemma

        text_config = paligemma.TextConfig(
            model_type="gemma",
            hidden_size=2048,
            num_hidden_layers=18,
            intermediate_size=16384,
            num_attention_heads=8,
            rms_norm_eps=1e-6,
            vocab_size=257216,
            num_key_value_heads=1,
            rope_theta=10000.0,
            rope_traditional=False,
        )

        vision_config = paligemma.VisionConfig(
            model_type="siglip_vision_model",
            num_hidden_layers=27,
            hidden_size=1152,
            intermediate_size=4304,
            num_attention_heads=16,
            image_size=224,
            patch_size=14,
            projection_dim=2048,
            num_channels=3,
            layer_norm_eps=1e-6,
        )

        config = paligemma.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            model_type="paligemma",
            ignore_index=-100,
            image_token_index=257152,
            hidden_size=2048,
            vocab_size=257216,
        )

        model = paligemma.Model(config)

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

        self.mm_projector_test_runner(
            model.multi_modal_projector,
            config.vision_config.hidden_size,
            config.text_config.hidden_size,
        )

        self.vision_test_runner(
            model.vision_tower,
            config.vision_config.model_type,
            config.vision_config.hidden_size,
            config.vision_config.num_channels,
            (config.vision_config.image_size, config.vision_config.image_size),
        )

    def test_multi_modality(self):
        from mlx_vlm.models import multi_modality

        text_config = multi_modality.TextConfig(
            model_type="llama",
            hidden_size=2048,
            num_hidden_layers=24,
            intermediate_size=5632,
            num_attention_heads=16,
            rms_norm_eps=1e-6,
            vocab_size=32000,
            rope_theta=10000.0,
            rope_traditional=False,
            rope_scaling=None,
        )

        vision_config = multi_modality.VisionConfig(
            model_type="vision",
            num_hidden_layers=24,
            hidden_size=1024,
            intermediate_size=4096,
            num_attention_heads=16,
            image_size=384,
            patch_size=14,
            num_channels=3,
            layer_norm_eps=1e-5,
            params={},
        )

        aligner_config = multi_modality.AlignerConfig(
            cls="MlpProjector",
            model_type="aligner",
            params={
                "depth": 2,
                "input_dim": 1024,
                "n_embed": 2048,
                "projector_type": "mlp_gelu",
            },
        )

        config = multi_modality.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            aligner_config=aligner_config,
            model_type="multi_modality",
            ignore_index=-100,
            image_token_index=100015,
            vocab_size=32000,
        )

        model = multi_modality.Model(config)

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

        self.mm_projector_test_runner(
            model.aligner,
            config.vision_config.hidden_size,
            config.text_config.hidden_size,
        )

        self.vision_test_runner(
            model.vision_model,
            config.vision_config.model_type,
            config.vision_config.hidden_size,
            config.vision_config.num_channels,
            (config.vision_config.image_size, config.vision_config.image_size),
        )

    def test_phi3_v(self):
        from mlx_vlm.models import phi3_v

        text_config = phi3_v.TextConfig()

        vision_config = phi3_v.VisionConfig(
            model_type="phi3_v",
            image_dim_out=1024,
            model_name="openai/clip-vit-large-patch14-336",
            name="clip_vision_model",
            num_img_tokens=144,
        )

        config = phi3_v.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            **{
                "hidden_size": 3072,
                "intermediate_size": 8192,
                "max_position_embeddings": 131072,
                "model_type": "phi3_v",
                "num_attention_heads": 32,
                "num_hidden_layers": 32,
                "num_key_value_heads": 32,
                "original_max_position_embeddings": 4096,
                "rms_norm_eps": 1e-05,
                "rope_scaling": {
                    "long_factor": [
                        1.0299999713897705,
                        1.0499999523162842,
                        1.0499999523162842,
                        1.0799999237060547,
                        1.2299998998641968,
                        1.2299998998641968,
                        1.2999999523162842,
                        1.4499999284744263,
                        1.5999999046325684,
                        1.6499998569488525,
                        1.8999998569488525,
                        2.859999895095825,
                        3.68999981880188,
                        5.419999599456787,
                        5.489999771118164,
                        5.489999771118164,
                        9.09000015258789,
                        11.579999923706055,
                        15.65999984741211,
                        15.769999504089355,
                        15.789999961853027,
                        18.360000610351562,
                        21.989999771118164,
                        23.079999923706055,
                        30.009998321533203,
                        32.35000228881836,
                        32.590003967285156,
                        35.56000518798828,
                        39.95000457763672,
                        53.840003967285156,
                        56.20000457763672,
                        57.95000457763672,
                        59.29000473022461,
                        59.77000427246094,
                        59.920005798339844,
                        61.190006256103516,
                        61.96000671386719,
                        62.50000762939453,
                        63.3700065612793,
                        63.48000717163086,
                        63.48000717163086,
                        63.66000747680664,
                        63.850006103515625,
                        64.08000946044922,
                        64.760009765625,
                        64.80001068115234,
                        64.81001281738281,
                        64.81001281738281,
                    ],
                    "short_factor": [
                        1.05,
                        1.05,
                        1.05,
                        1.1,
                        1.1,
                        1.1,
                        1.2500000000000002,
                        1.2500000000000002,
                        1.4000000000000004,
                        1.4500000000000004,
                        1.5500000000000005,
                        1.8500000000000008,
                        1.9000000000000008,
                        2.000000000000001,
                        2.000000000000001,
                        2.000000000000001,
                        2.000000000000001,
                        2.000000000000001,
                        2.000000000000001,
                        2.000000000000001,
                        2.000000000000001,
                        2.000000000000001,
                        2.000000000000001,
                        2.000000000000001,
                        2.000000000000001,
                        2.000000000000001,
                        2.000000000000001,
                        2.000000000000001,
                        2.000000000000001,
                        2.000000000000001,
                        2.000000000000001,
                        2.000000000000001,
                        2.1000000000000005,
                        2.1000000000000005,
                        2.2,
                        2.3499999999999996,
                        2.3499999999999996,
                        2.3499999999999996,
                        2.3499999999999996,
                        2.3999999999999995,
                        2.3999999999999995,
                        2.6499999999999986,
                        2.6999999999999984,
                        2.8999999999999977,
                        2.9499999999999975,
                        3.049999999999997,
                        3.049999999999997,
                        3.049999999999997,
                    ],
                    "type": "su",
                },
                "rope_theta": 10000.0,
                "vocab_size": 32064,
            },
        )

        model = phi3_v.Model(config)

        self.language_test_runner(
            model.language_model,
            config.model_type,
            config.vocab_size,
            config.num_hidden_layers,
        )

        self.vision_test_runner(
            model.vision_model,
            config.vision_config.model_type,
            config.vision_config.hidden_size,
            config.vision_config.num_channels,
            (config.vision_config.image_size, config.vision_config.image_size),
        )

    def test_pixtral(self):
        from mlx_vlm.models import pixtral

        text_config = pixtral.TextConfig(
            model_type="mistral",
            hidden_size=4096,
            num_hidden_layers=32,
            intermediate_size=11008,
            num_attention_heads=32,
            rms_norm_eps=1e-5,
            vocab_size=32000,
            num_key_value_heads=32,
            rope_theta=10000.0,
            rope_traditional=False,
            rope_scaling=None,
        )

        vision_config = pixtral.VisionConfig(
            model_type="pixtral",
            num_hidden_layers=24,
            hidden_size=1024,
            intermediate_size=4096,
            num_attention_heads=16,
            image_size=336,
            patch_size=14,
            projection_dim=768,
            vocab_size=32000,
            num_channels=3,
            rms_norm_eps=1e-6,
        )

        config = pixtral.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            model_type="pixtral",
            ignore_index=-100,
            image_token_index=32000,
            vocab_size=32000,
            vision_feature_layer=-2,
            vision_feature_select_strategy="default",
        )

        model = pixtral.Model(config)

    def test_qwen2_vl(self):
        from mlx_vlm.models import qwen2_vl

        text_config = qwen2_vl.TextConfig(
            model_type="qwen2_vl",
            hidden_size=32,
            num_hidden_layers=4,
            intermediate_size=37,
            num_attention_heads=4,
            rms_norm_eps=1e-6,
            vocab_size=152064,
            num_key_value_heads=4,
            max_position_embeddings=512,
            rope_theta=10000,
            rope_scaling={"type": "mrope", "mrope_section": [2, 1, 1]},
            tie_word_embeddings=False,
        )

        vision_config = qwen2_vl.VisionConfig(
            model_type="qwen2_vl",
            depth=2,
            embed_dim=32,
            hidden_size=32,
            image_size=224,
            num_heads=4,
            patch_size=14,
            mlp_ratio=4,
            in_channels=3,
            spatial_merge_size=1,
            temporal_patch_size=2,
        )

        config = qwen2_vl.ModelConfig(
            model_type="qwen2_vl",
            text_config=text_config,
            vision_config=vision_config,
            image_token_index=151655,
            vocab_size=32000,
        )

        model = qwen2_vl.Model(config)

        self.language_test_runner(
            model.language_model,
            config.text_config.model_type,
            config.text_config.vocab_size,
            config.text_config.num_hidden_layers,
        )

        self.vision_test_runner(
            model.vision_tower,
            config.vision_config.model_type,
            config.vision_config.hidden_size,
            config.vision_config.in_channels,
            (config.vision_config.image_size, config.vision_config.image_size),
            vision_feature_layer=-1,
            grid_thw=mx.ones((1, 3)),  # image temporals shape (num_images, 3)
        )


if __name__ == "__main__":
    unittest.main()
