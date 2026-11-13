import tempfile
import unittest

import torch
import transformers
from PIL import Image
from transformers import CLIPImageProcessor, CLIPTextConfig, CLIPTextModelWithProjection

from library.clip_compat import CLIPTextModel
from library.sdxl_model_util import _load_state_dict_on_device
from networks.lora import LoRANetwork, create_network


class ClipCompatibilityTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)
        self.config = CLIPTextConfig(
            vocab_size=32, hidden_size=32, intermediate_size=64,
            num_hidden_layers=1, num_attention_heads=4, max_position_embeddings=8,
            projection_dim=32, eos_token_id=2, bos_token_id=1, pad_token_id=0,
        )
        self.tokens = torch.tensor([[1, 5, 7, 2, 0, 0, 0, 0]])

    def test_checkpoint_keys_and_meta_device_loading(self):
        from accelerate import init_empty_weights

        original = CLIPTextModel(self.config).eval()
        weights = {key: value.clone() for key, value in original.state_dict().items()}
        self.assertTrue(all(key.startswith('text_model.') for key in weights))
        self.assertIn('text_model.encoder.layers.0.self_attn.q_proj.weight', weights)
        self.assertIs(original.get_input_embeddings(), original.text_model.embeddings.token_embedding)
        with init_empty_weights():
            restored = CLIPTextModel._from_config(self.config)
        _load_state_dict_on_device(restored, weights, 'cpu')
        restored.eval()
        torch.testing.assert_close(
            original(self.tokens).last_hidden_state,
            restored(self.tokens).last_hidden_state, rtol=0, atol=0,
        )

    def test_image_processor_normalization(self):
        processor = CLIPImageProcessor(do_resize=False, do_center_crop=False)
        pixels = processor(Image.new('RGB', (4, 4), (128, 64, 32)), return_tensors='pt').pixel_values
        expected = (torch.tensor([128, 64, 32]) / 255 - torch.tensor(processor.image_mean))
        expected /= torch.tensor(processor.image_std)
        torch.testing.assert_close(pixels, expected.reshape(1, 3, 1, 1).expand(1, 3, 4, 4))

    def test_local_pretrained_roundtrip(self):
        original = CLIPTextModel(self.config).eval()
        with tempfile.TemporaryDirectory() as directory:
            original.save_pretrained(directory, safe_serialization=True)
            restored = CLIPTextModel.from_pretrained(directory, local_files_only=True).eval()
        self.assertEqual(set(original.state_dict()), set(restored.state_dict()))
        torch.testing.assert_close(
            original(self.tokens).last_hidden_state,
            restored(self.tokens).last_hidden_state, rtol=0, atol=0,
        )

    def test_lora_names_and_checkpointed_backward(self):
        # Compare the same attention backend across versions. Default backend
        # target differences are checked separately and require adoption review.
        self.config._attn_implementation = 'eager'
        encoders = [CLIPTextModel(self.config), CLIPTextModelWithProjection(self.config)]
        unet = torch.nn.Identity()
        network = LoRANetwork(encoders, unet, lora_dim=4, is_sdxl=True)
        suffixes = ['self_attn_q_proj', 'self_attn_k_proj', 'self_attn_v_proj',
                    'self_attn_out_proj', 'mlp_fc1', 'mlp_fc2']
        expected = {f'lora_te{index}_text_model_encoder_layers_0_{suffix}'
                    for index in (1, 2) for suffix in suffixes}
        self.assertEqual({module.lora_name for module in network.text_encoder_loras}, expected)
        network.apply_to(encoders, unet, apply_text_encoder=True, apply_unet=False)
        for encoder in encoders:
            encoder.gradient_checkpointing_enable()
            encoder.text_model.embeddings.requires_grad_(True)
            encoder.train()
            encoder(self.tokens).last_hidden_state.square().mean().backward()
        for module in network.text_encoder_loras:
            self.assertIsNotNone(module.lora_up.weight.grad)
            self.assertTrue(torch.isfinite(module.lora_up.weight.grad).all())

    def test_default_attention_target_inventory(self):
        encoders = [CLIPTextModel(self.config), CLIPTextModelWithProjection(self.config)]
        network = LoRANetwork(encoders, torch.nn.Identity(), lora_dim=4, is_sdxl=True)
        expected_count = 4 if transformers.__version__ == '4.44.0' else 12
        self.assertEqual(len(network.text_encoder_loras), expected_count)

    def test_explicit_legacy_targets_match_across_backends(self):
        expected = {f'lora_te{index}_text_model_encoder_layers_0_mlp_fc{layer}'
                    for index in (1, 2) for layer in (1, 2)}
        for backend in ('eager', 'sdpa'):
            with self.subTest(backend=backend):
                self.config._attn_implementation = backend
                encoders = [CLIPTextModel(self.config), CLIPTextModelWithProjection(self.config)]
                network = create_network(1.0, 4, 1.0, None, encoders, torch.nn.Identity(),
                                         text_encoder_mlp_only='true')
                self.assertEqual({module.lora_name for module in network.text_encoder_loras}, expected)
        with self.assertRaisesRegex(ValueError, 'text_encoder_mlp_only'):
            create_network(1.0, 4, 1.0, None, encoders, torch.nn.Identity(), text_encoder_mlp_only='invalid')


if __name__ == '__main__':
    unittest.main()
