# CLIP compatibility

Stable Diffusion checkpoint keys and text-encoder LoRA names depend on the CLIP module hierarchy. Transformers 4 wraps the unprojected encoder in `text_model`. Transformers 5 removes that wrapper, while its projected encoder retains it.

`library.clip_compat.CLIPTextModel` preserves the `text_model` hierarchy when the checkpoint loaders construct the unprojected encoder. It uses the upstream model directly under Transformers 4 and wraps the upstream Transformers 5 encoder without changing its forward computation. This keeps checkpoint loading, embedding access, gradient checkpointing, text-encoder LoRA names and saved state dictionaries compatible. The image pipelines use `CLIPImageProcessor`, which replaces the removed `CLIPFeatureExtractor` name.

## LoRA target selection

Checkpoint key compatibility does not imply identical trainable modules. Transformers 4.44 uses a separate CLIP SDPA attention class, which the existing class-name target matcher skips. Newer Transformers uses the matched attention class for SDPA too. Default new-network creation can therefore include attention adapters that were absent in an older SDPA run.

For new-network creation that must preserve the older MLP-only text-encoder inventory, use `--network_args text_encoder_mlp_only=true`. This option applies to text encoders only and leaves U-Net targets unchanged. Its default is `false`, preserving ordinary class-based selection. Only `true` and `false` are accepted. Reconstruction from existing LoRA weights continues to derive the module inventory from those weights. Coordinate a consumer's dependency upgrade and explicit target choice together, and compare training outputs before adopting it.

## Verification

With the selected runtime dependencies installed, run:

```bash
python -B -m unittest discover -s tests
git diff --check
```

The CLIP tests use tiny synthetic models. They cover loading legacy-prefixed weights into a meta-device model, local safetensors pretrained roundtrips, image normalization, text-encoder LoRA names, default target differences, explicit MLP-only targets on eager and SDPA backends, and checkpointed gradients. No model downloads, provider access or real datasets are required.

November 12, 2026 candidate validation covers Python 3.11 with Transformers 4.44.0, 4.57.6 and 5.17.0. The Transformers 5 test environment pairs Diffusers 0.38.0, huggingface-hub 1.5.0, tokenizers 0.23.1, safetensors 0.8.0 and regex 2025.10.22. The final eight-test suite passed on 4.44.0 and 5.17.0. SDXL CPU component checks and synthetic output/gradient comparisons also passed, including legacy MLP-only LoRA gradients and optimizer updates. GPU execution and full model training are not established by these tests.

The default requirements remain on Transformers 4.44.0. This compatibility change does not itself upgrade a consumer's dependencies. Transformers 5 adoption requires coordinated dependency resolution and testing of the intended entry point. General generation, captioning, conversion and other training workflows have not all been validated with Transformers 5. Do not infer whole-repository compatibility from the SDXL training checks.
