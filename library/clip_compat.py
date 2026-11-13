"""Preserve CLIP checkpoint keys and LoRA module names across Transformers versions."""

from packaging.version import Version
import transformers
from transformers import CLIPTextModel as TransformersCLIPTextModel
from transformers.models.clip.modeling_clip import CLIPPreTrainedModel


if Version(transformers.__version__).major < 5:
    CLIPTextModel = TransformersCLIPTextModel
else:
    class CLIPTextModel(TransformersCLIPTextModel):
        """Retain the text_model prefix used by SD checkpoints and text-encoder LoRAs."""

        def __init__(self, config):
            # Transformers 5 flattens the unprojected encoder. Keep the same nesting
            # as its projected encoder so existing checkpoints and LoRA keys agree.
            CLIPPreTrainedModel.__init__(self, config)
            self.text_model = TransformersCLIPTextModel._from_config(config)
            self.post_init()

        def forward(self, input_ids=None, attention_mask=None, position_ids=None, **kwargs):
            return self.text_model(
                input_ids=input_ids, attention_mask=attention_mask,
                position_ids=position_ids, **kwargs
            )

        def get_input_embeddings(self):
            return self.text_model.get_input_embeddings()

        def set_input_embeddings(self, value):
            self.text_model.set_input_embeddings(value)
