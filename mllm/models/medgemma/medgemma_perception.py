"""
MedGemma Perception Model

Based on Google's MedGemma with FLARE25 medical imaging fine-tuning.
Uses Gemma 3 architecture with integrated SigLIP vision encoder.
"""

import copy
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import (
    GemmaConfig,
    GemmaForCausalLM,
    AutoImageProcessor
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from mllm.models.utils.modeling_outputs import CausalLMOutputWithPastCustom
from peft import PeftModel, LoraConfig, get_peft_model
import torch.distributed as dist

# Special tokens for MedGemma
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


class MedGemmaConfig(GemmaConfig):
    """
    Config for MedGemma-based Perception model.
    Extends GemmaConfig with vision-language specific parameters.
    """
    model_type = "medgemma"

    def __init__(
        self,
        image_size=448,
        image_token_len=256,
        mm_use_im_start_end=True,
        mm_vision_select_layer=-2,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.image_token_len = image_token_len
        self.mm_use_im_start_end = mm_use_im_start_end
        self.mm_vision_select_layer = mm_vision_select_layer


class MedGemmaPerception(GemmaForCausalLM):
    """
    MedGemma Perception Model for medical imaging tasks.

    This is a wrapper around GemmaForCausalLM with FLARE25 medical imaging
    fine-tuning. The vision encoder (SigLIP) is integrated within MedGemma,
    unlike PerceptionGPT which uses separate CLIP encoder.

    Features:
    - Gemma 3 architecture (4B parameters)
    - SigLIP vision encoder (integrated)
    - LoRA fine-tuned on 19 medical imaging datasets
    - Supports 7 imaging modalities: CT, MRI, X-ray, Ultrasound, Fundus, Pathology, Endoscopy
    """
    config_class = MedGemmaConfig

    def __init__(self, config: MedGemmaConfig):
        # Fix for transformers >= 4.57: Convert dict configs to Config objects
        # This prevents AttributeError: 'dict' object has no attribute 'to_dict'
        # when GenerationConfig.from_model_config() is called

        if hasattr(config, 'decoder_config') and isinstance(config.decoder_config, dict):
            # Convert dict to Config object
            decoder_config_dict = config.decoder_config
            config.decoder_config = type(config)(**decoder_config_dict)

        # For text_config, preserve it as Gemma3 needs it for vocab_size
        if hasattr(config, 'text_config') and isinstance(config.text_config, dict):
            from transformers import GemmaConfig
            text_config_dict = config.text_config
            config.text_config = GemmaConfig(**text_config_dict)

            # Copy essential attributes from text_config to main config
            # GemmaModel expects these at top level, not in text_config
            essential_attrs = ['vocab_size', 'hidden_size', 'num_hidden_layers',
                             'num_attention_heads', 'intermediate_size']
            for attr in essential_attrs:
                if hasattr(config.text_config, attr) and not hasattr(config, attr):
                    setattr(config, attr, getattr(config.text_config, attr))

        super(MedGemmaPerception, self).__init__(config)
        self.config = config

        # Vision-related attributes
        self.image_size = getattr(config, 'image_size', 448)
        self.image_token_len = getattr(config, 'image_token_len', 256)

        # Loss weights (compatible with PerceptionTrainer)
        self.lm_loss_weight = 1.0
        self.recon_loss_weight = 0.0  # Not used in MedGemma
        self.l2_loss_weight = 0.0      # Not used in MedGemma
        self.box_loss_weight = 0.0     # Not used in MedGemma

    def set_loss_weights(self, model_args):
        """Set loss weights from model_args (for compatibility with PerceptionTrainer)"""
        self.lm_loss_weight = getattr(model_args, "lm_loss_weight", 1.0)
        self.recon_loss_weight = getattr(model_args, "recon_loss_weight", 0.0)
        self.l2_loss_weight = getattr(model_args, "l2_loss_weight", 0.0)
        self.box_loss_weight = getattr(model_args, "box_loss_weight", 0.0)

        print(f"lm_loss_weight {self.lm_loss_weight}")
        print(f"recon_loss_weight {self.recon_loss_weight}")
        print(f"l2_loss_weight {self.l2_loss_weight}")
        print(f"box_loss_weight {self.box_loss_weight}")

    def get_model(self):
        """Return self (for compatibility with PerceptionGPT interface)"""
        return self

    def initialize_vision_tokenizer(
        self,
        mm_use_im_start_end,
        tokenizer,
        device,
        tune_mm_mlp_adapter=False,
        pretrain_mm_mlp_adapter=None,
        vision_config=None
    ):
        """
        Initialize vision-related tokens in tokenizer.

        For MedGemma, vision processing is integrated, but we still need
        to add special tokens for compatibility with conversation templates.
        """
        # Add image patch token
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        self.resize_token_embeddings(len(tokenizer))

        # Add start/end tokens if needed
        if mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN],
                special_tokens=True
            )
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                # Initialize new tokens with mean of existing embeddings
                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True
                )
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True
                )

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

        # Store config
        self.config.im_start_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN])[0] if mm_use_im_start_end else None
        self.config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_END_TOKEN])[0] if mm_use_im_start_end else None
        self.config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Forward pass through MedGemma model.

        Note: MedGemma handles vision processing internally, so we just pass
        through to the parent GemmaForCausalLM with minimal modifications.

        For compatibility with PerceptionTrainer, we accept 'images' parameter
        but MedGemma processes them differently than PerceptionGPT.
        """
        # Call parent forward (GemmaForCausalLM handles vision internally)
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs
        )

        # For compatibility with PerceptionTrainer, wrap in custom output
        # Only LM loss is used for MedGemma (no reconstruction/box losses)
        if labels is not None and hasattr(outputs, 'loss'):
            lm_loss = outputs.loss
            total_loss = self.lm_loss_weight * lm_loss

            # Return custom output with loss breakdown
            return CausalLMOutputWithPastCustom(
                loss=total_loss,
                lm_loss=lm_loss,
                recon_loss=torch.tensor(0.0, device=lm_loss.device),
                box_loss=torch.tensor(0.0, device=lm_loss.device),
                l2_loss=torch.tensor(0.0, device=lm_loss.device),
                logits=outputs.logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        return outputs

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Load pretrained MedGemma model.

        This can load either:
        1. Base google/medgemma-4b-it model
        2. LoRA adapters from leoyinn/flare25-medgemma (handled by builder)
        """
        # Fix for transformers >= 4.57: Convert dict configs to Config objects
        # Load config first to clean it
        from transformers import AutoConfig, PretrainedConfig

        config = kwargs.get('config', None)
        if config is None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path,
                trust_remote_code=kwargs.get('trust_remote_code', True)
            )

        # Convert decoder_config from dict to Config object if needed
        if hasattr(config, 'decoder_config') and isinstance(config.decoder_config, dict):
            # Create a PretrainedConfig from the dict
            decoder_config_dict = config.decoder_config
            # Use the same config class as the main config
            config.decoder_config = type(config)(**decoder_config_dict)

        # Convert text_config from dict to Config object if needed
        # BUT preserve it (don't set to None) as Gemma3 needs it for vocab_size
        if hasattr(config, 'text_config') and isinstance(config.text_config, dict):
            text_config_dict = config.text_config
            # Use GemmaConfig for text_config
            from transformers import GemmaConfig
            config.text_config = GemmaConfig(**text_config_dict)

            # Copy essential attributes from text_config to main config
            # GemmaModel expects these at top level, not in text_config
            essential_attrs = ['vocab_size', 'hidden_size', 'num_hidden_layers',
                             'num_attention_heads', 'intermediate_size']
            for attr in essential_attrs:
                if hasattr(config.text_config, attr) and not hasattr(config, attr):
                    setattr(config, attr, getattr(config.text_config, attr))

        # Pass cleaned config
        kwargs['config'] = config

        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs
    ):
        """Prepare inputs for generation (inference)"""
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # If using inputs_embeds, only use them in first forward pass
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update({
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
            "images": kwargs.get("images", None),
        })

        return model_inputs


# Register config
MedGemmaConfig.register_for_auto_class()
MedGemmaPerception.register_for_auto_class("AutoModelForCausalLM")
