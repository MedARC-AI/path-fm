import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf


@pytest.fixture
def adv_cfg():
    """Minimal OmegaConf config with adversarial training enabled."""
    return OmegaConf.create(
        {
            "compute_precision": {
                "grad_scaler": False,
                "teacher": {
                    "backbone": {
                        "sharding_strategy": "SHARD_GRAD_OP",
                        "mixed_precision": {
                            "param_dtype": "fp16",
                            "reduce_dtype": "fp16",
                            "buffer_dtype": "fp32",
                        },
                    },
                    "dino_head": {
                        "sharding_strategy": "SHARD_GRAD_OP",
                        "mixed_precision": {
                            "param_dtype": "fp16",
                            "reduce_dtype": "fp16",
                            "buffer_dtype": "fp32",
                        },
                    },
                },
                "student": {
                    "backbone": {
                        "sharding_strategy": "SHARD_GRAD_OP",
                        "mixed_precision": {
                            "param_dtype": "fp16",
                            "reduce_dtype": "fp16",
                            "buffer_dtype": "fp32",
                        },
                    },
                    "dino_head": {
                        "sharding_strategy": "SHARD_GRAD_OP",
                        "mixed_precision": {
                            "param_dtype": "fp16",
                            "reduce_dtype": "fp32",
                            "buffer_dtype": "fp32",
                        },
                    },
                    "slide_classifier": {
                        "sharding_strategy": "SHARD_GRAD_OP",
                        "mixed_precision": {
                            "param_dtype": "fp16",
                            "reduce_dtype": "fp32",
                            "buffer_dtype": "fp32",
                        },
                    },
                },
            },
            "dino": {
                "loss_weight": 1.0,
                "head_n_prototypes": 64,
                "head_bottleneck_dim": 32,
                "head_hidden_dim": 64,
                "head_nlayers": 2,
                "koleo_loss_weight": 0.0,
                "kde_loss_weight": 0.0,
            },
            "ibot": {
                "loss_weight": 0.0,
                "mask_sample_probability": 0.5,
                "mask_ratio_min_max": [0.1, 0.5],
                "separate_head": False,
                "head_n_prototypes": 64,
                "head_bottleneck_dim": 32,
                "head_hidden_dim": 64,
                "head_nlayers": 2,
            },
            "student": {
                "arch": "vit_small",
                "pretrained_weights": "",
                "patch_size": 16,
            },
            "teacher": {
                "momentum_teacher": 0.992,
                "final_momentum_teacher": 1.0,
                "warmup_teacher_temp": 0.04,
                "teacher_temp": 0.07,
                "warmup_teacher_temp_epochs": 1,
            },
            "train": {
                "centering": "centering",
            },
            "crops": {
                "local_crops_number": 4,
            },
            "optim": {
                "layerwise_decay": 1.0,
                "patch_embed_lr_mult": 0.2,
            },
            "adversarial": {
                "loss_weight": 1.0,
                "alpha": 1.0,
                "hidden_dim": 32,
                "average_crop": False,
                "local_crop_weight": 1.0,
                "min_wsi_count": 2,
            },
        }
    )
