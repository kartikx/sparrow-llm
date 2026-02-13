from __future__ import annotations

import json
from pathlib import Path
from safetensors.torch import load_file

import torch
import logging

from typing import Any

logger = logging.getLogger(__name__)

from models.base import BaseLLMModel
from models.llama import LlamaForCausalLM

dtype_map = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32   
}

def _read_config(model_path: Path):
    cfg_path: Path = model_path.joinpath("config.json")
    
    if not cfg_path.exists():
        raise FileNotFoundError(f"unable to find config.json at {model_path}")
    
    return json.loads(cfg_path.read_text())

def _locate_model_dir(model_name: str) -> Path:
    """
    Returns the hf model dir for the given model.
    Potentially downloading it from hf hub
    """
    from huggingface_hub import snapshot_download
    
    return Path(snapshot_download(model_name, local_files_only=False))

def _load_safetensors(model_path: Path) -> dict[str, torch.Tensor]:
    """
    it should load weights from the safetensors into a dict, that we can then use
    to actually load the model on the GPU.
    """

    weights: dict[str, torch.Tensor] = {}

    safetensors_index_file: Path = model_path.joinpath("model.safetensors.index.json")

    # if folder contains model.safetensors.index.json, you need to process each shard.
    if safetensors_index_file.exists():
        # read the file for all the shards.
        safetensors_index_json = json.loads(safetensors_index_file.read_text())
        
        weight_map = safetensors_index_json.get("weight_map")
        if not weight_map:
            raise RuntimeError("model.model.safetensors.index.json must contain a non-empty weight_map")

        shards = sorted(set(weight_map.values()))
        if not shards:
            raise RuntimeError("unable to find shards in safetensors index")

        for shard in shards:
            shard_path = model_path.joinpath(shard)
            if not shard_path.exists():
                raise FileNotFoundError(f"unable to find shard {shard} at {model_path}")
            weights.update(load_file(shard_path, device="cpu"))
    else:
        safetensors_file:Path = model_path.joinpath("model.safetensors")
        
        if not safetensors_file.exists():
            raise FileNotFoundError(f"unable to find model.safetensors at {model_path}")
        
        weights.update(load_file(safetensors_file, device="cpu"))
        
    return weights

def load_model(model_name: str, device: torch.device) -> BaseLLMModel | None:
    model_path = _locate_model_dir(model_name)
    
    llama_config: dict[str, Any] = _read_config(model_path)
    
    logger.debug("loading safetensors")
    weights_dict = _load_safetensors(model_path)
    logger.debug("loaded safetensors")
    
    model: BaseLLMModel | None = None
    
    model_type = llama_config.get("model_type", None)
    
    assert model_type is not None, "unable to get model_type from model config"
    
    match model_type:
        case "llama":
            model = LlamaForCausalLM(llama_config)
        case _:
            raise ValueError(f"Unssupported model_type: {model_type}")

    logger.debug("loading weights")
    model.load_weights(weights_dict)
    logger.debug("loaded weights")

    model_dtype = llama_config.get("torch_dtype", None)
    if model_dtype:
        model_dtype = dtype_map[model_dtype]
        model = model.to(device=device, dtype= model_dtype)

    return model
    
