import json
from pathlib import Path

SMALL_MODEL_FOR_TEST = "meta-llama/Llama-3.2-1B"


def load_model_config(model: str):
    from huggingface_hub import snapshot_download
    model_path = Path(snapshot_download(model, local_files_only=True))

    cfg_path: Path = model_path.joinpath("config.json")
    
    if not cfg_path.exists():
        raise FileNotFoundError(f"unable to find config.json at {model_path}")
    
    return json.loads(cfg_path.read_text())
