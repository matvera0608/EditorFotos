import torch, importlib.util

def obtener_device():
    # CUDA
    if torch.cuda.is_available():
        return torch.device("cuda")

    # DirectML (protegido)
    try:
        if importlib.util.find_spec("torch_directml") is not None:
            import torch_directml
            return torch_directml.device()
    except Exception:
        pass  # DLL incompatible → ignorar

    return torch.device("cpu")

