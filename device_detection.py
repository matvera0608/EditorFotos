import torch, importlib.util

tiene_CUDA = torch.cuda.is_available()
tiene_directML = importlib.util.find_spec("torch_directml")

def obtener_device():
    # CUDA
    if tiene_CUDA:
        return torch.device("cuda")

    # DirectML (protegido)
    try:
        if tiene_directML is not None:
            import torch_directml
            return torch_directml.device()
    except Exception:
        pass  # DLL incompatible → ignorar

    return torch.device("cpu")