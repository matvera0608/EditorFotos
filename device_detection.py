import torch
import torch_directml #Esto me subraya amarillo diciendo " No se ha podido resolver la importación "torch_directml". "

def obtener_device():
    if torch.cuda.is_available():
        try:
            compute = torch.cuda.get_device_capability()
            if compute >= (12, 0):
                print("⚠ GPU detectada pero PyTorch aún no soporta la arquitectura sm_120")
                print("➡ Se utilizará CPU temporalmente para evitar fallos.")
                return torch.device("cpu")
            else:
                print("🔧 Usando CUDA")
                return torch.device("cuda")
        except:
            return torch.device("cpu")
    else:
        try:
            print("🔧 Usando DirectML como aceleración alternativa.")
            return torch_directml.device()
        except:
            print("🔧 No hay aceleración disponible, usando CPU.")
            return torch.device("cpu")

