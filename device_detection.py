import torch
import torch_directml

import torch, torch_directml

print("Torch:", torch.__version__)
print("CUDA disponible:", torch.cuda.is_available())

try:
    dml = torch_directml.device()
    print("DirectML OK:", dml)
except Exception as e:
    print("DirectML no disponible:", e)


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
            print("🔧 Usando CPU")
            return torch.device("cpu")
    else:
        try:
            print("🔧 Usando DirectML como aceleración alternativa.") #Esto me imprime al usar mi vieja Notebook
            return torch_directml.device()
        except:
            print("🔧 No hay aceleración disponible, usando CPU.")
            return torch.device("cpu")