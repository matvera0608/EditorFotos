import os
os.system("pip uninstall numpy -y")
os.system("pip install numpy==1.26.4")

import torch

try:
    # Carga el contenido del archivo .pth
    checkpoint = torch.load('REAL-ESRGAN_SCALE/RealESRGAN_x4plus.pth')

    print(f"Tipo de objeto cargado: {type(checkpoint)}")

except Exception as e:
    print(f"Error al cargar el archivo: {e}")
    exit()

# 2. Acceder al diccionario de estado del modelo (state_dict)
if 'params_ema' in checkpoint:
    state_dict = checkpoint['params_ema']
    
    print("--- 🔬 Parámetros y Pesos del Modelo ---")
    print(f"Número total de capas/parámetros: {len(state_dict)}")
    print("-" * 40)
    
    # 3. Iterar y mostrar los primeros 5 pesos
    for name, tensor in list(state_dict.items())[:5]:
        # 'name' es el nombre de la capa (e.g., 'conv_first.weight')
        # 'tensor.shape' es la forma (shape) de los pesos
        print(f"Capa: {name}")
        print(f"  Forma (Shape): {tensor.shape}")
        print(f"  Tipo de Dato: {tensor.dtype}")
        print("-" * 40)

else:
    print("Error: La clave 'params_ema' no se encontró en el archivo.")