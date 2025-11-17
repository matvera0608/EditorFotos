import numpy as np, os

# def limpiar_entorno():
#     print("🔧 Limpiando entorno para Real-ESRGAN...")
#     os.system("pip uninstall torchaudio torchvision -y")
#     os.system("pip install torchvision==0.14.1")
#     print("✅ Entorno listo para mejorar imágenes sin conflictos.")

if int(np.__version__.split('.')[0]) >= 2:
    print("ERROR: NumPy 2.x no es compatible con Real-ESRGAN.")
    print("Por favor, ejecutá: pip install numpy==1.26.4")
    input()
    exit()

from rembg import remove
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from PIL import Image
import torch

print("✅ Todo importado correctamente.\n")


print("MENÚ DE OPCIONES: ")

print("1: Quitar el fondo de un archivo")
print("2: Mejorar la calidad de una foto")

# try:
#     datos = torch.load(modelo_path, map_location="cpu")
#     print("Claves dentro del archivo:", list(datos.keys()))
# except Exception as e:
#     print("Error al abrir el modelo:", e)

def quitar_fondo_de_la_imagen():
    
    directorio = os.path.dirname(os.path.abspath(__file__))

    input_path = os.path.join(directorio, "daniPerro.png")
    
    output_path = os.path.join(directorio, "daniPerro mejorado.png")

    image = Image.open(input_path)

    output_image = remove(image)

    output_image.save(output_path)

    print("¡Fondo eliminado con éxito!")
    input()
    
def mejorar_calidad_de_la_imagen(image):
    
    directorio = os.path.dirname(os.path.abspath(__file__))
    
    escalas = os.path.join(directorio, "REAL-ESRGAN_SCALE")
    modelo_path = os.path.join(escalas, "RealESRGAN_x4plus.pth")
    input_path = os.path.join(directorio, f"{image}.png")
    output_path = os.path.join(directorio, f"{image} mejorado.png")
    
    if not os.path.exists(modelo_path):
        print("El modelo no se encuentra en la ruta especificada:", modelo_path)
        return
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo seleccionado: {device}")
    try:
        device_torch = torch.device(device)
        image = Image.open(input_path).convert("RGB")

        width, height = image.size
        if max(width, height) >= 2000:
            tile = 200
        else:
            tile = 0

        model = RRDBNet(
            num_in_ch=3, num_out_ch=3,
            num_feat=64, num_block=23,
            num_grow_ch=32, scale=4
        )
        
        upsampler = RealESRGANer(
            scale=4,
            model_path=modelo_path,
            model=model,
            dni_weight=None,
            device=device_torch,
            tile=tile,
            tile_pad=10,
            pre_pad=0,
            half=False
        )
        image_np = np.array(image) 
        output, _ = upsampler.enhance(image_np, outscale=4)
        Image.fromarray(output).save(output_path)
        
        print("¡Calidad mejorada con éxito!")
        input()
    except Exception as e:
        print("Ocurrió un error al mejorar la imagen: ", e)

try:
    opción = int(input("¿Que te gustaría hacer? ").strip())

    match opción:
        case 1:
            quitar_fondo_de_la_imagen()
        case 2:
            mejorar_calidad_de_la_imagen()
except:
    print("Ocurrió un error")