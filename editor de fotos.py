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

imagen = None

print("✅ Todo importado correctamente.\n")
print("MENÚ DE OPCIONES: ")

print("1: Quitar el fondo de un archivo")
print("2: Mejorar la calidad de una foto")

directorio = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(directorio, f"{imagen}.png")
output_path = os.path.join(directorio, f"{imagen} mejorado.png")

def quitar_fondo_de_la_imagen():
    image = Image.open(input_path)
    output_image = remove(image)
    output_image.save(output_path)

    print("¡Fondo eliminado con éxito!")
    input()
    
def mejorar_calidad_de_la_imagen():
    def calcular_tile(image):
        w, h = image.size
        lado = max(w, h)

        if lado > 4000:
            return 400
        elif lado > 2000:
            return 200
        else:
            return 0

    
    directorio = os.path.dirname(os.path.abspath(__file__))
    escalas = os.path.join(directorio, "REAL-ESRGAN_SCALE")
    modelo_path = os.path.join(escalas, "RealESRGAN_x4plus.pth")
    input_path = os.path.join(directorio, f"{imagen}.png")
    output_path = os.path.join(directorio, f"{imagen} mejorado.png")
    
    if not os.path.exists(modelo_path):
        print("El modelo no se encuentra en la ruta especificada:", modelo_path)
        return
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.8)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    print(f"Dispositivo seleccionado: {device}")
    try:
        device_torch = torch.device(device)
        image = Image.open(input_path).convert("RGB")
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
            tile=calcular_tile(image),
            tile_pad=10,
            pre_pad=0,
            half=False
        )
        image_np = np.array(image) 
        output, _ = upsampler.enhance(image_np, outscale=4)
        Image.fromarray(output).save(output_path)
        
        print("¡Calidad mejorada con éxito!")
        
    except Exception as e:
        print("Ocurrió un error al mejorar la imagen: ", e)
    finally:
        torch.cuda.empty_cache() #Esto limpia los datos basura

try:
    print(f"imágen elegida: {imagen}.png")
    opción = int(input("¿Que te gustaría hacer? ").strip())
    match opción:
        case 1:
            quitar_fondo_de_la_imagen()
        case 2:
            mejorar_calidad_de_la_imagen()
except:
    print("Ocurrió un error")