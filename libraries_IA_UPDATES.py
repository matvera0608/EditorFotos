import asyncio, sys, datetime, pkg_resources, subprocess, torch
from importlib.metadata import version, PackageNotFoundError
import importlib.util

# ------------------ REGISTRO DE VERSIONES ------------------ #

def registrar_version(paquete, archivo_log="paquetes_ia_log.txt"):
    try:
        version_actual = version(paquete)
    except PackageNotFoundError:
        version_actual = "No instalado"

    fecha = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    linea = f"[{fecha}] {paquete}: {version_actual}\n"

    with open(archivo_log, "a", encoding="utf-8") as f:
        f.write(linea)

    print(f"📌 Registro guardado: {paquete} → {version_actual}")

LIBRERIAS_PARA_CPU_Y_CUDA = {
    "torch": "2.0.1",
    "torchvision": "0.15.2",
    "torchaudio": "2.0.2",
    "basicsr": "1.4.2",
    "realesrgan": "0.3.0",
    "gfpgan": "1.3.8",
    "rembg": "2.0.67"
}

LIBRERIAS_PARA_DML = {
    "torch-directml": "0.2.5.dev240914",
    "basicsr": "1.4.2",
    "realesrgan": "0.3.0",
    "gfpgan": "1.3.8",
    "rembg": "2.0.67"
}

def detectar_backend():
    if torch.cuda.is_available():
        return "cuda"
    elif importlib.util.find_spec("torch_directml") is not None:
        return "dml"
    else:
        return "cpu"

def limpiar_y_sincronizar_paquetes(compat):
    instalados = {pkg.key: pkg.version for pkg in pkg_resources.working_set}

    for paquete, version_ok in compat.items():
        version_instalada = instalados.get(paquete)

        if version_instalada and version_instalada != version_ok:
            print(f"🧹 {paquete} {version_instalada} ≠ {version_ok} → desinstalando")
            subprocess.run(
                [sys.executable, "-m", "pip", "uninstall", paquete, "-y"],
                stdout=subprocess.DEVNULL
            )

        if not version_instalada or version_instalada != version_ok:
            print(f"📦 Instalando {paquete}=={version_ok}")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", f"{paquete}=={version_ok}"],
                check=True
            )

    print("✅ Entorno estabilizado.")


# ------------------ UTILIDADES ------------------ #
LIBRERIAS_IA = [
    "torch", "torchvision", "torchaudio",
    "torch-directml",
    "basicsr", "realesrgan", "gfpgan", "rembg",
    "onnx", "onnxruntime", "onnxruntime-directml"
]

async def eliminar_librerias(lista):
    print("🧹 Eliminando librerías de IA...")
    for pkg in lista:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "pip", "uninstall", "-y", pkg,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL
        )
        await proc.wait()
        print(f"⛔ {pkg} eliminado.")
    print("✨ Limpieza completa.")


async def verificar_paquetes():
    paquetes = ["gfpgan", "realesrgan","torch-directml", "rembg", "torch", "torchvision", "torchaudio"]

    estado = {pkg: importlib.util.find_spec(pkg) is not None for pkg in paquetes}

    for pkg in paquetes:
        registrar_version(pkg)

    for pkg, instalado in estado.items():
        if not instalado:
            print(f"📦 {pkg} no estaba instalado, se instalará ahora...")
        else:
            print(f"✅ {pkg} ya está instalado.")
    
    faltantes = [pkg for pkg, instalado in estado.items() if not instalado]

    if faltantes:
        print("🚀 Instalando paquetes faltantes:", ", ".join(faltantes))
        await actualizar_paquetes(faltantes, False)

async def desinstalar(paquete):
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "pip", "uninstall", "-y", paquete,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL
    )
    await proc.wait()
    print(f"⛔ {paquete} desinstalado.")

# Instalar paquete
async def instalar(paquete):
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "pip", "install", "-U", paquete,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL
    )
    await proc.wait()
    print(f"✅ {paquete} instalado correctamente.")

# --- ⭐ FUNCIÓN GENÉRICA ---
async def actualizar_libreria(paquete, registrar):
    print(f"\n🔄 Actualizando {paquete}...")

    # Intentar actualizar
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "pip", "install", "-U", paquete,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL
    )
    await proc.wait()

    # Falló → reinstalación limpia
    if proc.returncode != 0:
        print(f"⚠ Error al actualizar {paquete}. Reinstalando limpio...")
        await desinstalar(paquete)
        await instalar(paquete)
    else:
        print(f"✅ {paquete} actualizado sin errores.")

    # Registrar versión si querés
    if registrar:
        registrar_version(paquete)

async def actualizar_paquetes(lista_de_paquetes, registrar=False):
    await asyncio.gather(*(actualizar_libreria(package, registrar) for package in lista_de_paquetes))

async def main():
    
    # await eliminar_librerias(LIBRERIAS_IA)
    
    backend = detectar_backend()

    match backend:
        case "cuda":
            limpiar_y_sincronizar_paquetes(LIBRERIAS_PARA_CPU_Y_CUDA)

        case "dml":
            limpiar_y_sincronizar_paquetes(LIBRERIAS_PARA_DML)

        case _:
            limpiar_y_sincronizar_paquetes(LIBRERIAS_PARA_CPU_Y_CUDA)
 
    await verificar_paquetes()

if __name__ == "__main__":
     
    asyncio.run(main())
    print("🎉 Tarea de actualización y verificación de paquetes faltantes de IA completada.")