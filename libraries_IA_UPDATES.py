import asyncio
import sys
import datetime
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

# ------------------ UTILIDADES ------------------ #

async def verificar_paquetes():
     paquetes = ["gfpgan", "realesrgan", "torch", "torchvision", "torchaudio"]

     estado = {pkg: importlib.util.find_spec(pkg) is not None for pkg in paquetes}

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
     # Verifica e instala paquetes faltantes automáticamente

     await verificar_paquetes()
     #En el main yo borré actualizar_paquetes, porque está en la función asíncrona que verifica si uno está instalado o no?}

if __name__ == "__main__":
     
    asyncio.run(main())
    print("🎉 Tarea de actualización y verificación de paquetes faltantes de IA completada.")