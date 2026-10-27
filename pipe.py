import subprocess
import sys

scripts = [
    "main.py",
    "train.py",
    "prediccion.py",
    "docs/generar_html.py"
]

for script in scripts:

    print()
    print("=" * 60)
    print(f"Ejecutando {script}")
    print("=" * 60)

    resultado = subprocess.run(
        [sys.executable, script]
    )

    if resultado.returncode != 0:
        print(f"ERROR en {script}")
        sys.exit(1)

print()
print("✅ PIPELINE FINALIZADOS")