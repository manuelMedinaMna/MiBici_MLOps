import subprocess
import logging
import sys

def run_step(module_name):
    """Ejecuta un módulo de Python como subprocess y maneja errores."""
    print(f"\n[PIPELINE] Ejecutando: python -m {module_name}")
    result = subprocess.run([sys.executable, "-m", module_name])
    
    if result.returncode != 0:
        print(f"\n[ERROR] El paso '{module_name}' falló con código de salida {result.returncode}.")
        sys.exit(result.returncode)

def main():
    # Lista de módulos en orden secuencial para el clustering
    steps = [
        "mibici_mlops.dataset",
        "mibici_mlops.features_p3",
        "mibici_mlops.modeling.train_p3",
        "mibici_mlops.plots"
    ]
    
    logging.info("=== INICIANDO PROCESO DE CLUSTERIZACIÓN ===")
    
    for step in steps:
        run_step(step)
        
    logging.info("=== ¡PROCESO DE CLUSTERIZACIÓN COMPLETADO EXITOSAMENTE! ===")

if __name__ == "__main__":
    main()