"""Utilidades de benchmarking para medir rendimiento de operaciones NumPy."""

import time
import numpy as np


def bench(fn, *args, n_runs=5, warmup=1):
    """Ejecuta fn(*args) con calentamiento y múltiples corridas cronometradas.

    Retorna dict con mean, std, min y results (lista de tiempos).
    """
    # Calentamiento: ejecutar sin medir para estabilizar caché/JIT
    for _ in range(warmup):
        fn(*args)

    tiempos = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn(*args)
        t1 = time.perf_counter()
        tiempos.append(t1 - t0)

    return {
        "mean": np.mean(tiempos),
        "std": np.std(tiempos),
        "min": np.min(tiempos),
        "results": tiempos,
    }


def demo_timing_tools():
    """Muestra cómo funciona perf_counter y la utilidad bench()."""

    print("\n--- Herramientas de Medición de Tiempo ---\n")

    # --- perf_counter básico ---
    print("1) time.perf_counter() — reloj de alta resolución:")
    print("   Mide el tiempo real transcurrido (wall-clock) con la mayor")
    print("   precisión disponible en el sistema.\n")

    t0 = time.perf_counter()
    total = sum(range(100_000))
    t1 = time.perf_counter()
    print(f"   Ejemplo: sumar range(100_000) = {total}")
    print(f"   Tiempo: {t1 - t0:.6f} s\n")

    # --- Demostración de bench() ---
    print("2) bench(fn, *args, n_runs=5, warmup=1):")
    print("   - Hace 'warmup' ejecuciones sin medir (calentar caché)")
    print("   - Luego mide 'n_runs' ejecuciones con perf_counter")
    print("   - Retorna: mean, std, min, results\n")

    rng = np.random.default_rng(42)
    arr = rng.random(500_000)

    def suma_numpy(a):
        return np.sum(a)

    resultado = bench(suma_numpy, arr, n_runs=7, warmup=2)
    print(f"   bench(np.sum, array_500K, n_runs=7, warmup=2):")
    print(f"   Media:  {resultado['mean']*1000:.4f} ms")
    print(f"   Std:    {resultado['std']*1000:.4f} ms")
    print(f"   Mínimo: {resultado['min']*1000:.4f} ms")
    print(f"   Corridas: {[f'{t*1000:.4f}' for t in resultado['results']]}")

    print("\n   Takeaway: bench() nos da estadísticas confiables al promediar")
    print("   múltiples corridas, eliminando ruido de la primera ejecución.")
