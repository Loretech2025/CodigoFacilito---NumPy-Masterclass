"""Vectorización: eliminar loops de Python usando operaciones NumPy."""

import math
import numpy as np

from src.benchmarks import bench


def demo_loop_vs_vectorized():
    """Suma de 1M elementos: loop de Python vs np.sum."""

    print("\n--- Loop de Python vs np.sum (1M elementos) ---\n")

    rng = np.random.default_rng(42)
    arr = rng.random(1_000_000)

    def suma_loop(a):
        total = 0.0
        for x in a:
            total += x
        return total

    def suma_numpy(a):
        return np.sum(a)

    # Verificar que dan el mismo resultado
    res_loop = suma_loop(arr)
    res_np = suma_numpy(arr)
    print(f"  Resultado loop:  {res_loop:.6f}")
    print(f"  Resultado np:    {res_np:.6f}")
    print(f"  Diferencia:      {abs(res_loop - res_np):.2e}\n")

    t_loop = bench(suma_loop, arr, n_runs=3, warmup=1)
    t_np = bench(suma_numpy, arr, n_runs=5, warmup=2)

    print(f"  Loop Python:  {t_loop['mean']*1000:.2f} ms")
    print(f"  np.sum:       {t_np['mean']*1000:.4f} ms")
    speedup = t_loop["mean"] / t_np["mean"]
    print(f"  Speedup:      {speedup:.0f}x")

    print("\n  Takeaway: np.sum opera en C sobre memoria contigua,")
    print("  evitando el overhead del loop interpretado de Python.")


def demo_elementwise():
    """Suma elemento a elemento: loop vs vectorizado en 100K elementos."""

    print("\n--- Suma Elemento a Elemento: Loop vs Vectorizado (100K) ---\n")

    rng = np.random.default_rng(42)
    a = rng.random(100_000)
    b = rng.random(100_000)

    def suma_loop(a, b):
        result = np.empty(len(a))
        for i in range(len(a)):
            result[i] = a[i] + b[i]
        return result

    def suma_vectorizada(a, b):
        return a + b

    # Verificar
    r1 = suma_loop(a, b)
    r2 = suma_vectorizada(a, b)
    print(f"  Resultados iguales: {np.allclose(r1, r2)}")
    print(f"  Primeros 5: {r2[:5]}\n")

    t_loop = bench(suma_loop, a, b, n_runs=3, warmup=1)
    t_vec = bench(suma_vectorizada, a, b, n_runs=5, warmup=2)

    print(f"  Loop Python:   {t_loop['mean']*1000:.2f} ms")
    print(f"  Vectorizado:   {t_vec['mean']*1000:.4f} ms")
    speedup = t_loop["mean"] / t_vec["mean"]
    print(f"  Speedup:       {speedup:.0f}x")

    print("\n  Takeaway: el operador + entre arrays se ejecuta en C,")
    print("  procesando bloques de memoria sin interpretar cada elemento.")


def demo_math_functions():
    """np.sqrt vs math.sqrt en array (con loop)."""

    print("\n--- np.sqrt vs math.sqrt (loop) en 100K elementos ---\n")

    rng = np.random.default_rng(42)
    arr = rng.random(100_000) + 0.01  # evitar sqrt(0) issues

    def sqrt_loop(a):
        result = np.empty(len(a))
        for i in range(len(a)):
            result[i] = math.sqrt(a[i])
        return result

    def sqrt_numpy(a):
        return np.sqrt(a)

    # Verificar
    r1 = sqrt_loop(arr)
    r2 = sqrt_numpy(arr)
    print(f"  Resultados iguales: {np.allclose(r1, r2)}")
    print(f"  Ejemplo sqrt(0.5): math={math.sqrt(0.5):.6f}, np={np.sqrt(0.5):.6f}\n")

    t_loop = bench(sqrt_loop, arr, n_runs=3, warmup=1)
    t_np = bench(sqrt_numpy, arr, n_runs=5, warmup=2)

    print(f"  math.sqrt (loop): {t_loop['mean']*1000:.2f} ms")
    print(f"  np.sqrt:          {t_np['mean']*1000:.4f} ms")
    speedup = t_loop["mean"] / t_np["mean"]
    print(f"  Speedup:          {speedup:.0f}x")

    print("\n  Takeaway: np.sqrt aplica SIMD (instrucciones vectoriales del CPU)")
    print("  sobre el array completo, mucho más rápido que llamar math.sqrt N veces.")


def demo_conditionals():
    """ReLU de 3 formas: loop con if, np.clip, np.maximum, np.where."""

    print("\n--- ReLU: 4 implementaciones comparadas (100K elementos) ---\n")
    print("  ReLU(x) = max(0, x)  — Función de activación común en redes neuronales\n")

    rng = np.random.default_rng(42)
    arr = rng.standard_normal(100_000)  # valores positivos y negativos

    def relu_loop(a):
        result = np.empty(len(a))
        for i in range(len(a)):
            result[i] = a[i] if a[i] > 0 else 0.0
        return result

    def relu_clip(a):
        return np.clip(a, 0, None)

    def relu_maximum(a):
        return np.maximum(a, 0)

    def relu_where(a):
        return np.where(a > 0, a, 0.0)

    # Verificar que todas dan el mismo resultado
    ref = relu_loop(arr)
    print(f"  Valores negativos en input: {np.sum(arr < 0)}")
    print(f"  Valores cero en output:     {np.sum(ref == 0)}")
    print(f"  clip==ref:    {np.allclose(relu_clip(arr), ref)}")
    print(f"  maximum==ref: {np.allclose(relu_maximum(arr), ref)}")
    print(f"  where==ref:   {np.allclose(relu_where(arr), ref)}\n")

    t_loop = bench(relu_loop, arr, n_runs=3, warmup=1)
    t_clip = bench(relu_clip, arr, n_runs=5, warmup=2)
    t_max = bench(relu_maximum, arr, n_runs=5, warmup=2)
    t_where = bench(relu_where, arr, n_runs=5, warmup=2)

    print(f"  Loop con if:    {t_loop['mean']*1000:.2f} ms")
    print(f"  np.clip:        {t_clip['mean']*1000:.4f} ms  (Speedup: {t_loop['mean']/t_clip['mean']:.0f}x)")
    print(f"  np.maximum:     {t_max['mean']*1000:.4f} ms  (Speedup: {t_loop['mean']/t_max['mean']:.0f}x)")
    print(f"  np.where:       {t_where['mean']*1000:.4f} ms  (Speedup: {t_loop['mean']/t_where['mean']:.0f}x)")

    print("\n  Takeaway: Las 3 opciones vectorizadas son equivalentes en velocidad.")
    print("  np.maximum es la más idiomática para ReLU.")
    print("  np.where es la más flexible para condiciones complejas.")
