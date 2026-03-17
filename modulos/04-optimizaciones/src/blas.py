"""BLAS: la biblioteca de álgebra lineal detrás de NumPy."""

import io
import contextlib

import numpy as np

from src.benchmarks import bench


def demo_blas_info():
    """Mostrar configuración de BLAS/LAPACK en NumPy."""

    print("\n--- ¿Qué es BLAS? ---\n")

    print("  BLAS = Basic Linear Algebra Subprograms")
    print("  Es una especificación de rutinas para operaciones de álgebra lineal:")
    print("    - Nivel 1: operaciones vector-vector (dot, axpy, norm)")
    print("    - Nivel 2: operaciones matriz-vector (gemv, ger)")
    print("    - Nivel 3: operaciones matriz-matriz (gemm = matmul)")
    print()
    print("  Implementaciones comunes:")
    print("    - OpenBLAS (open source, la más común en Linux)")
    print("    - MKL (Intel Math Kernel Library, optimizada para Intel)")
    print("    - BLIS (AMD, moderna)")
    print("    - Accelerate (macOS, Apple Silicon)")
    print()
    print("  NumPy delega operaciones como @ (matmul) a BLAS,")
    print("  que usa instrucciones SIMD, multithreading, y optimización de caché.\n")

    print("  Configuración actual de NumPy:")
    print("  " + "─" * 50)

    # Capturar output de show_config
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        np.show_config()
    config_text = buf.getvalue()

    # Imprimir con indentación
    for line in config_text.strip().split("\n")[:30]:  # limitar a 30 líneas
        print(f"  {line}")
    if config_text.strip().count("\n") > 30:
        print("  ... (truncado)")
    print("  " + "─" * 50)


def demo_naive_vs_blas():
    """Triple loop matmul vs operador @ (BLAS)."""

    print("\n--- Matmul: Triple Loop vs @ (BLAS) ---\n")

    n = 100
    rng = np.random.default_rng(42)
    A = rng.random((n, n))
    B = rng.random((n, n))

    print(f"  Matrices {n}×{n}\n")

    print("  Triple loop (el algoritmo 'ingenuo'):")
    print("    for i in range(N):")
    print("        for j in range(N):")
    print("            for k in range(N):")
    print("                C[i,j] += A[i,k] * B[k,j]\n")

    def matmul_naive(A, B):
        n = A.shape[0]
        C = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                s = 0.0
                for k in range(n):
                    s += A[i, k] * B[k, j]
                C[i, j] = s
        return C

    def matmul_blas(A, B):
        return A @ B

    # Verificar
    C_naive = matmul_naive(A, B)
    C_blas = matmul_blas(A, B)
    print(f"  Resultados iguales: {np.allclose(C_naive, C_blas)}\n")

    t_naive = bench(matmul_naive, A, B, n_runs=2, warmup=1)
    t_blas = bench(matmul_blas, A, B, n_runs=10, warmup=3)

    print(f"  Triple loop Python: {t_naive['mean']*1000:.1f} ms")
    print(f"  Operador @ (BLAS):  {t_blas['mean']*1000:.4f} ms")
    speedup = t_naive["mean"] / t_blas["mean"]
    print(f"  Speedup: {speedup:.0f}x")

    print("\n  ¿Por qué tanta diferencia?")
    print("  1. BLAS está escrito en C/Fortran optimizado")
    print("  2. Usa instrucciones SIMD (procesa múltiples floats a la vez)")
    print("  3. Optimiza acceso a caché (tiling/blocking)")
    print("  4. Puede usar múltiples hilos de CPU")
    print("  5. El triple loop Python tiene overhead por cada operación interpretada")


def demo_scaling():
    """Operador @ a diferentes tamaños — escalamiento cúbico."""

    print("\n--- Escalamiento del Operador @ (BLAS matmul) ---\n")

    sizes = [10, 50, 100, 500, 1000]
    rng = np.random.default_rng(42)

    print("  Complejidad teórica de matmul: O(N³)")
    print("  Si duplicamos N, el tiempo debería multiplicarse por ~8\n")

    print(f"  {'N':<8} {'Tiempo (ms)':<15} {'Ops (M)':<12} {'GFLOPS':<10} {'Ratio vs anterior'}")
    print(f"  {'─'*8} {'─'*15} {'─'*12} {'─'*10} {'─'*20}")

    prev_time = None
    prev_n = None

    for n in sizes:
        A = rng.random((n, n))
        B = rng.random((n, n))

        def matmul(A, B):
            return A @ B

        t = bench(matmul, A, B, n_runs=5, warmup=2)
        tiempo_ms = t["mean"] * 1000

        # Operaciones: 2*N³ (multiply + add)
        ops = 2 * n**3
        gflops = ops / t["mean"] / 1e9

        if prev_time is not None and prev_time > 0:
            ratio = tiempo_ms / prev_time
            n_ratio = n / prev_n
            expected = n_ratio**3
            ratio_str = f"{ratio:.1f}x (esperado ~{expected:.0f}x)"
        else:
            ratio_str = "—"

        print(f"  {n:<8} {tiempo_ms:<15.4f} {ops/1e6:<12.1f} {gflops:<10.2f} {ratio_str}")
        prev_time = tiempo_ms
        prev_n = n

    print(f"\n  GFLOPS = Giga operaciones de punto flotante por segundo")
    print(f"  Valores típicos: OpenBLAS ~10-50 GFLOPS, MKL ~20-100 GFLOPS")

    print("\n  Takeaway: BLAS mantiene alto throughput incluso con matrices")
    print("  grandes gracias a optimización de caché (tiling). El escalamiento")
    print("  sigue siendo O(N³) pero con una constante mucho menor.")
