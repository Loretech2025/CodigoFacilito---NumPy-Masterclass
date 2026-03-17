"""Einstein summation (einsum): notación compacta para operaciones tensoriales."""

import numpy as np

from src.benchmarks import bench


def demo_einsum_basics():
    """Notación de Einstein con ejemplos simples."""

    print("\n--- Einsum: Notación de Einstein Básica ---\n")

    print("  np.einsum('subíndices', *operandos)")
    print("  Los subíndices son letras que representan ejes.")
    print("  Ejes que NO aparecen en la salida se SUMAN.\n")

    # 'i->' → suma de un vector
    v = np.array([1, 2, 3, 4, 5])
    print(f"  v = {v}")
    print(f"  einsum('i->', v) = {np.einsum('i->', v)}")
    print(f"  Equivale a: np.sum(v) = {np.sum(v)}")
    print(f"  'i->' significa: el eje i desaparece en la salida → se suma\n")

    # 'ij->' → suma total de una matriz
    M = np.array([[1, 2], [3, 4]])
    print(f"  M = {M.tolist()}")
    print(f"  einsum('ij->', M) = {np.einsum('ij->', M)}")
    print(f"  Equivale a: np.sum(M) = {np.sum(M)}")
    print(f"  'ij->' significa: ambos ejes desaparecen → suma total\n")

    # 'ij->i' → suma por filas
    print(f"  einsum('ij->i', M) = {np.einsum('ij->i', M)}")
    print(f"  Equivale a: M.sum(axis=1) = {M.sum(axis=1)}")
    print(f"  'ij->i' significa: j desaparece (se suma), i permanece → suma por filas\n")

    # 'ij->j' → suma por columnas
    print(f"  einsum('ij->j', M) = {np.einsum('ij->j', M)}")
    print(f"  Equivale a: M.sum(axis=0) = {M.sum(axis=0)}")
    print(f"  'ij->j' significa: i desaparece → suma por columnas")

    print("\n  Regla de oro: letras que aparecen a la izquierda del -> pero")
    print("  NO a la derecha se suman. Letras a la derecha son los ejes del resultado.")


def demo_einsum_operations():
    """Operaciones comunes expresadas con einsum."""

    print("\n--- Einsum: Operaciones Comunes ---\n")

    rng = np.random.default_rng(42)
    A = rng.integers(0, 5, (3, 4))
    B = rng.integers(0, 5, (4, 2))
    C = rng.integers(0, 5, (3, 4))
    u = rng.integers(0, 5, 3)
    w = rng.integers(0, 5, 4)

    ops = []

    # Matmul
    r1 = np.einsum("ik,kj->ij", A, B)
    r1_ref = A @ B
    ops.append(("ik,kj->ij", "Multiplicación de matrices", r1, r1_ref, "A @ B"))

    # Traza
    S = rng.integers(0, 5, (3, 3))
    r2 = np.einsum("ii->", S)
    r2_ref = np.trace(S)
    ops.append(("ii->", "Traza de matriz", r2, r2_ref, "np.trace(S)"))

    # Transpuesta
    r3 = np.einsum("ij->ji", A)
    r3_ref = A.T
    ops.append(("ij->ji", "Transpuesta", r3, r3_ref, "A.T"))

    # Producto externo
    r4 = np.einsum("i,j->ij", u, w)
    r4_ref = np.outer(u, w)
    ops.append(("i,j->ij", "Producto externo", r4, r4_ref, "np.outer(u, w)"))

    # Multiplicación elemento a elemento
    r5 = np.einsum("ij,ij->ij", A, C)
    r5_ref = A * C
    ops.append(("ij,ij->ij", "Multiplicación element-wise", r5, r5_ref, "A * C"))

    # Suma por filas
    r6 = np.einsum("ij->i", A)
    r6_ref = A.sum(axis=1)
    ops.append(("ij->i", "Suma por filas", r6, r6_ref, "A.sum(axis=1)"))

    for subs, desc, result, ref, equiv in ops:
        match = np.array_equal(result, ref)
        result_str = str(result.tolist()) if result.ndim > 0 else str(result)
        if len(result_str) > 60:
            result_str = result_str[:57] + "..."
        print(f"  '{subs}' — {desc}")
        print(f"    einsum:  {result_str}")
        print(f"    {equiv}: {'OK ✓' if match else 'FALLO'}\n")

    print("  Takeaway: einsum es una navaja suiza — una sola función")
    print("  reemplaza matmul, trace, transpose, outer, sum, y más.")


def demo_batch_matmul():
    """Multiplicación de matrices por lote con einsum."""

    print("\n--- Einsum: Multiplicación por Lote (Batch MatMul) ---\n")

    rng = np.random.default_rng(42)
    batch_size = 50
    A = rng.random((batch_size, 64, 32))
    B = rng.random((batch_size, 32, 16))

    print(f"  {batch_size} pares de matrices: A[{batch_size}×64×32] @ B[{batch_size}×32×16]")
    print(f"  Resultado esperado: C[{batch_size}×64×16]\n")

    # Con loop
    def batch_loop(A, B):
        results = np.empty((A.shape[0], A.shape[1], B.shape[2]))
        for i in range(A.shape[0]):
            results[i] = A[i] @ B[i]
        return results

    # Con einsum
    def batch_einsum(A, B):
        return np.einsum("bij,bjk->bik", A, B)

    # Con @ (que ya soporta batch)
    def batch_matmul(A, B):
        return A @ B

    r_loop = batch_loop(A, B)
    r_einsum = batch_einsum(A, B)
    r_matmul = batch_matmul(A, B)

    print(f"  Resultados iguales:")
    print(f"    einsum vs loop:  {np.allclose(r_einsum, r_loop)}")
    print(f"    matmul vs loop:  {np.allclose(r_matmul, r_loop)}\n")

    t_loop = bench(batch_loop, A, B, n_runs=5, warmup=2)
    t_einsum = bench(batch_einsum, A, B, n_runs=5, warmup=2)
    t_matmul = bench(batch_matmul, A, B, n_runs=5, warmup=2)

    print(f"  Benchmark:")
    print(f"    Loop:    {t_loop['mean']*1000:.2f} ms")
    print(f"    einsum:  {t_einsum['mean']*1000:.2f} ms  (Speedup: {t_loop['mean']/t_einsum['mean']:.1f}x)")
    print(f"    A @ B:   {t_matmul['mean']*1000:.2f} ms  (Speedup: {t_loop['mean']/t_matmul['mean']:.1f}x)")

    print("\n  Nota: 'bij,bjk->bik' — b es el índice de lote, j se suma (contracción).")
    print("  El operador @ también soporta batch nativamente en NumPy >= 1.16.")


def demo_optimize():
    """einsum optimize=True en cadena de 3 matrices."""

    print("\n--- Einsum: optimize=True (Orden de Contracción) ---\n")

    rng = np.random.default_rng(42)
    A = rng.random((100, 50))
    B = rng.random((50, 80))
    C = rng.random((80, 100))

    print("  Cadena de 3 matrices: A(100×50) @ B(50×80) @ C(80×100)")
    print("  einsum('ij,jk,kl->il', A, B, C)\n")

    print("  Sin optimize: evalúa de izquierda a derecha")
    print("  Con optimize=True: elige el orden de contracción óptimo")
    print("  (minimiza el número de operaciones intermedias)\n")

    def einsum_noopt(A, B, C):
        return np.einsum("ij,jk,kl->il", A, B, C, optimize=False)

    def einsum_opt(A, B, C):
        return np.einsum("ij,jk,kl->il", A, B, C, optimize=True)

    def matmul_chain(A, B, C):
        return A @ B @ C

    r1 = einsum_noopt(A, B, C)
    r2 = einsum_opt(A, B, C)
    r3 = matmul_chain(A, B, C)
    print(f"  Resultados iguales: noopt≈opt: {np.allclose(r1, r2)}, opt≈@: {np.allclose(r2, r3)}\n")

    t_noopt = bench(einsum_noopt, A, B, C, n_runs=10, warmup=3)
    t_opt = bench(einsum_opt, A, B, C, n_runs=10, warmup=3)
    t_matmul = bench(matmul_chain, A, B, C, n_runs=10, warmup=3)

    print(f"  Benchmark:")
    print(f"    einsum sin optimize: {t_noopt['mean']*1000:.4f} ms")
    print(f"    einsum optimize=True: {t_opt['mean']*1000:.4f} ms")
    print(f"    A @ B @ C:           {t_matmul['mean']*1000:.4f} ms")

    if t_noopt["mean"] > t_opt["mean"]:
        speedup = t_noopt["mean"] / t_opt["mean"]
        print(f"    optimize=True es {speedup:.1f}x más rápido")

    # Mostrar el path óptimo
    path, info = np.einsum_path("ij,jk,kl->il", A, B, C, optimize=True)
    print(f"\n  Camino de contracción óptimo: {path}")
    print(f"  {info}")

    print("\n  Takeaway: Para cadenas de >2 matrices, optimize=True puede")
    print("  ser significativamente más rápido al elegir el orden de contracción.")
