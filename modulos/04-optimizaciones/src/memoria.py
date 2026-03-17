"""Memoria: orden C vs F, localidad de caché, vistas vs copias."""

import numpy as np

from src.benchmarks import bench


def demo_c_vs_f_order():
    """Crear arrays en orden C y F, comparar strides y layout de memoria."""

    print("\n--- Orden de Memoria: C (row-major) vs F (column-major) ---\n")

    arr_c = np.array([[1, 2, 3],
                       [4, 5, 6]], order="C")
    arr_f = np.array([[1, 2, 3],
                       [4, 5, 6]], order="F")

    print(f"  Matriz (2x3):")
    print(f"  [[1, 2, 3],")
    print(f"   [4, 5, 6]]\n")

    print(f"  Orden C (row-major) — filas contiguas en memoria:")
    print(f"    strides: {arr_c.strides}  (bytes para avanzar por fila, columna)")
    print(f"    flags.c_contiguous: {arr_c.flags['C_CONTIGUOUS']}")
    print(f"    En memoria: {arr_c.ravel(order='C').tolist()}")
    print(f"    → [fila0..., fila1...] = [1, 2, 3, 4, 5, 6]\n")

    print(f"  Orden F (column-major) — columnas contiguas en memoria:")
    print(f"    strides: {arr_f.strides}  (bytes para avanzar por fila, columna)")
    print(f"    flags.f_contiguous: {arr_f.flags['F_CONTIGUOUS']}")
    print(f"    En memoria: {arr_f.ravel(order='F').tolist()}")
    print(f"    → [col0..., col1..., col2...] = [1, 4, 2, 5, 3, 6]\n")

    print("  Strides explicados:")
    print(f"    C: stride fila={arr_c.strides[0]} bytes (3 elem × {arr_c.itemsize} bytes)")
    print(f"       stride col={arr_c.strides[1]} bytes (1 elem × {arr_c.itemsize} bytes)")
    print(f"    F: stride fila={arr_f.strides[0]} bytes (1 elem × {arr_f.itemsize} bytes)")
    print(f"       stride col={arr_f.strides[1]} bytes (2 elem × {arr_f.itemsize} bytes)")

    print("\n  Takeaway: NumPy usa orden C por defecto. El stride más pequeño")
    print("  indica la dimensión contigua en memoria — recorrerla es más rápido.")


def demo_cache_locality():
    """Suma por filas vs por columnas en array grande — efecto de caché."""

    print("\n--- Localidad de Caché: Suma por Filas vs Columnas ---\n")

    rng = np.random.default_rng(42)
    arr = rng.random((2000, 2000), dtype=np.float64)  # ~30 MB, orden C

    print(f"  Array: {arr.shape}, dtype={arr.dtype}, orden C")
    print(f"  Tamaño: {arr.nbytes / 1024 / 1024:.1f} MB\n")

    def suma_por_filas(a):
        return a.sum(axis=1)

    def suma_por_columnas(a):
        return a.sum(axis=0)

    t_filas = bench(suma_por_filas, arr, n_runs=5, warmup=2)
    t_cols = bench(suma_por_columnas, arr, n_runs=5, warmup=2)

    print(f"  Suma por filas (axis=1):    {t_filas['mean']*1000:.4f} ms")
    print(f"  Suma por columnas (axis=0): {t_cols['mean']*1000:.4f} ms")

    if t_cols["mean"] > t_filas["mean"]:
        ratio = t_cols["mean"] / t_filas["mean"]
        print(f"  Filas es {ratio:.1f}x más rápido")
    else:
        ratio = t_filas["mean"] / t_cols["mean"]
        print(f"  Columnas es {ratio:.1f}x más rápido (NumPy optimiza internamente)")

    print("\n  En arrays orden C, recorrer por filas accede a memoria contigua,")
    print("  aprovechando las líneas de caché del CPU (típicamente 64 bytes).")
    print("  Recorrer por columnas salta entre posiciones distantes → cache misses.")

    # Demostrar con F order
    arr_f = np.asfortranarray(arr)
    t_filas_f = bench(suma_por_filas, arr_f, n_runs=5, warmup=2)
    t_cols_f = bench(suma_por_columnas, arr_f, n_runs=5, warmup=2)

    print(f"\n  Con orden F (column-major):")
    print(f"  Suma por filas (axis=1):    {t_filas_f['mean']*1000:.4f} ms")
    print(f"  Suma por columnas (axis=0): {t_cols_f['mean']*1000:.4f} ms")
    print("  → El patrón se invierte: ahora columnas son contiguas.")


def demo_views_vs_copies():
    """Detectar vistas con .base, costo de .copy() innecesario."""

    print("\n--- Vistas vs Copias ---\n")

    rng = np.random.default_rng(42)
    arr = rng.random(1_000_000)

    # --- Vistas ---
    vista = arr[::2]         # slicing → vista
    copia = arr[::2].copy()  # copia explícita

    print("  Slicing crea una VISTA (comparte memoria):")
    print(f"    arr.base is None:     {arr.base is None}   (arr es dueño de sus datos)")
    print(f"    vista.base is arr:    {vista.base is arr}  (vista apunta a arr)")
    print(f"    copia.base is None:   {copia.base is None}  (copia tiene su propia memoria)\n")

    # Modificar vista afecta el original
    original_val = arr[0]
    vista[0] = -999.0
    print(f"  vista[0] = -999.0")
    print(f"  arr[0] ahora es: {arr[0]}  (¡se modificó el original!)")
    arr[0] = original_val  # restaurar

    print(f"\n  Copia es independiente:")
    copia[0] = -888.0
    print(f"  copia[0] = -888.0")
    print(f"  arr[0] sigue siendo: {arr[0]}  (no se modificó)\n")

    # --- Benchmark: crear vista vs copia ---
    big = rng.random(5_000_000)

    def crear_vista(a):
        return a[::2]

    def crear_copia(a):
        return a[::2].copy()

    t_vista = bench(crear_vista, big, n_runs=10, warmup=3)
    t_copia = bench(crear_copia, big, n_runs=10, warmup=3)

    print(f"  Benchmark crear vista vs copia (5M elementos, slice [::2]):")
    print(f"    Vista:  {t_vista['mean']*1e6:.2f} µs")
    print(f"    Copia:  {t_copia['mean']*1000:.4f} ms")
    speedup = t_copia["mean"] / t_vista["mean"]
    print(f"    Copia es {speedup:.0f}x más lento que vista")

    print("\n  Takeaway: Las vistas son casi instantáneas (solo metadata).")
    print("  Las copias requieren asignar memoria y copiar datos.")
    print("  Usa .copy() solo cuando necesites independencia del original.")
