"""Operaciones in-place y selección de dtypes para optimizar rendimiento."""

import numpy as np

from src.benchmarks import bench


def demo_inplace():
    """Operaciones in-place (+=) vs creación de nuevo array (a = a + b)."""

    print("\n--- Operaciones In-Place vs Nuevo Array ---\n")

    rng = np.random.default_rng(42)
    a = rng.random(1_000_000)
    b = rng.random(1_000_000)

    # --- Demostrar que += no cambia id ---
    a_copy = a.copy()
    id_antes = id(a_copy)
    a_copy += b
    id_despues = id(a_copy)
    print(f"  a += b:")
    print(f"    id antes:  {id_antes}")
    print(f"    id después: {id_despues}")
    print(f"    Mismo objeto: {id_antes == id_despues}  (in-place, no crea array nuevo)\n")

    a_copy2 = a.copy()
    id_antes2 = id(a_copy2)
    a_copy2 = a_copy2 + b
    id_despues2 = id(a_copy2)
    print(f"  a = a + b:")
    print(f"    id antes:  {id_antes2}")
    print(f"    id después: {id_despues2}")
    print(f"    Mismo objeto: {id_antes2 == id_despues2}  (crea array nuevo)\n")

    # --- Benchmark ---
    def op_inplace(a, b):
        a += b
        return a

    def op_new(a, b):
        return a + b

    # Usar copias frescas para cada benchmark
    def bench_inplace():
        x = a.copy()
        x += b

    def bench_new():
        _ = a + b

    t_inplace = bench(bench_inplace, n_runs=10, warmup=3)
    t_new = bench(bench_new, n_runs=10, warmup=3)

    print(f"  Benchmark (1M elementos):")
    print(f"    In-place (+=):      {t_inplace['mean']*1000:.4f} ms")
    print(f"    Nuevo array (a+b):  {t_new['mean']*1000:.4f} ms")

    # --- out= parameter ---
    print(f"\n  Parámetro out= en ufuncs:")
    out = np.empty_like(a)
    np.add(a, b, out=out)
    print(f"    np.add(a, b, out=out) — escribe directo en 'out', sin asignar memoria")
    print(f"    Equivalente a out[:] = a + b pero más eficiente")
    print(f"    Resultado correcto: {np.allclose(out, a + b)}")

    print("\n  Takeaway: In-place ahorra una asignación de memoria.")
    print("  Con arrays grandes, la diferencia es significativa.")
    print("  Cuidado: in-place modifica el array original (puede ser un problema con vistas).")


def demo_dtype_selection():
    """Benchmark float32 vs float64 — velocidad y memoria."""

    print("\n--- Selección de dtype: float32 vs float64 ---\n")

    rng = np.random.default_rng(42)
    n = 1_000_000

    a64 = rng.random(n, dtype=np.float64)
    b64 = rng.random(n, dtype=np.float64)
    a32 = a64.astype(np.float32)
    b32 = b64.astype(np.float32)

    print(f"  Tamaño en memoria ({n:,} elementos):")
    print(f"    float64: {a64.nbytes / 1024 / 1024:.1f} MB  ({a64.itemsize} bytes/elem)")
    print(f"    float32: {a32.nbytes / 1024 / 1024:.1f} MB  ({a32.itemsize} bytes/elem)")
    print(f"    Ahorro:  {(1 - a32.nbytes / a64.nbytes) * 100:.0f}%\n")

    def suma_64(a, b):
        return a + b

    def suma_32(a, b):
        return a + b

    t64 = bench(suma_64, a64, b64, n_runs=10, warmup=3)
    t32 = bench(suma_32, a32, b32, n_runs=10, warmup=3)

    print(f"  Benchmark suma (1M elementos):")
    print(f"    float64: {t64['mean']*1000:.4f} ms")
    print(f"    float32: {t32['mean']*1000:.4f} ms")
    speedup = t64["mean"] / t32["mean"]
    print(f"    Speedup float32: {speedup:.1f}x")

    # Precisión
    ref = np.float64(1.0) + np.float64(1e-10)
    val32 = np.float32(1.0) + np.float32(1e-10)
    print(f"\n  Precisión:")
    print(f"    1.0 + 1e-10 en float64: {ref:.15f}")
    print(f"    1.0 + 1e-10 en float32: {val32:.15f}")
    print(f"    float32 pierde los últimos dígitos significativos")

    print("\n  Takeaway: float32 usa la mitad de memoria y puede ser más rápido")
    print("  (más datos caben en caché). Úsalo cuando la precisión de ~7 dígitos basta")
    print("  (gráficos, ML, sensores). Para ciencia/finanzas, usa float64.")


def demo_dtype_table():
    """Tabla de dtypes comunes con tamaño y caso de uso."""

    print("\n--- Tabla de dtypes Comunes ---\n")

    dtypes_info = [
        ("bool_",    "np.bool_",    "1 bit*",  "Máscaras, flags"),
        ("int8",     "np.int8",     "1 byte",  "Imágenes (0-255 con uint8), categorías pequeñas"),
        ("int16",    "np.int16",    "2 bytes", "Audio (PCM 16-bit), sensores"),
        ("int32",    "np.int32",    "4 bytes", "Contadores, índices (default entero en Windows)"),
        ("int64",    "np.int64",    "8 bytes", "Contadores grandes (default entero en Linux/Mac)"),
        ("float16",  "np.float16",  "2 bytes", "ML inferencia, GPU (precisión ~3 dígitos)"),
        ("float32",  "np.float32",  "4 bytes", "ML entrenamiento, gráficos (precisión ~7 dígitos)"),
        ("float64",  "np.float64",  "8 bytes", "Ciencia, finanzas (precisión ~15 dígitos, default)"),
        ("complex64","np.complex64","8 bytes", "Señales, FFT (2 × float32)"),
        ("complex128","np.complex128","16 bytes","Física, ingeniería (2 × float64)"),
    ]

    print(f"  {'dtype':<14} {'Constructor':<16} {'Tamaño':<10} {'Uso típico'}")
    print(f"  {'─'*14} {'─'*16} {'─'*10} {'─'*40}")
    for name, constructor, size, use in dtypes_info:
        print(f"  {name:<14} {constructor:<16} {size:<10} {use}")

    print(f"\n  * bool_ ocupa 1 byte en memoria (no 1 bit), por alineación")

    # Demostrar tamaños reales
    print(f"\n  Tamaños reales de un array de 1000 elementos:")
    for dtype_str in ["bool_", "int8", "int16", "int32", "int64",
                       "float16", "float32", "float64"]:
        dtype = getattr(np, dtype_str)
        arr = np.zeros(1000, dtype=dtype)
        print(f"    {dtype_str:<10}: {arr.nbytes:>6} bytes ({arr.itemsize} bytes/elem)")

    print("\n  Takeaway: Elegir el dtype correcto puede reducir memoria 2-8x")
    print("  y mejorar velocidad por mejor uso de caché.")
