"""Creación de arrays en NumPy."""

import numpy as np


def demo_from_lists():
    """Crea arrays a partir de listas de Python, 1D y anidadas, especificando dtype."""
    print("\n--- Creación de arrays desde listas ---\n")

    # Array 1D desde una lista
    lista = [10, 20, 30, 40, 50]
    arr = np.array(lista)
    print(f"Lista original: {lista}")
    print(f"np.array(lista): {arr}")
    print(f"  dtype: {arr.dtype}")

    # Array desde lista anidada (2D)
    lista_2d = [[1, 2, 3], [4, 5, 6]]
    arr_2d = np.array(lista_2d)
    print(f"\nLista anidada: {lista_2d}")
    print(f"np.array(lista_2d):\n{arr_2d}")
    print(f"  shape: {arr_2d.shape}  (2 filas, 3 columnas)")

    # Especificando dtype explícitamente
    arr_float = np.array([1, 2, 3], dtype=np.float32)
    print(f"\nnp.array([1, 2, 3], dtype=np.float32): {arr_float}")
    print(f"  dtype: {arr_float.dtype}")

    arr_complex = np.array([1, 2, 3], dtype=np.complex128)
    print(f"\nnp.array([1, 2, 3], dtype=np.complex128): {arr_complex}")
    print(f"  dtype: {arr_complex.dtype}")

    print()
    print("Punto clave:")
    print("  - np.array() convierte listas (y listas anidadas) en arrays.")
    print("  - El parámetro dtype= permite forzar el tipo de dato.")


def demo_multidimensional():
    """Crea arrays 2D y 3D mostrando shape, ndim y size."""
    print("\n--- Arrays multidimensionales ---\n")

    # Array 2D
    arr_2d = np.array([[1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12]])
    print("Array 2D (3x4):")
    print(arr_2d)
    print(f"  shape: {arr_2d.shape}   → 3 filas, 4 columnas")
    print(f"  ndim:  {arr_2d.ndim}       → 2 dimensiones")
    print(f"  size:  {arr_2d.size}      → 12 elementos en total")
    print(f"  dtype: {arr_2d.dtype}")

    # Array 3D
    arr_3d = np.array([[[1, 2], [3, 4]],
                        [[5, 6], [7, 8]]])
    print(f"\nArray 3D (2x2x2):")
    print(arr_3d)
    print(f"  shape: {arr_3d.shape}  → 2 bloques, 2 filas, 2 columnas")
    print(f"  ndim:  {arr_3d.ndim}       → 3 dimensiones")
    print(f"  size:  {arr_3d.size}       → 8 elementos en total")

    print()
    print("Atributos clave de un ndarray:")
    print("  - shape: tupla con el tamaño de cada dimensión")
    print("  - ndim:  número de dimensiones (ejes)")
    print("  - size:  cantidad total de elementos")
    print("  - dtype: tipo de dato de los elementos")


def demo_utility_functions():
    """Demuestra funciones utilitarias: zeros, ones, full, arange, linspace, eye, random."""
    print("\n--- Funciones utilitarias de creación ---\n")

    # np.zeros
    z = np.zeros((2, 3))
    print(f"np.zeros((2, 3)):\n{z}")
    print(f"  dtype: {z.dtype}  (float64 por defecto)\n")

    # np.ones
    o = np.ones((3, 2), dtype=np.int32)
    print(f"np.ones((3, 2), dtype=np.int32):\n{o}")
    print(f"  dtype: {o.dtype}\n")

    # np.full
    f = np.full((2, 4), 7.5)
    print(f"np.full((2, 4), 7.5):\n{f}")
    print(f"  Rellena con el valor indicado.\n")

    # np.arange
    a = np.arange(0, 20, 3)
    print(f"np.arange(0, 20, 3): {a}")
    print(f"  Similar a range(), pero devuelve un array. Soporta pasos flotantes.\n")

    a_float = np.arange(0, 1, 0.2)
    print(f"np.arange(0, 1, 0.2): {a_float}")
    print(f"  Cuidado: con flotantes pueden aparecer errores de redondeo.\n")

    # np.linspace
    lin = np.linspace(0, 1, 5)
    print(f"np.linspace(0, 1, 5): {lin}")
    print(f"  Genera 5 valores equiespaciados entre 0 y 1 (ambos incluidos).\n")

    lin10 = np.linspace(0, 100, 11)
    print(f"np.linspace(0, 100, 11): {lin10}")
    print(f"  Más seguro que arange para pasos flotantes.\n")

    # np.eye
    eye = np.eye(4)
    print(f"np.eye(4) — Matriz identidad 4x4:\n{eye}\n")

    # np.random con default_rng(42)
    rng = np.random.default_rng(42)

    rand_uniform = rng.random((2, 3))
    print(f"rng.random((2, 3)) — Uniformes en [0, 1):\n{rand_uniform}\n")

    rand_int = rng.integers(0, 100, size=(2, 5))
    print(f"rng.integers(0, 100, size=(2, 5)) — Enteros aleatorios en [0, 100):\n{rand_int}\n")

    rand_normal = rng.standard_normal((3, 3))
    print(f"rng.standard_normal((3, 3)) — Distribución normal estándar:\n{rand_normal}\n")

    print("Resumen de funciones de creación:")
    print("  np.zeros(shape)        → array de ceros")
    print("  np.ones(shape)         → array de unos")
    print("  np.full(shape, valor)  → array relleno con un valor")
    print("  np.arange(ini, fin, paso) → secuencia (como range)")
    print("  np.linspace(ini, fin, n)  → n valores equiespaciados")
    print("  np.eye(n)              → matriz identidad n×n")
    print("  rng.random(shape)      → aleatorios uniformes [0,1)")
    print("  rng.integers(a, b, size) → enteros aleatorios [a, b)")
    print("  rng.standard_normal(shape) → normal estándar (μ=0, σ=1)")
