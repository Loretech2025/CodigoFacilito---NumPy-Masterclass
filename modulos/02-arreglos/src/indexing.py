"""Indexing y slicing en arreglos NumPy."""

import numpy as np


def demo_indexing_1d():
    """Indexación básica en arreglos 1D."""
    print("\n--- Indexing en 1D ---\n")

    arr = np.arange(10, 20)
    print(f"Arreglo: {arr}\n")

    print(f"arr[3]    (cuarto elemento)       : {arr[3]}")
    print(f"arr[-1]   (último elemento)       : {arr[-1]}")
    print(f"arr[2:7]  (del índice 2 al 6)     : {arr[2:7]}")
    print(f"arr[::2]  (cada 2 elementos)      : {arr[::2]}")
    print(f"arr[::-1] (arreglo invertido)      : {arr[::-1]}")

    print("\nRecuerda: el índice final en slicing es exclusivo (no se incluye).")


def demo_indexing_2d():
    """Indexación en arreglos 2D (matrices)."""
    print("\n--- Indexing en 2D ---\n")

    matrix = np.arange(1, 13).reshape(3, 4)
    print(f"Matriz (3×4):\n{matrix}\n")

    print(f"matrix[0, 2]   (fila 0, col 2)    : {matrix[0, 2]}")
    print(f"matrix[1, :]   (toda la fila 1)    : {matrix[1, :]}")
    print(f"matrix[:, 3]   (toda la columna 3) : {matrix[:, 3]}")

    sub = matrix[0:2, 1:3]
    print(f"\nmatrix[0:2, 1:3] (submatriz 2×2):\n{sub}")
    print("\nEsta submatriz toma filas 0-1 y columnas 1-2.")


def demo_truncation_warning():
    """Mostrar truncamiento al asignar floats en un arreglo de enteros."""
    print("\n--- Advertencia: truncamiento float → int ---\n")

    arr = np.array([1, 2, 3, 4, 5])
    print(f"Arreglo original (dtype={arr.dtype}): {arr}")

    arr[2] = 3.99
    print(f"Después de arr[2] = 3.99:            {arr}")
    print(f"arr[2] ahora vale {arr[2]}, NO 3.99 ni 4.")

    print("\nNumPy trunca (no redondea) al asignar un float a un arreglo de enteros.")
    print("Si necesitas decimales, crea el arreglo con dtype=float.")


def demo_views_vs_copies():
    """Demostrar que los slices son vistas y .copy() crea copias independientes."""
    print("\n--- Vistas vs Copias ---\n")

    original = np.array([10, 20, 30, 40, 50])
    print(f"Original: {original}")

    # Vista (slice)
    vista = original[1:4]
    print(f"Vista (original[1:4]): {vista}")

    vista[0] = 999
    print(f"\nDespués de vista[0] = 999:")
    print(f"  Vista:    {vista}")
    print(f"  Original: {original}")
    print("  ¡El original también cambió! Los slices son VISTAS de la misma memoria.")

    # Copia independiente
    original2 = np.array([10, 20, 30, 40, 50])
    copia = original2[1:4].copy()
    copia[0] = 888
    print(f"\nCon .copy():")
    print(f"  Copia:    {copia}")
    print(f"  Original: {original2}")
    print("  El original NO cambió. .copy() crea un arreglo independiente.")

    print("\nRegla: si necesitas modificar un slice sin afectar el original, usa .copy().")
