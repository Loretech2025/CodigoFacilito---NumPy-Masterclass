"""Boolean indexing, np.where y fancy indexing."""

import numpy as np


def demo_boolean_indexing():
    """Indexación booleana con máscaras."""
    print("\n--- Boolean indexing ---\n")

    rng = np.random.default_rng(42)
    arr = rng.integers(1, 20, size=10)
    print(f"Arreglo: {arr}\n")

    mask = arr > 10
    print(f"Máscara (arr > 10): {mask}")
    print(f"Elementos > 10:     {arr[mask]}")

    # Condiciones combinadas
    mask_and = (arr > 5) & (arr < 15)
    print(f"\n(arr > 5) & (arr < 15): {arr[mask_and]}")

    mask_or = (arr < 5) | (arr > 15)
    print(f"(arr < 5) | (arr > 15): {arr[mask_or]}")

    # Negación
    mask_not = ~(arr > 10)
    print(f"~(arr > 10):            {arr[mask_not]}")

    print("\nImportante: usar & (no 'and'), | (no 'or'), ~ (no 'not').")
    print("Cada condición debe ir entre paréntesis.")

    # Conteo
    print(f"\n¿Cuántos elementos > 10? {np.sum(arr > 10)}")
    print("(True cuenta como 1, False como 0)")


def demo_where():
    """np.where: seleccionar valores según una condición."""
    print("\n--- np.where ---\n")

    arr = np.array([1, -2, 3, -4, 5, -6])
    print(f"Arreglo: {arr}\n")

    # np.where con 3 argumentos: condition, x, y
    resultado = np.where(arr > 0, arr, 0)
    print(f"np.where(arr > 0, arr, 0):  {resultado}")
    print("  → Mantiene positivos, reemplaza negativos con 0\n")

    # Reemplazar con valores personalizados
    labels = np.where(arr >= 0, "positivo", "negativo")
    print(f"np.where(arr >= 0, 'positivo', 'negativo'):")
    print(f"  {labels}\n")

    # np.where con 1 argumento: devuelve índices
    indices = np.where(arr < 0)
    print(f"np.where(arr < 0): {indices[0]}")
    print(f"  → Índices donde la condición es True")
    print(f"  → Valores en esos índices: {arr[indices]}")


def demo_fancy_indexing():
    """Indexación con arreglos de enteros (fancy indexing)."""
    print("\n--- Fancy indexing ---\n")

    arr = np.array([10, 20, 30, 40, 50, 60, 70])
    print(f"Arreglo: {arr}\n")

    indices = np.array([0, 3, 5])
    print(f"Índices: {indices}")
    print(f"arr[indices]: {arr[indices]}")

    # Fancy indexing en 2D
    print("\n--- Fancy indexing en 2D ---")
    matrix = np.arange(1, 13).reshape(3, 4)
    print(f"\nMatriz:\n{matrix}\n")

    # Seleccionar filas específicas
    filas = matrix[[0, 2]]
    print(f"matrix[[0, 2]] (filas 0 y 2):\n{filas}")

    # Seleccionar elementos específicos
    rows = np.array([0, 1, 2])
    cols = np.array([3, 1, 0])
    print(f"\nrows = {rows}, cols = {cols}")
    print(f"matrix[rows, cols]: {matrix[rows, cols]}")
    print("  → Elementos: matrix[0,3], matrix[1,1], matrix[2,0]")

    print("\nFancy indexing siempre devuelve una COPIA, no una vista.")


def demo_sort_argsort():
    """Ordenamiento con np.sort y np.argsort."""
    print("\n--- Ordenamiento: sort y argsort ---\n")

    rng = np.random.default_rng(42)
    arr = rng.integers(1, 50, size=8)
    print(f"Arreglo original: {arr}\n")

    print(f"np.sort(arr):    {np.sort(arr)}")
    print("  (devuelve copia ordenada, el original no cambia)")

    indices_ord = np.argsort(arr)
    print(f"\nnp.argsort(arr): {indices_ord}")
    print(f"  → Índices que ordenarían el arreglo")
    print(f"  → arr[argsort]: {arr[indices_ord]}")

    # Ordenar por columna en 2D
    print("\n--- Ordenar matriz por columna ---")
    datos = np.array([
        [3, 90],
        [1, 70],
        [2, 85],
        [4, 60],
    ])
    print(f"\nDatos (id, puntaje):\n{datos}\n")

    # Ordenar por primera columna (id)
    orden_id = np.argsort(datos[:, 0])
    print(f"Ordenados por columna 0 (id):\n{datos[orden_id]}")

    # Ordenar por segunda columna (puntaje)
    orden_punt = np.argsort(datos[:, 1])
    print(f"\nOrdenados por columna 1 (puntaje):\n{datos[orden_punt]}")

    print("\nargsort es clave para ordenar datos tabulares por cualquier columna.")
