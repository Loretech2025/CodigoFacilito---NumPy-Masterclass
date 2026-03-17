"""Broadcasting: cómo NumPy opera con arreglos de distinta forma."""

import numpy as np


def demo_broadcasting_rules():
    """Demostrar las 3 reglas de broadcasting."""
    print("\n--- Broadcasting: las 3 reglas ---\n")

    # Regla 1: escalar + arreglo
    arr = np.array([1, 2, 3, 4])
    print(f"arr = {arr}")
    print(f"arr + 10 = {arr + 10}")
    print("  → El escalar 10 se 'estira' para coincidir con shape (4,)\n")

    # Regla 2: 1D + 2D
    matrix = np.arange(1, 7).reshape(2, 3)
    vec = np.array([10, 20, 30])
    print(f"Matriz (2×3):\n{matrix}")
    print(f"Vector (3,): {vec}")
    print(f"\nMatriz + Vector:\n{matrix + vec}")
    print("  → El vector se repite por cada fila\n")

    # Explicación de las reglas
    print("Las 3 reglas de broadcasting:")
    print("  1. Si los arreglos tienen distinto número de dimensiones,")
    print("     se agrega un 1 al inicio del shape del más pequeño.")
    print("  2. Si los tamaños no coinciden en alguna dimensión,")
    print("     se estira el arreglo que tiene tamaño 1 en esa dimensión.")
    print("  3. Si los tamaños no coinciden y ninguno es 1, → ERROR.")

    # Ejemplo detallado
    print(f"\nEjemplo detallado:")
    print(f"  matrix.shape = {matrix.shape}  (2, 3)")
    print(f"  vec.shape    = {vec.shape}      (3,)")
    print(f"  Paso 1: vec se convierte a (1, 3)")
    print(f"  Paso 2: vec se estira a (2, 3) repitiendo la fila")
    print(f"  Resultado: operación element-wise normal")


def demo_outer_product():
    """Producto externo usando broadcasting."""
    print("\n--- Producto externo con broadcasting ---\n")

    a = np.array([1, 2, 3])
    b = np.array([10, 20, 30, 40])
    print(f"a = {a}  — shape {a.shape}")
    print(f"b = {b}  — shape {b.shape}")

    outer = a[:, np.newaxis] * b[np.newaxis, :]
    print(f"\na[:, newaxis] shape: {a[:, np.newaxis].shape}  (columna)")
    print(f"b[newaxis, :] shape: {b[np.newaxis, :].shape}  (fila)")
    print(f"\nProducto externo (a[:, None] * b[None, :]):\n{outer}")
    print(f"Resultado shape: {outer.shape}")

    # Verificar con np.outer
    print(f"\nVerificación con np.outer(a, b):\n{np.outer(a, b)}")
    print("\nEl producto externo multiplica cada elemento de a con cada elemento de b.")


def demo_row_normalization():
    """Normalizar cada fila de una matriz por su media usando broadcasting."""
    print("\n--- Normalización por filas con broadcasting ---\n")

    rng = np.random.default_rng(42)
    matrix = rng.integers(1, 50, size=(3, 4))
    print(f"Matriz original:\n{matrix}\n")

    medias = matrix.mean(axis=1)
    print(f"Media de cada fila: {medias}")
    print(f"Shape de medias:    {medias.shape}")

    # Necesitamos shape (3,1) para broadcasting con (3,4)
    medias_col = medias[:, np.newaxis]
    print(f"Medias como columna (shape {medias_col.shape}):\n{medias_col}")

    normalizada = matrix - medias_col
    print(f"\nMatriz normalizada (cada fila - su media):\n{np.round(normalizada, 2)}")

    # Verificar
    print(f"\nVerificación — media de cada fila normalizada:")
    print(f"  {np.round(normalizada.mean(axis=1), 10)}")
    print("  (todas deberían ser ~0)")


def demo_incompatible():
    """Mostrar un error de broadcasting con shapes incompatibles."""
    print("\n--- Error de broadcasting ---\n")

    a = np.ones((3, 4))
    b = np.ones((2, 3))
    print(f"a.shape = {a.shape}")
    print(f"b.shape = {b.shape}")

    try:
        resultado = a + b
    except ValueError as e:
        print(f"\nValueError: {e}")

    print("\nExplicación:")
    print("  (3, 4) vs (2, 3)")
    print("  Dimensión 0: 3 vs 2 → ninguno es 1 → INCOMPATIBLE")
    print("  Dimensión 1: 4 vs 3 → ninguno es 1 → INCOMPATIBLE")
    print("\nPara que broadcasting funcione, en cada dimensión los tamaños")
    print("deben ser iguales o uno de ellos debe ser 1.")
