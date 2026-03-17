"""Producto punto, producto matricial y la diferencia entre @ y *."""

import numpy as np

np.set_printoptions(precision=4, suppress=True)


# ---------------------------------------------------------------------------
# Producto punto (dot product)
# ---------------------------------------------------------------------------
def demo_dot_product():
    print("\n--- Producto punto (dot product) ---\n")

    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])

    dot1 = np.dot(a, b)
    dot2 = a @ b
    dot3 = np.sum(a * b)

    print(f"a = {a}")
    print(f"b = {b}")
    print(f"\nnp.dot(a, b) = {dot1}")
    print(f"a @ b        = {dot2}")
    print(f"sum(a * b)   = {dot3}   (verificación manual: 1*4 + 2*5 + 3*6 = {1*4 + 2*5 + 3*6})")

    # Conmutatividad
    print(f"\n¿Conmutativo? a @ b == b @ a → {a @ b == b @ a}")

    # Vectores ortogonales
    u = np.array([1, 0, 0])
    w = np.array([0, 1, 0])
    print(f"\nVectores ortogonales: u = {u}, w = {w}")
    print(f"u @ w = {u @ w}   (dot = 0 indica ortogonalidad)")


# ---------------------------------------------------------------------------
# Producto matricial (matmul)
# ---------------------------------------------------------------------------
def demo_matmul():
    print("\n--- Producto matricial (matmul) ---\n")

    A = np.array([[1, 2],
                  [3, 4],
                  [5, 6]])  # (3, 2)
    B = np.array([[7, 8, 9],
                  [10, 11, 12]])  # (2, 3)

    C = A @ B
    print(f"A (shape {A.shape}):\n{A}\n")
    print(f"B (shape {B.shape}):\n{B}\n")
    print(f"C = A @ B (shape {C.shape}):\n{C}\n")
    print(">> Regla de shapes: (m, n) @ (n, p) → (m, p)")
    print(f"   Aquí: ({A.shape[0]}, {A.shape[1]}) @ ({B.shape[0]}, {B.shape[1]}) → ({C.shape[0]}, {C.shape[1]})")

    # No conmutatividad
    print("\n--- No conmutatividad del producto matricial ---\n")
    X = np.array([[1, 2],
                  [3, 4]])
    Y = np.array([[5, 6],
                  [7, 8]])

    print(f"X @ Y:\n{X @ Y}\n")
    print(f"Y @ X:\n{Y @ X}\n")
    print(f"¿X @ Y == Y @ X?  →  {np.array_equal(X @ Y, Y @ X)}")
    print(">> En general, A @ B ≠ B @ A. El orden importa.")


# ---------------------------------------------------------------------------
# @ vs * (matmul vs element-wise)
# ---------------------------------------------------------------------------
def demo_at_vs_star():
    print("\n--- @ (matmul) vs * (elemento a elemento) ---\n")

    A = np.array([[1, 2],
                  [3, 4]])
    B = np.array([[5, 6],
                  [7, 8]])

    print(f"A:\n{A}\n")
    print(f"B:\n{B}\n")

    print(f"A @ B (producto matricial):\n{A @ B}")
    print(f"  → (1*5 + 2*7) = {1*5 + 2*7},  (1*6 + 2*8) = {1*6 + 2*8}, ...\n")

    print(f"A * B (elemento a elemento):\n{A * B}")
    print(f"  → 1*5 = {1*5},  2*6 = {2*6},  3*7 = {3*7},  4*8 = {4*8}\n")

    print(">> @ calcula el producto matricial (filas × columnas).")
    print("   * multiplica cada par de elementos en la misma posición.")
