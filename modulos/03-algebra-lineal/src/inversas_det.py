"""Identidad, inversa, determinante, rango y sistemas de ecuaciones."""

import numpy as np

np.set_printoptions(precision=4, suppress=True)


# ---------------------------------------------------------------------------
# Matriz identidad
# ---------------------------------------------------------------------------
def demo_identity():
    print("\n--- Matriz identidad ---\n")

    I = np.eye(3)
    print(f"I = np.eye(3):\n{I}\n")

    A = np.array([[2, 5, 1],
                  [4, 3, 7],
                  [8, 6, 9]])
    print(f"A:\n{A}\n")

    AI = A @ I
    print(f"A @ I:\n{AI}\n")
    print(f"¿A @ I == A?  → {np.array_equal(A, AI)}")
    print("\n>> La identidad es el '1' del álgebra matricial: A @ I = I @ A = A.")


# ---------------------------------------------------------------------------
# Inversa
# ---------------------------------------------------------------------------
def demo_inverse():
    print("\n--- Inversa de una matriz ---\n")

    A = np.array([[1.0, 2.0],
                  [3.0, 4.0]])
    print(f"A:\n{A}\n")

    A_inv = np.linalg.inv(A)
    print(f"A⁻¹ = np.linalg.inv(A):\n{A_inv}\n")

    product = A @ A_inv
    print(f"A @ A⁻¹:\n{product}\n")
    print(f"¿A @ A⁻¹ ≈ I?  → {np.allclose(product, np.eye(2))}")

    print("\n>> La inversa A⁻¹ cumple A @ A⁻¹ = A⁻¹ @ A = I.")
    print("   Solo existe para matrices cuadradas no singulares (det ≠ 0).")


# ---------------------------------------------------------------------------
# Determinante
# ---------------------------------------------------------------------------
def demo_determinant():
    print("\n--- Determinante ---\n")

    A = np.array([[1.0, 2.0],
                  [3.0, 4.0]])
    det_A = np.linalg.det(A)
    print(f"A:\n{A}")
    print(f"det(A) = {det_A:.4f}")
    print(f"  Cálculo manual: (1)(4) - (2)(3) = {1*4 - 2*3}\n")

    # Matriz singular
    S = np.array([[1.0, 2.0],
                  [2.0, 4.0]])  # fila 2 = 2 * fila 1
    det_S = np.linalg.det(S)
    print(f"Matriz singular S (fila 2 = 2 × fila 1):\n{S}")
    print(f"det(S) = {det_S:.4f}")
    print("\n>> Si det(A) ≈ 0, la matriz es singular y NO tiene inversa.")


# ---------------------------------------------------------------------------
# Rango
# ---------------------------------------------------------------------------
def demo_rank():
    print("\n--- Rango de una matriz ---\n")

    A = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 10]])  # filas linealmente independientes
    rank_A = np.linalg.matrix_rank(A)
    print(f"A (filas independientes):\n{A}")
    print(f"  Rango = {rank_A}  (rango completo para 3×3)\n")

    B = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [5, 7, 9]])  # fila 3 = fila 1 + fila 2
    rank_B = np.linalg.matrix_rank(B)
    print(f"B (fila 3 = fila 1 + fila 2):\n{B}")
    print(f"  Rango = {rank_B}  (deficiente en rango)\n")

    print(">> El rango indica cuántas filas/columnas son linealmente independientes.")
    print("   Una matriz de rango completo (n×n con rango n) es invertible.")


# ---------------------------------------------------------------------------
# Resolver sistemas de ecuaciones
# ---------------------------------------------------------------------------
def demo_solve():
    print("\n--- Resolver Ax = b con np.linalg.solve ---\n")

    # Sistema: 2x + y = 5
    #          x + 3y = 7
    A = np.array([[2.0, 1.0],
                  [1.0, 3.0]])
    b = np.array([5.0, 7.0])

    print("Sistema de ecuaciones:")
    print("  2x +  y = 5")
    print("   x + 3y = 7\n")

    x = np.linalg.solve(A, b)
    print(f"A:\n{A}")
    print(f"b: {b}")
    print(f"\nx = np.linalg.solve(A, b) = {x}")
    print(f"  → x = {x[0]:.4f}, y = {x[1]:.4f}\n")

    # Verificación
    print(f"Verificación: A @ x = {A @ x}")
    print(f"¿A @ x ≈ b?  → {np.allclose(A @ x, b)}")

    # ¿Por qué solve y no inv?
    print("\n>> ¿Por qué usar solve en vez de inv(A) @ b?")
    print("   1. solve es más rápido (usa factorización LU internamente)")
    print("   2. solve es numéricamente más estable")
    print("   3. solve consume menos memoria")
    print("   Regla: SIEMPRE preferir np.linalg.solve sobre inv(A) @ b.")
