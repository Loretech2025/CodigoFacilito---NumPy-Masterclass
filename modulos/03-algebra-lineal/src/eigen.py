"""Eigendescomposición: valores y vectores propios."""

import numpy as np

np.set_printoptions(precision=4, suppress=True)


# ---------------------------------------------------------------------------
# Eigendescomposición general
# ---------------------------------------------------------------------------
def demo_eigendecomposition():
    print("\n--- Eigendescomposición: np.linalg.eig ---\n")

    A = np.array([[4.0, 2.0],
                  [1.0, 3.0]])
    print(f"A:\n{A}\n")

    eigenvalues, eigenvectors = np.linalg.eig(A)
    print(f"Eigenvalores (λ): {eigenvalues}")
    print(f"\nEigenvectores (columnas de V):\n{eigenvectors}\n")

    # Verificación: A @ v = λ * v
    print("Verificación: A @ v = λ · v")
    for i in range(len(eigenvalues)):
        v = eigenvectors[:, i]
        lam = eigenvalues[i]
        Av = A @ v
        lam_v = lam * v
        ok = np.allclose(Av, lam_v)
        print(f"  λ_{i} = {lam:.4f}")
        print(f"    A @ v_{i}  = {Av}")
        print(f"    λ_{i} · v_{i} = {lam_v}")
        print(f"    ¿Iguales? → {ok}\n")

    print(">> Av = λv es la ecuación fundamental de eigenvalores.")
    print("   λ escala el vector v cuando A lo transforma.")


# ---------------------------------------------------------------------------
# Eigendescomposición para matrices simétricas
# ---------------------------------------------------------------------------
def demo_eigh():
    print("\n--- np.linalg.eigh: matrices simétricas ---\n")

    # Matriz simétrica (A = A.T)
    A = np.array([[6.0, 2.0, 1.0],
                  [2.0, 3.0, 1.0],
                  [1.0, 1.0, 1.0]])
    print(f"A (simétrica, A = A.T):\n{A}")
    print(f"¿A == A.T? → {np.array_equal(A, A.T)}\n")

    eigenvalues, eigenvectors = np.linalg.eigh(A)
    print(f"Eigenvalores: {eigenvalues}")
    print(f"  → Todos reales (garantizado para matrices simétricas)\n")

    print(f"Eigenvectores (columnas):\n{eigenvectors}\n")

    # Verificar ortogonalidad: V.T @ V ≈ I
    VtV = eigenvectors.T @ eigenvectors
    print(f"V.T @ V (debe ser ≈ I para eigenvectores ortogonales):\n{VtV}\n")
    print(f"¿V.T @ V ≈ I? → {np.allclose(VtV, np.eye(3))}")

    print("\n>> eigh vs eig:")
    print("   - eigh aprovecha la simetría → más rápido y estable")
    print("   - Eigenvalores siempre reales")
    print("   - Eigenvectores siempre ortogonales")
    print("   - Usa eigh siempre que sepas que la matriz es simétrica.")
