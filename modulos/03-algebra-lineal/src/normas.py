"""Normas de vectores y matrices, y normalización."""

import numpy as np

np.set_printoptions(precision=4, suppress=True)


# ---------------------------------------------------------------------------
# Normas vectoriales
# ---------------------------------------------------------------------------
def demo_vector_norms():
    print("\n--- Normas vectoriales: L1, L2 e infinito ---\n")

    v = np.array([3, -4, 0, 5])
    print(f"v = {v}\n")

    # L1
    l1 = np.linalg.norm(v, ord=1)
    l1_manual = np.sum(np.abs(v))
    print(f"Norma L1 (ord=1):   {l1}")
    print(f"  Verificación manual: |3| + |-4| + |0| + |5| = {l1_manual}\n")

    # L2 (euclidiana)
    l2 = np.linalg.norm(v, ord=2)
    l2_default = np.linalg.norm(v)  # ord=2 es el default
    l2_manual = np.sqrt(np.sum(v**2))
    print(f"Norma L2 (ord=2):   {l2}")
    print(f"  Sin especificar ord: {l2_default}  (L2 es el default)")
    print(f"  Verificación manual: sqrt(9 + 16 + 0 + 25) = sqrt(50) = {l2_manual:.4f}\n")

    # L-infinito
    linf = np.linalg.norm(v, ord=np.inf)
    linf_manual = np.max(np.abs(v))
    print(f"Norma L∞ (ord=inf): {linf}")
    print(f"  Verificación manual: max(|3|, |-4|, |0|, |5|) = {linf_manual}")

    print("\n>> L1 = suma de valores absolutos")
    print("   L2 = raíz de la suma de cuadrados (distancia euclidiana)")
    print("   L∞ = valor absoluto máximo")


# ---------------------------------------------------------------------------
# Normalización
# ---------------------------------------------------------------------------
def demo_normalization():
    print("\n--- Normalización: vector unitario ---\n")

    v = np.array([3.0, 4.0])
    print(f"v = {v}")
    print(f"||v|| = {np.linalg.norm(v)}")

    v_hat = v / np.linalg.norm(v)
    print(f"\nv normalizado (v̂ = v / ||v||): {v_hat}")
    print(f"||v̂|| = {np.linalg.norm(v_hat):.4f}")

    print(f"\nVerificación: np.allclose(||v̂||, 1.0) → {np.allclose(np.linalg.norm(v_hat), 1.0)}")
    print("\n>> Normalizar un vector lo convierte en un vector unitario (norma = 1)")
    print("   que mantiene la misma dirección.")


# ---------------------------------------------------------------------------
# Normas matriciales
# ---------------------------------------------------------------------------
def demo_matrix_norms():
    print("\n--- Norma matricial: Frobenius ---\n")

    A = np.array([[1, 2],
                  [3, 4]])
    print(f"A:\n{A}\n")

    fro = np.linalg.norm(A, ord='fro')
    fro_manual = np.sqrt(np.sum(A**2))
    print(f"Norma Frobenius (ord='fro'): {fro:.4f}")
    print(f"  Verificación manual: sqrt(1² + 2² + 3² + 4²) = sqrt({np.sum(A**2)}) = {fro_manual:.4f}")

    print("\n>> La norma de Frobenius es la extensión de la norma L2 a matrices:")
    print("   sqrt de la suma de todos los elementos al cuadrado.")
