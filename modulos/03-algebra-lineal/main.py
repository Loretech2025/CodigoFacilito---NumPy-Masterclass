"""
Módulo 3 — Álgebra Lineal con NumPy
====================================
Vectores, matrices, normas, inversas, eigenvalores y SVD.
"""

from src.vectores_matrices import (
    demo_vectors,
    demo_matrices,
    demo_special_matrices,
    demo_transpose,
)
from src.productos import demo_dot_product, demo_matmul, demo_at_vs_star
from src.normas import demo_vector_norms, demo_normalization, demo_matrix_norms
from src.inversas_det import (
    demo_identity,
    demo_inverse,
    demo_determinant,
    demo_rank,
    demo_solve,
)
from src.eigen import demo_eigendecomposition, demo_eigh
from src.svd import demo_svd, demo_low_rank, demo_applications


def main():
    print("=" * 60)
    print("  MÓDULO 3 — ÁLGEBRA LINEAL CON NUMPY")
    print("=" * 60)

    # ── Sección 1: Vectores y Matrices ────────────────────────
    print("\n" + "=" * 60)
    print("  SECCIÓN 1: VECTORES Y MATRICES")
    print("=" * 60)
    demo_vectors()
    demo_matrices()
    demo_special_matrices()
    demo_transpose()

    # ── Sección 2: Productos ──────────────────────────────────
    print("\n" + "=" * 60)
    print("  SECCIÓN 2: PRODUCTOS (DOT, MATMUL, @ vs *)")
    print("=" * 60)
    demo_dot_product()
    demo_matmul()
    demo_at_vs_star()

    # ── Sección 3: Normas ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SECCIÓN 3: NORMAS Y NORMALIZACIÓN")
    print("=" * 60)
    demo_vector_norms()
    demo_normalization()
    demo_matrix_norms()

    # ── Sección 4: Inversas, Determinante y Sistemas ──────────
    print("\n" + "=" * 60)
    print("  SECCIÓN 4: IDENTIDAD, INVERSA, DETERMINANTE, RANGO Y SOLVE")
    print("=" * 60)
    demo_identity()
    demo_inverse()
    demo_determinant()
    demo_rank()
    demo_solve()

    # ── Sección 5: Eigenvalores ───────────────────────────────
    print("\n" + "=" * 60)
    print("  SECCIÓN 5: EIGENVALORES Y EIGENVECTORES")
    print("=" * 60)
    demo_eigendecomposition()
    demo_eigh()

    # ── Sección 6: SVD ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SECCIÓN 6: SVD (DESCOMPOSICIÓN EN VALORES SINGULARES)")
    print("=" * 60)
    demo_svd()
    demo_low_rank()
    demo_applications()

    print("\n" + "=" * 60)
    print("  FIN DEL MÓDULO 3 — ÁLGEBRA LINEAL")
    print("=" * 60)


if __name__ == "__main__":
    main()
