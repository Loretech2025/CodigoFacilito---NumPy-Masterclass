"""SVD: Descomposición en Valores Singulares."""

import numpy as np

np.set_printoptions(precision=4, suppress=True)
rng = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# SVD básico
# ---------------------------------------------------------------------------
def demo_svd():
    print("\n--- SVD: np.linalg.svd ---\n")

    A = np.array([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0],
                  [7.0, 8.0, 9.0],
                  [10.0, 11.0, 12.0]])
    print(f"A (shape {A.shape}):\n{A}\n")

    U, S, Vt = np.linalg.svd(A, full_matrices=False)

    print(f"U  shape: {U.shape}")
    print(f"S  shape: {S.shape}   (valores singulares)")
    print(f"Vt shape: {Vt.shape}\n")

    print(f"Valores singulares: {S}\n")

    # Reconstrucción: A ≈ U @ diag(S) @ Vt
    A_reconstructed = U @ np.diag(S) @ Vt
    print(f"Reconstrucción U @ diag(S) @ Vt:\n{A_reconstructed}\n")
    print(f"¿A ≈ reconstrucción? → {np.allclose(A, A_reconstructed)}")

    print("\n>> SVD descompone cualquier matriz A (m×n) en tres factores:")
    print("   A = U · Σ · Vᵀ")
    print("   U: vectores singulares izquierdos (m×k)")
    print("   Σ: valores singulares (diagonal, k×k)")
    print("   Vᵀ: vectores singulares derechos (k×n)")


# ---------------------------------------------------------------------------
# Aproximación de bajo rango
# ---------------------------------------------------------------------------
def demo_low_rank():
    print("\n--- Aproximación de bajo rango con SVD ---\n")

    # Crear una matriz con estructura clara
    A = rng.standard_normal((5, 4))
    A = A @ A.T  # (5, 5) simétrica positiva semidefinida — más interesante
    # Usemos una rectangular para mayor generalidad
    A = np.array([[9.0, 1.0, 3.0, 7.0],
                  [2.0, 8.0, 4.0, 6.0],
                  [5.0, 5.0, 5.0, 5.0],
                  [1.0, 7.0, 3.0, 9.0],
                  [4.0, 6.0, 2.0, 8.0]])
    print(f"A (shape {A.shape}):\n{A}\n")

    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    print(f"Valores singulares: {S}\n")

    for k in [1, 2, 3]:
        # Aproximación de rango k
        A_k = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
        error = np.linalg.norm(A - A_k, ord='fro')
        print(f"Rango {k}: error Frobenius = {error:.4f}")

    print(f"\n>> A medida que aumentamos k, el error disminuye.")
    print("   Los primeros valores singulares capturan la mayor parte de la información.")
    print("   Esto permite comprimir la matriz manteniendo lo esencial.")


# ---------------------------------------------------------------------------
# Aplicaciones de SVD
# ---------------------------------------------------------------------------
def demo_applications():
    print("\n--- Aplicaciones de SVD ---\n")

    print("1. REDUCCIÓN DE DIMENSIONALIDAD (PCA)")
    print("   - PCA internamente usa SVD para encontrar las direcciones")
    print("     de máxima varianza en los datos.")
    print("   - Se conservan las primeras k componentes (mayor valor singular).\n")

    print("2. COMPRESIÓN DE IMÁGENES")
    print("   - Una imagen en escala de grises es una matriz (alto × ancho).")
    print("   - Con SVD, puedes aproximarla con rango k << min(alto, ancho).")
    print("   - Ejemplo: una imagen 1000×1000 con rango 50 usa ~100k valores")
    print("     en vez de 1,000,000.\n")

    print("3. SISTEMAS DE RECOMENDACIÓN")
    print("   - La matriz usuarios × productos se descompone con SVD.")
    print("   - Los factores latentes capturan preferencias ocultas.")
    print("   - Se pueden predecir ratings faltantes reconstruyendo la matriz.\n")

    print("4. PSEUDOINVERSA (Moore-Penrose)")
    print("   - np.linalg.pinv usa SVD internamente.")
    print("   - Permite 'resolver' sistemas sobredeterminados o subdeterminados.\n")

    print(">> SVD es una de las herramientas más versátiles del álgebra lineal numérica.")
