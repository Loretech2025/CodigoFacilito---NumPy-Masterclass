"""Vectores y matrices: creación, formas especiales y transposición."""

import numpy as np

np.set_printoptions(precision=4, suppress=True)


# ---------------------------------------------------------------------------
# Vectores
# ---------------------------------------------------------------------------
def demo_vectors():
    print("\n--- Vectores: 1-D, columna y fila ---\n")

    v = np.array([1, 2, 3, 4])
    print(f"Vector 1-D:          {v}")
    print(f"  shape = {v.shape}        (una sola dimensión, NO es fila ni columna)")

    col = v.reshape(-1, 1)
    print(f"\nVector columna (reshape(-1, 1)):\n{col}")
    print(f"  shape = {col.shape}")

    row = v.reshape(1, -1)
    print(f"\nVector fila (reshape(1, -1)):\n{row}")
    print(f"  shape = {row.shape}")

    print("\n>> Punto clave: un array 1-D de shape (n,) no es lo mismo que")
    print("   un vector columna (n,1) ni un vector fila (1,n).")


# ---------------------------------------------------------------------------
# Matrices
# ---------------------------------------------------------------------------
def demo_matrices():
    print("\n--- Matrices: creación y acceso ---\n")

    A = np.array([[1, 2, 3],
                  [4, 5, 6]])
    print(f"Matriz A (desde lista anidada):\n{A}")
    print(f"  shape = {A.shape}   (2 filas, 3 columnas)")

    print(f"\nAcceso a elementos:")
    print(f"  A[0, 0] = {A[0, 0]}   (primera fila, primera columna)")
    print(f"  A[1, 2] = {A[1, 2]}   (segunda fila, tercera columna)")
    print(f"  A[0]    = {A[0]}     (primera fila completa)")
    print(f"  A[:, 1] = {A[:, 1]}     (segunda columna completa)")


# ---------------------------------------------------------------------------
# Matrices especiales
# ---------------------------------------------------------------------------
def demo_special_matrices():
    print("\n--- Matrices especiales: zeros, ones, eye, diag ---\n")

    Z = np.zeros((2, 3))
    print(f"np.zeros((2, 3)):\n{Z}\n")

    O = np.ones((3, 2))
    print(f"np.ones((3, 2)):\n{O}\n")

    I = np.eye(3)
    print(f"np.eye(3) — matriz identidad:\n{I}\n")

    # np.diag para CREAR una matriz diagonal
    d = np.array([5, 10, 15])
    D = np.diag(d)
    print(f"np.diag([5, 10, 15]) — crear matriz diagonal:\n{D}\n")

    # np.diag para EXTRAER la diagonal
    M = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
    print(f"Matriz M:\n{M}")
    print(f"np.diag(M) — extraer diagonal: {np.diag(M)}")
    print("\n>> np.diag tiene doble uso: pásale un 1-D y crea matriz; pásale un 2-D y extrae diagonal.")


# ---------------------------------------------------------------------------
# Transposición
# ---------------------------------------------------------------------------
def demo_transpose():
    print("\n--- Transposición (.T) ---\n")

    A = np.array([[1, 2, 3],
                  [4, 5, 6]])
    print(f"A (shape {A.shape}):\n{A}\n")
    print(f"A.T (shape {A.T.shape}):\n{A.T}\n")

    # Caveat importante
    v = np.array([1, 2, 3])
    print(f"Vector 1-D v:   {v}   shape = {v.shape}")
    print(f"v.T:            {v.T}   shape = {v.T.shape}")
    print("\n>> ¡CUIDADO! .T en un array 1-D NO hace nada.")
    print("   Si necesitas transponer un vector, primero dale forma 2-D:")
    v2d = v.reshape(-1, 1)
    print(f"   v.reshape(-1,1).T = {v2d.T}  shape = {v2d.T.shape}")
