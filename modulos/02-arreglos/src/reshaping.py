"""Reshaping: cambiar la forma de arreglos NumPy."""

import numpy as np


def demo_reshape():
    """Demostrar reshape básico y el uso de -1."""
    print("\n--- Reshape ---\n")

    arr = np.arange(1, 13)
    print(f"Arreglo original (12 elementos): {arr}")

    r1 = arr.reshape(3, 4)
    print(f"\nreshape(3, 4):\n{r1}")

    r2 = arr.reshape(-1, 2)
    print(f"\nreshape(-1, 2) — NumPy calcula la primera dimensión automáticamente:")
    print(f"  Resultado shape: {r2.shape}")
    print(r2)

    r3 = arr.reshape(2, -1)
    print(f"\nreshape(2, -1) — NumPy calcula la segunda dimensión:")
    print(f"  Resultado shape: {r3.shape}")
    print(r3)

    print("\nError de reshape (tamaño incompatible):")
    try:
        arr.reshape(5, 5)
    except ValueError as e:
        print(f"  ValueError: {e}")
    print("  12 elementos no se pueden reorganizar en 5×5 (necesitaría 25).")


def demo_flatten_ravel():
    """Mostrar la diferencia entre flatten (copia) y ravel (vista)."""
    print("\n--- flatten() vs ravel() ---\n")

    matrix = np.arange(1, 7).reshape(2, 3)
    print(f"Matriz:\n{matrix}\n")

    flat = matrix.flatten()
    rav = matrix.ravel()
    print(f"flatten(): {flat}")
    print(f"ravel():   {rav}")

    # Modificar y ver efecto
    flat[0] = 999
    print(f"\nDespués de flat[0] = 999:")
    print(f"  flatten resultado: {flat}")
    print(f"  Matriz original:\n{matrix}")
    print("  flatten() devuelve una COPIA — el original no cambia.")

    rav[0] = 888
    print(f"\nDespués de rav[0] = 888:")
    print(f"  ravel resultado: {rav}")
    print(f"  Matriz original:\n{matrix}")
    print("  ravel() devuelve una VISTA — el original SÍ cambia.")


def demo_newaxis():
    """Usar np.newaxis para agregar dimensiones."""
    print("\n--- np.newaxis ---\n")

    arr = np.array([1, 2, 3])
    print(f"Arreglo 1D: {arr}  — shape: {arr.shape}")

    row = arr[np.newaxis, :]
    print(f"\nVector fila (arr[np.newaxis, :]):")
    print(f"  {row}  — shape: {row.shape}")

    col = arr[:, np.newaxis]
    print(f"\nVector columna (arr[:, np.newaxis]):")
    print(f"  {col}")
    print(f"  shape: {col.shape}")

    print("\nnp.newaxis agrega una dimensión de tamaño 1.")
    print("Esto es clave para broadcasting y operaciones entre vectores.")


def demo_practical_reshape():
    """Ejemplo práctico: aplanar una 'imagen' y reshape de batches."""
    print("\n--- Reshape práctico ---\n")

    # Simular una imagen 4×4 con 3 canales (RGB)
    rng = np.random.default_rng(42)
    imagen = rng.integers(0, 256, size=(4, 4, 3), dtype=np.uint8)
    print(f"Imagen simulada shape: {imagen.shape}  (alto × ancho × canales)")
    print(f"Total de valores: {imagen.size}")

    flat = imagen.reshape(-1)
    print(f"\nAplanada con reshape(-1): shape {flat.shape}")
    print(f"Primeros 12 valores: {flat[:12]}")

    # Batch reshape
    print("\n--- Batch reshape ---")
    datos = np.arange(24)
    print(f"Datos lineales (24 elementos): {datos}")

    batch = datos.reshape(4, 2, 3)
    print(f"\nReorganizados en batch de 4 muestras de 2×3:")
    print(f"  Shape: {batch.shape}")
    for i in range(batch.shape[0]):
        print(f"  Muestra {i}: {batch[i].tolist()}")

    print("\nreshape es fundamental para preparar datos en machine learning.")
