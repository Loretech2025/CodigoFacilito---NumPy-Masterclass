"""Agregaciones y reducciones en NumPy."""

import numpy as np


def demo_reductions():
    """Funciones de reducción en un arreglo 1D."""
    print("\n--- Reducciones (1D) ---\n")

    rng = np.random.default_rng(42)
    arr = rng.integers(1, 100, size=10)
    print(f"Arreglo: {arr}\n")

    print(f"np.sum(arr)  : {np.sum(arr)}")
    print(f"np.min(arr)  : {np.min(arr)}")
    print(f"np.max(arr)  : {np.max(arr)}")
    print(f"np.mean(arr) : {np.mean(arr):.2f}")
    print(f"np.std(arr)  : {np.std(arr):.2f}")
    print(f"np.var(arr)  : {np.var(arr):.2f}")

    print(f"\nnp.argmin(arr) : {np.argmin(arr)}  (índice del mínimo)")
    print(f"np.argmax(arr) : {np.argmax(arr)}  (índice del máximo)")

    print("\nEstas funciones colapsan todo el arreglo a un solo valor.")


def demo_cumulative():
    """Funciones acumulativas: cumsum y cumprod."""
    print("\n--- Funciones acumulativas ---\n")

    arr = np.array([1, 2, 3, 4, 5])
    print(f"Arreglo: {arr}\n")

    print(f"np.cumsum(arr)  : {np.cumsum(arr)}")
    print("  → [1, 1+2, 1+2+3, 1+2+3+4, 1+2+3+4+5]")

    print(f"\nnp.cumprod(arr) : {np.cumprod(arr)}")
    print("  → [1, 1×2, 1×2×3, 1×2×3×4, 1×2×3×4×5]")

    print("\ncumsum es útil para calcular totales acumulados (ej. ventas mensuales).")


def demo_axis():
    """Agregaciones por eje en una matriz 2D."""
    print("\n--- Agregaciones por eje (axis) ---\n")

    matrix = np.arange(1, 13).reshape(3, 4)
    print(f"Matriz (3×4):\n{matrix}\n")

    print(f"np.sum(matrix)           : {np.sum(matrix)}  (suma total)")

    sum_axis0 = np.sum(matrix, axis=0)
    print(f"\nnp.sum(matrix, axis=0)   : {sum_axis0}")
    print("  → Suma por COLUMNAS (colapsa las filas)")
    print(f"  Resultado shape: {sum_axis0.shape}  (una suma por cada columna)")

    sum_axis1 = np.sum(matrix, axis=1)
    print(f"\nnp.sum(matrix, axis=1)   : {sum_axis1}")
    print("  → Suma por FILAS (colapsa las columnas)")
    print(f"  Resultado shape: {sum_axis1.shape}  (una suma por cada fila)")

    mean_axis0 = np.mean(matrix, axis=0)
    mean_axis1 = np.mean(matrix, axis=1)
    print(f"\nnp.mean(matrix, axis=0)  : {mean_axis0}")
    print(f"np.mean(matrix, axis=1)  : {mean_axis1}")

    print("\nRegla mnemotécnica:")
    print("  axis=0 → opera 'hacia abajo' (colapsa filas)    → resultado tiene shape de columnas")
    print("  axis=1 → opera 'hacia la derecha' (colapsa cols) → resultado tiene shape de filas")
