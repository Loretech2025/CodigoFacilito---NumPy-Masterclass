"""Atributos fundamentales de los arreglos NumPy."""

import numpy as np


def demo_attributes():
    """Crear un arreglo 3×4 y mostrar sus atributos principales."""
    print("\n--- Atributos de un arreglo NumPy ---\n")

    arr = np.arange(1, 13).reshape(3, 4)
    print(f"Arreglo:\n{arr}\n")

    print(f"ndim     (número de dimensiones) : {arr.ndim}")
    print(f"shape    (forma del arreglo)      : {arr.shape}")
    print(f"size     (total de elementos)     : {arr.size}")
    print(f"dtype    (tipo de dato)           : {arr.dtype}")
    print(f"itemsize (bytes por elemento)     : {arr.itemsize}")
    print(f"nbytes   (bytes totales)          : {arr.nbytes}")
    print(f"strides  (salto en bytes por dim) : {arr.strides}")

    print("\n📌 Nota: strides indica cuántos bytes hay que saltar en memoria")
    print(f"   para avanzar en cada dimensión.")
    print(f"   - Eje 0 (filas):    {arr.strides[0]} bytes = {arr.shape[1]} elementos × {arr.itemsize} bytes")
    print(f"   - Eje 1 (columnas): {arr.strides[1]} bytes = 1 elemento × {arr.itemsize} bytes")


def demo_memory_comparison():
    """Comparar el uso de memoria con distintos dtypes."""
    print("\n--- Comparación de memoria por dtype ---\n")

    base = np.arange(1, 13).reshape(3, 4)

    arr_i64 = base.astype(np.int64)
    arr_i32 = base.astype(np.int32)
    arr_i8 = base.astype(np.int8)

    print(f"{'dtype':<10} {'itemsize':>10} {'nbytes':>10}")
    print("-" * 32)
    for label, a in [("int64", arr_i64), ("int32", arr_i32), ("int8", arr_i8)]:
        print(f"{label:<10} {a.itemsize:>10} {a.nbytes:>10}")

    print(f"\nEl arreglo int64 usa {arr_i64.nbytes // arr_i8.nbytes}× más memoria que int8.")
    print("Elegir el dtype correcto ahorra memoria en datasets grandes.")
