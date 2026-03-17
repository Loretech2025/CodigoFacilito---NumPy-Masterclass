"""Tipos de datos en Python y NumPy."""

import sys

import numpy as np


def demo_python_types():
    """Muestra los tipos numéricos nativos de Python con type() y sys.getsizeof()."""
    print("\n--- Tipos numéricos nativos de Python ---\n")

    entero = 42
    flotante = 3.14
    complejo = 2 + 3j
    booleano = True

    valores = [
        ("int", entero),
        ("float", flotante),
        ("complex", complejo),
        ("bool", booleano),
    ]

    print(f"{'Nombre':<10} {'Valor':<15} {'type()':<25} {'Bytes (sys.getsizeof)'}")
    print("-" * 75)
    for nombre, valor in valores:
        print(f"{nombre:<10} {str(valor):<15} {str(type(valor)):<25} {sys.getsizeof(valor)}")

    print()
    print("Observaciones:")
    print("  - Un int de Python ocupa ~28 bytes porque es un objeto completo.")
    print("  - Un float ocupa ~24 bytes; un complex ~32 bytes.")
    print("  - bool es subclase de int, por eso ocupa lo mismo.")
    print("  - NumPy reduce drásticamente este consumo de memoria.")


def demo_numpy_types():
    """Muestra los dtypes de NumPy con itemsize y rangos min/max."""
    print("\n--- Tipos de datos (dtypes) de NumPy ---\n")

    # Tipos enteros
    int_types = [np.int8, np.int16, np.int32, np.int64]
    # Tipos flotantes
    float_types = [np.float16, np.float32, np.float64]
    # Tipos complejos
    complex_types = [np.complex64, np.complex128]

    print("ENTEROS")
    print(f"  {'dtype':<15} {'itemsize (bytes)':<20} {'Mínimo':<25} {'Máximo'}")
    print("  " + "-" * 75)
    for dt in int_types:
        info = np.iinfo(dt)
        print(f"  {dt.__name__:<15} {np.dtype(dt).itemsize:<20} {info.min:<25} {info.max}")

    print()
    print("FLOTANTES")
    print(f"  {'dtype':<15} {'itemsize (bytes)':<20} {'Mínimo':<25} {'Máximo'}")
    print("  " + "-" * 75)
    for dt in float_types:
        info = np.finfo(dt)
        print(f"  {dt.__name__:<15} {np.dtype(dt).itemsize:<20} {str(info.min):<25} {info.max}")

    print()
    print("COMPLEJOS")
    print(f"  {'dtype':<15} {'itemsize (bytes)':<20} {'Parte real/imag usa'}")
    print("  " + "-" * 55)
    for dt in complex_types:
        size = np.dtype(dt).itemsize
        parte = size // 2
        print(f"  {dt.__name__:<15} {size:<20} float{parte * 8}")

    print()
    print("BOOLEANO")
    dt_bool = np.bool_
    print(f"  {dt_bool.__name__:<15} itemsize = {np.dtype(dt_bool).itemsize} byte")
    print(f"  Valores posibles: {np.bool_(True)}, {np.bool_(False)}")

    print()
    print("Conclusión:")
    print("  - NumPy ofrece control preciso sobre el tamaño en memoria.")
    print("  - int8 usa solo 1 byte vs ~28 bytes del int de Python.")
    print("  - Elegir el dtype correcto es clave para optimizar memoria.")


def demo_type_inference():
    """Muestra cómo NumPy infiere el dtype a partir de listas de Python."""
    print("\n--- Inferencia de tipos en NumPy ---\n")

    # Lista de enteros → int64
    lista_ints = [1, 2, 3, 4, 5]
    arr_ints = np.array(lista_ints)
    print(f"Lista de enteros:   {lista_ints}")
    print(f"  dtype inferido:   {arr_ints.dtype}  (int64 por defecto)")

    # Lista de flotantes → float64
    lista_floats = [1.0, 2.5, 3.7]
    arr_floats = np.array(lista_floats)
    print(f"\nLista de flotantes: {lista_floats}")
    print(f"  dtype inferido:   {arr_floats.dtype}  (float64 por defecto)")

    # Lista mixta int + float → float64
    lista_mixta = [1, 2.5, 3]
    arr_mixta = np.array(lista_mixta)
    print(f"\nLista mixta (int + float): {lista_mixta}")
    print(f"  dtype inferido:   {arr_mixta.dtype}  (se promueve a float64)")
    print(f"  Valores:          {arr_mixta}  (el 1 y el 3 ahora son 1. y 3.)")

    # Lista de strings → Unicode
    lista_strings = ["hola", "mundo", "numpy"]
    arr_strings = np.array(lista_strings)
    print(f"\nLista de strings:   {lista_strings}")
    print(f"  dtype inferido:   {arr_strings.dtype}  (Unicode, longitud máxima de la cadena)")

    print()
    print("Reglas de inferencia:")
    print("  - Solo enteros         → int64")
    print("  - Solo flotantes       → float64")
    print("  - Mezcla int + float   → float64 (promoción automática)")
    print("  - Strings              → Unicode (U + longitud máxima)")
    print("  - Siempre se puede forzar el dtype con el parámetro dtype=...")
