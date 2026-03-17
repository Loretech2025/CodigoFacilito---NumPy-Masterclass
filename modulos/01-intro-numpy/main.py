"""Módulo 1 — Introducción a NumPy."""

from src.tipos import demo_python_types, demo_numpy_types, demo_type_inference
from src.creacion import demo_from_lists, demo_multidimensional, demo_utility_functions


def main():
    print("=" * 60)
    print("  MÓDULO 1 — INTRODUCCIÓN A NUMPY")
    print("=" * 60)

    # Tipos de datos
    print("\n" + "=" * 60)
    print("  SECCIÓN 1: TIPOS DE DATOS")
    print("=" * 60)
    demo_python_types()
    demo_numpy_types()
    demo_type_inference()

    # Creación de arrays
    print("\n" + "=" * 60)
    print("  SECCIÓN 2: CREACIÓN DE ARRAYS")
    print("=" * 60)
    demo_from_lists()
    demo_multidimensional()
    demo_utility_functions()

    print("\n" + "=" * 60)
    print("  FIN DEL MÓDULO 1")
    print("=" * 60)


if __name__ == "__main__":
    main()
