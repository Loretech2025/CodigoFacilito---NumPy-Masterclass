"""
Módulo 2 — Arreglos NumPy
=========================
Atributos, indexing, reshaping, operaciones, agregaciones,
broadcasting, boolean/fancy indexing.

Ejecutar:
    python main.py
"""

from src.atributos import demo_attributes, demo_memory_comparison
from src.indexing import (
    demo_indexing_1d,
    demo_indexing_2d,
    demo_truncation_warning,
    demo_views_vs_copies,
)
from src.reshaping import (
    demo_reshape,
    demo_flatten_ravel,
    demo_newaxis,
    demo_practical_reshape,
)
from src.operaciones import demo_arithmetic, demo_performance, demo_ufuncs
from src.agregaciones import demo_reductions, demo_cumulative, demo_axis
from src.broadcasting import (
    demo_broadcasting_rules,
    demo_outer_product,
    demo_row_normalization,
    demo_incompatible,
)
from src.boolean_fancy import (
    demo_boolean_indexing,
    demo_where,
    demo_fancy_indexing,
    demo_sort_argsort,
)


def print_header(title: str) -> None:
    """Imprimir encabezado de sección principal."""
    width = 60
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║       MÓDULO 2 — ARREGLOS NUMPY                        ║")
    print("║  Atributos · Indexing · Reshaping · Operaciones         ║")
    print("║  Agregaciones · Broadcasting · Boolean/Fancy Indexing   ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # --- 1. Atributos ---
    print_header("1. ATRIBUTOS DE ARREGLOS")
    demo_attributes()
    demo_memory_comparison()

    # --- 2. Indexing ---
    print_header("2. INDEXING Y SLICING")
    demo_indexing_1d()
    demo_indexing_2d()
    demo_truncation_warning()
    demo_views_vs_copies()

    # --- 3. Reshaping ---
    print_header("3. RESHAPING")
    demo_reshape()
    demo_flatten_ravel()
    demo_newaxis()
    demo_practical_reshape()

    # --- 4. Operaciones ---
    print_header("4. OPERACIONES ARITMÉTICAS Y UFUNCS")
    demo_arithmetic()
    demo_performance()
    demo_ufuncs()

    # --- 5. Agregaciones ---
    print_header("5. AGREGACIONES Y REDUCCIONES")
    demo_reductions()
    demo_cumulative()
    demo_axis()

    # --- 6. Broadcasting ---
    print_header("6. BROADCASTING")
    demo_broadcasting_rules()
    demo_outer_product()
    demo_row_normalization()
    demo_incompatible()

    # --- 7. Boolean / Fancy Indexing ---
    print_header("7. BOOLEAN Y FANCY INDEXING")
    demo_boolean_indexing()
    demo_where()
    demo_fancy_indexing()
    demo_sort_argsort()

    print("\n" + "=" * 60)
    print("  Fin del Módulo 2 — ¡Arreglos NumPy dominados!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
