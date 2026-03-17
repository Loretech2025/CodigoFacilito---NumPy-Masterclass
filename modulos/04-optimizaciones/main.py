"""Módulo 4 — Optimizaciones NumPy.

Cubre: vectorización, memoria, operaciones in-place, dtypes,
preasignación, Einstein summation (einsum), y BLAS.
"""


def header(titulo: str) -> None:
    """Imprime un encabezado de sección principal."""
    ancho = 60
    print("\n" + "=" * ancho)
    print(f"  {titulo}")
    print("=" * ancho)


def main() -> None:
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   Módulo 4: Optimizaciones NumPy                       ║")
    print("║   Vectorización · Memoria · einsum · BLAS              ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # --- 1. Herramientas de Benchmarking ---
    header("1. HERRAMIENTAS DE BENCHMARKING")
    from src.benchmarks import demo_timing_tools
    demo_timing_tools()

    # --- 2. Vectorización ---
    header("2. VECTORIZACIÓN")
    from src.vectorizacion import (
        demo_loop_vs_vectorized,
        demo_elementwise,
        demo_math_functions,
        demo_conditionals,
    )
    demo_loop_vs_vectorized()
    demo_elementwise()
    demo_math_functions()
    demo_conditionals()

    # --- 3. Memoria ---
    header("3. MEMORIA Y CACHÉ")
    from src.memoria import (
        demo_c_vs_f_order,
        demo_cache_locality,
        demo_views_vs_copies,
    )
    demo_c_vs_f_order()
    demo_cache_locality()
    demo_views_vs_copies()

    # --- 4. Operaciones In-Place y dtypes ---
    header("4. OPERACIONES IN-PLACE Y DTYPES")
    from src.inplace_dtypes import (
        demo_inplace,
        demo_dtype_selection,
        demo_dtype_table,
    )
    demo_inplace()
    demo_dtype_selection()
    demo_dtype_table()

    # --- 5. Preasignación ---
    header("5. PREASIGNACIÓN DE ARRAYS")
    from src.preasignacion import (
        demo_append_antipattern,
        demo_preallocation,
        demo_list_accumulate,
    )
    demo_append_antipattern()
    demo_preallocation()
    demo_list_accumulate()

    # --- 6. Einstein Summation ---
    header("6. EINSTEIN SUMMATION (einsum)")
    from src.einsum_demo import (
        demo_einsum_basics,
        demo_einsum_operations,
        demo_batch_matmul,
        demo_optimize,
    )
    demo_einsum_basics()
    demo_einsum_operations()
    demo_batch_matmul()
    demo_optimize()

    # --- 7. BLAS ---
    header("7. BLAS — ÁLGEBRA LINEAL OPTIMIZADA")
    from src.blas import (
        demo_blas_info,
        demo_naive_vs_blas,
        demo_scaling,
    )
    demo_blas_info()
    demo_naive_vs_blas()
    demo_scaling()

    # --- Resumen Final ---
    print("\n" + "=" * 60)
    print("  RESUMEN: REGLAS DE ORO PARA OPTIMIZAR NUMPY")
    print("=" * 60)
    print("""
  1. VECTORIZAR: Reemplazar loops Python por operaciones NumPy
  2. DTYPES:     Usar el dtype más pequeño que mantenga precisión
  3. IN-PLACE:   Preferir += y out= para evitar asignaciones
  4. MEMORIA:    Entender vistas vs copias, orden C vs F
  5. PREASIGNAR: Nunca np.append en loop; usar np.empty o listas
  6. EINSUM:     Para operaciones tensoriales complejas
  7. BLAS:       Confiar en @ para álgebra lineal (ya está optimizado)

  La regla más importante: medir antes de optimizar.
  Usa bench() o %%timeit para verificar que tu optimización funciona.
""")


if __name__ == "__main__":
    main()
