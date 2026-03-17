"""Preasignación de arrays: evitar np.append y patrones O(N²)."""

import numpy as np

from src.benchmarks import bench


def demo_append_antipattern():
    """np.append en loop — antipatrón O(N²)."""

    print("\n--- Antipatrón: np.append en un Loop ---\n")

    n = 1000

    print(f"  Construir array de {n} elementos con np.append en cada iteración:\n")
    print("  Código:")
    print("    arr = np.array([])")
    print("    for i in range(N):")
    print("        arr = np.append(arr, i**2)\n")

    print("  ¿Por qué es lento?")
    print("  Cada np.append() crea un array NUEVO, copia todo, y agrega 1 elemento.")
    print(f"  Para N={n}: se copian 0+1+2+...+{n-1} = {n*(n-1)//2:,} elementos en total → O(N²)\n")

    def append_loop(n):
        arr = np.array([])
        for i in range(n):
            arr = np.append(arr, i**2)
        return arr

    t_append = bench(append_loop, n, n_runs=3, warmup=1)
    print(f"  Tiempo (N={n}): {t_append['mean']*1000:.2f} ms")

    resultado = append_loop(n)
    print(f"  Resultado: [{resultado[0]:.0f}, {resultado[1]:.0f}, {resultado[2]:.0f}, ..., {resultado[-1]:.0f}]")
    print(f"  Longitud: {len(resultado)}")


def demo_preallocation():
    """np.empty + llenar en loop — patrón eficiente O(N)."""

    print("\n--- Preasignación con np.empty ---\n")

    n = 1000

    print("  Código:")
    print("    arr = np.empty(N)")
    print("    for i in range(N):")
    print("        arr[i] = i**2\n")

    print("  Se asigna la memoria UNA vez al inicio.")
    print("  Cada iteración solo escribe en una posición existente.\n")

    def prealloc_loop(n):
        arr = np.empty(n)
        for i in range(n):
            arr[i] = i**2
        return arr

    t_prealloc = bench(prealloc_loop, n, n_runs=3, warmup=1)
    print(f"  Tiempo (N={n}): {t_prealloc['mean']*1000:.2f} ms")

    resultado = prealloc_loop(n)
    print(f"  Resultado: [{resultado[0]:.0f}, {resultado[1]:.0f}, {resultado[2]:.0f}, ..., {resultado[-1]:.0f}]")


def demo_list_accumulate():
    """Lista de Python + np.array al final — alternativa práctica."""

    print("\n--- Lista Python + np.array al Final ---\n")

    n = 1000

    print("  Código:")
    print("    valores = []")
    print("    for i in range(N):")
    print("        valores.append(i**2)")
    print("    arr = np.array(valores)\n")

    print("  list.append() es O(1) amortizado (usa tabla de sobreescritura).")
    print("  np.array() al final copia todo una vez.\n")

    def list_accumulate(n):
        valores = []
        for i in range(n):
            valores.append(i**2)
        return np.array(valores)

    t_list = bench(list_accumulate, n, n_runs=3, warmup=1)
    print(f"  Tiempo (N={n}): {t_list['mean']*1000:.2f} ms")

    # --- Comparación final ---
    print("\n--- Comparación de los 3 Métodos ---\n")

    def append_loop(n):
        arr = np.array([])
        for i in range(n):
            arr = np.append(arr, i**2)
        return arr

    def prealloc_loop(n):
        arr = np.empty(n)
        for i in range(n):
            arr[i] = i**2
        return arr

    t_append = bench(append_loop, n, n_runs=3, warmup=1)
    t_prealloc = bench(prealloc_loop, n, n_runs=3, warmup=1)
    t_list = bench(list_accumulate, n, n_runs=3, warmup=1)

    # También: la forma totalmente vectorizada
    def vectorizado(n):
        return np.arange(n, dtype=np.float64) ** 2

    t_vec = bench(vectorizado, n, n_runs=5, warmup=2)

    print(f"  {'Método':<25} {'Tiempo (ms)':<15} {'Speedup vs append'}")
    print(f"  {'─'*25} {'─'*15} {'─'*20}")
    base = t_append["mean"]
    print(f"  {'np.append (loop)':<25} {t_append['mean']*1000:<15.2f} {'1.0x (referencia)'}")
    print(f"  {'np.empty + fill':<25} {t_prealloc['mean']*1000:<15.2f} {base/t_prealloc['mean']:.0f}x")
    print(f"  {'list + np.array':<25} {t_list['mean']*1000:<15.2f} {base/t_list['mean']:.0f}x")
    print(f"  {'Vectorizado (arange²)':<25} {t_vec['mean']*1000:<15.4f} {base/t_vec['mean']:.0f}x")

    print("\n  Takeaway:")
    print("  1. NUNCA uses np.append en un loop — es O(N²)")
    print("  2. Preasigna con np.empty si conoces el tamaño")
    print("  3. Usa lista Python si el tamaño es dinámico")
    print("  4. Si puedes, vectoriza completamente (lo más rápido)")
