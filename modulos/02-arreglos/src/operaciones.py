"""Operaciones aritméticas, rendimiento y funciones universales (ufuncs)."""

import time

import numpy as np


def demo_arithmetic():
    """Operaciones aritméticas elemento a elemento."""
    print("\n--- Operaciones aritméticas (element-wise) ---\n")

    a = np.array([10, 20, 30, 40])
    b = np.array([1, 2, 3, 4])
    print(f"a = {a}")
    print(f"b = {b}\n")

    print(f"a + b  (suma)              : {a + b}")
    print(f"a - b  (resta)             : {a - b}")
    print(f"a * b  (multiplicación)    : {a * b}")
    print(f"a / b  (división)          : {a / b}")
    print(f"a ** b (potencia)          : {a ** b}")
    print(f"a % b  (módulo)            : {a % b}")
    print(f"a // b (división entera)   : {a // b}")

    print("\nTodas las operaciones se aplican elemento a elemento.")
    print("No se necesitan bucles — NumPy las ejecuta en C internamente.")


def demo_performance():
    """Comparar rendimiento: bucle Python vs np.sum."""
    print("\n--- Rendimiento: Python loop vs NumPy ---\n")

    n = 1_000_000
    rng = np.random.default_rng(42)
    arr = rng.random(n)
    arr_list = arr.tolist()

    # Python loop
    t0 = time.perf_counter()
    total_py = 0.0
    for x in arr_list:
        total_py += x
    t1 = time.perf_counter()
    tiempo_py = t1 - t0

    # NumPy
    t0 = time.perf_counter()
    total_np = np.sum(arr)
    t1 = time.perf_counter()
    tiempo_np = t1 - t0

    print(f"Elementos: {n:,}")
    print(f"Suma Python loop: {total_py:.6f}  — Tiempo: {tiempo_py:.6f} s")
    print(f"Suma np.sum:      {total_np:.6f}  — Tiempo: {tiempo_np:.6f} s")

    speedup = tiempo_py / tiempo_np if tiempo_np > 0 else float("inf")
    print(f"\nNumPy fue ~{speedup:.0f}× más rápido que el bucle de Python.")
    print("Conclusión: evita bucles; usa operaciones vectorizadas de NumPy.")


def demo_ufuncs():
    """Funciones universales (ufuncs) de NumPy."""
    print("\n--- Funciones universales (ufuncs) ---\n")

    arr = np.array([1, 4, 9, 16, 25], dtype=float)
    print(f"arr = {arr}\n")

    print(f"np.sqrt(arr)  : {np.sqrt(arr)}")
    print(f"np.log(arr)   : {np.log(arr)}")
    print(f"np.exp([1,2]) : {np.exp(np.array([1.0, 2.0]))}")

    angulos = np.array([0, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2])
    print(f"\nÁngulos (rad) : {angulos}")
    print(f"np.sin        : {np.round(np.sin(angulos), 4)}")
    print(f"np.cos        : {np.round(np.cos(angulos), 4)}")

    a = np.array([3, 7, 1, 9])
    b = np.array([5, 2, 8, 4])
    print(f"\na = {a}")
    print(f"b = {b}")
    print(f"np.maximum(a, b) : {np.maximum(a, b)}")
    print(f"np.minimum(a, b) : {np.minimum(a, b)}")

    print("\nLas ufuncs operan elemento a elemento y son muy rápidas.")
    print("Son la base del cálculo vectorizado en NumPy.")
