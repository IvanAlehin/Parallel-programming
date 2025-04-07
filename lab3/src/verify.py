import numpy as np
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def verify(size):
    try:
        
        matrix_a = np.loadtxt(os.path.join(ROOT_DIR, "data", f"matrix_A_{size}.txt"))
        matrix_b = np.loadtxt(os.path.join(ROOT_DIR, "data", f"matrix_B_{size}.txt"))
        result = np.loadtxt(os.path.join(ROOT_DIR, "results", f"result_{size}.txt"))

        result_numpy = np.dot(matrix_a, matrix_b)

        if np.allclose(result, result_numpy, atol=1e-6):
            print(f"Результаты для размера {size}x{size} совпадают!")
        else:
            print(f"Результаты для размера {size}x{size} не совпадают!")
    except Exception as e:
        print(f"Ошибка при проверке размера {size}x{size}: {e}")

if __name__ == "__main__":
    sizes = [50, 100, 150, 250, 500, 1000]

    for size in sizes:
        verify(size)