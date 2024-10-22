from copy import deepcopy
import math


class Matrix:
    def __init__(self, matrix: list[list[float | int]]) -> None:
        self.data = matrix
        if not self.data or len(self.data) != len(self.data[0]) - 1:
            raise AssertionError
        self.n_row = len(matrix)
        self.solution_vector = None
        self.forward_data = self._forward_gauss()
        self.ax_b = self.convert()
        self._backward_gauss()

    @staticmethod
    def _multiply_number(matrix, value) -> list[list[float | int]]:
        return [
            [matrix[i][j] * value for j in range(len(matrix[0]))]
            for i in range(len(matrix))
        ]

    # Умножаем одинаковые матрицы...
    @staticmethod
    def _multiply_matrix(first_matrix, second_matrix) -> list[list[float | int]]:
        size_first = len(first_matrix)
        size_second = len(second_matrix)

        if size_first != size_second:
            raise ValueError

        result = [[0 for _ in range(size_second)] for _ in range(size_first)]
        for i in range(size_first):
            for j in range(size_second):
                for k in range(size_second):
                    result[i][j] += first_matrix[i][k] * second_matrix[k][j]
        return result

    def __mul__(self, value) -> list[list[float | int]]:
        if isinstance(value, int) or isinstance(value, float):
            return self._multiply_number(self.data, value)
        if isinstance(value, list):
            return self._multiply_matrix(self.data, value)

    @property
    def forward_vector(self) -> list[float | int]:
        if not self.forward_data:
            raise AssertionError
        return [self.forward_data[i][self.n_row] for i in range(self.n_row)]

    @property
    def backwards(self) -> list[list[float | int]]:
        if not self.solution_vector:
            raise AssertionError
        temp_matrix = [[0] * (self.n_row + 1) for _ in range(self.n_row)]
        for i in range(self.n_row):
            temp_matrix[i][i] = 1.0
            temp_matrix[i][self.n_row] = self.solution_vector[i]
        return temp_matrix

    @property
    def forwards(self) -> list[list[float | int]]:
        if not self.forward_data:
            raise AssertionError
        return self.forward_data

    @property
    def vector(self) -> list[float | int]:
        if not self.solution_vector:
            raise AssertionError
        return self.solution_vector

    def _forward_gauss(self) -> list[list[float | int]]:
        temp_matrix = deepcopy(self.data)
        for i in range(self.n_row):
            for k in range(i + 1, self.n_row):
                multiplier = temp_matrix[k][i] / temp_matrix[i][i]
                for j in range(self.n_row, i - 1, -1):
                    temp_matrix[k][j] -= (
                        multiplier * temp_matrix[i][j]
                    )  # round(multiplier * temp_matrix[i][j], 15)
            for j in range(self.n_row, i - 1, -1):
                temp_matrix[i][j] = (
                    temp_matrix[i][j] / temp_matrix[i][i]
                )  # round(temp_matrix[i][j] / temp_matrix[i][i], 15)

        return temp_matrix

    def matrix_multiply(self, A, B):
        rows_A = len(A)
        cols_A = len(A[0]) if rows_A > 0 else 0
        rows_B = len(B)
        cols_B = len(B[0]) if rows_B > 0 else 0

        if cols_A != rows_B:
            raise ValueError()

        result = [[0 for _ in range(cols_B)] for _ in range(rows_A)]

        for i in range(rows_A):
            for j in range(cols_B):
                for k in range(cols_A):
                    result[i][j] += A[i][k] * B[k][j]

        return result

    def _backward_gauss(self) -> list[float]:
        temp_matrix = deepcopy(self.forward_data)
        n = self.n_row
        self.solution_vector = [0] * n

        for i in range(n - 1, -1, -1):
            self.solution_vector[i] = temp_matrix[i][n]
            for j in range(i + 1, n):
                self.solution_vector[i] -= temp_matrix[i][j] * self.solution_vector[j]
            self.solution_vector[i] /= temp_matrix[i][i]

        return self.solution_vector

    def residual_vector(self) -> list[float]:
        Ax = [
            sum(self.data[i][j] * self.solution_vector[j] for j in range(self.n_row))
            for i in range(self.n_row)
        ]
        b = [self.data[i][-1] for i in range(self.n_row)]
        return [b[i] - Ax[i] for i in range(self.n_row)]

    @staticmethod
    def norm(vector: list[float], p: int) -> float:
        if p == 1:
            return sum(abs(x) for x in vector)
        elif p == 2:
            return math.sqrt(sum(x**2 for x in vector))
        elif p == float("inf"):
            return max(abs(x) for x in vector)
        else:
            raise ValueError("Неподдерживаемая норма")

    def determinant(self) -> float:
        temp_matrix = deepcopy(self.data)
        det = 1.0
        for i in range(self.n_row):
            det *= temp_matrix[i][i]
            for k in range(i + 1, self.n_row):
                multiplier = temp_matrix[k][i] / temp_matrix[i][i]
                for j in range(i, self.n_row + 1):
                    temp_matrix[k][j] -= multiplier * temp_matrix[i][j]
        return det

    def inverse(self) -> list[list[float]]:
        n = self.n_row
        augmented_matrix = [
            row[:-1] + [1 if i == j else 0 for j in range(n)]
            for i, row in enumerate(self.data)
        ]

        for i in range(n):
            pivot = augmented_matrix[i][i]
            if pivot == 0:
                raise ValueError("Матрица вырождена и не имеет обратной.")
            for j in range(2 * n):
                augmented_matrix[i][j] /= pivot

            for k in range(n):
                if k != i:
                    multiplier = augmented_matrix[k][i]
                    for j in range(2 * n):
                        augmented_matrix[k][j] -= multiplier * augmented_matrix[i][j]

        return [row[n:] for row in augmented_matrix]

    def condition_number(self) -> float:
        A_inv = self.inverse()
        norm_A = self.norm(
            [
                sum(self.data[i][j] for j in range(self.n_row))
                for i in range(self.n_row)
            ],
            2,
        )
        norm_A_inv = self.norm(
            [sum(A_inv[i][j] for j in range(self.n_row)) for i in range(self.n_row)], 2
        )
        return norm_A * norm_A_inv

    def check_convergence(self) -> bool:
        return self.norm(
            [sum(self.ax_b[i]) - self.ax_b[i][-1] for i in range(self.n_row)],
            float("inf"),
        )

    def check_dominance(self) -> bool:
        tmp = [
            [abs(self.data[i][j]) for j in range(self.n_row)] for i in range(self.n_row)
        ]
        for i in range(self.n_row):
            if sum(tmp[i]) - tmp[i][i] >= tmp[i][i]:
                return False
        return True

    def seidel_method(self, eps=1e-8):
        n = self.n_row
        matrix = deepcopy(self.ax_b)
        x = [matrix[i][-1] for i in range(n)]
        k = 0
        while True:
            x_old = x.copy()
            for i in range(n):
                x[i] = (
                    sum((matrix[i][j] * x[j] for j in range(i)))
                    + sum((matrix[i][j] * x_old[j] for j in range(i, n)))
                    + matrix[i][-1]
                )
            if self.norm([x[j] - x_old[j] for j in range(n)], float("inf")) < eps:
                break
            k += 1
        return x, k

    def simple_iteration_method(self, eps=1e-8):
        n = self.n_row
        matrix = deepcopy(self.ax_b)
        x = [matrix[i][-1] for i in range(n)]
        k = 0
        while True:
            x_old = x.copy()
            for i in range(n):
                x[i] = sum(matrix[i][j] * x_old[j] for j in range(n)) + matrix[i][-1]
            if self.norm([x[j] - x_old[j] for j in range(n)], float("inf")) < eps:
                break
            k += 1
        return x, k

    def compute_errors(self, solution: list[float], norm) -> tuple[float, float]:
        absolute_error = [
            self.solution_vector[i] - solution[i] for i in range(self.n_row)
        ]
        relative_error = [
            (
                abs(absolute_error[i]) / abs(solution[i])
                if solution[i] != 0
                else float("inf")
            )
            for i in range(self.n_row)
        ]
        return self.norm(absolute_error, norm), self.norm(relative_error, norm)

    def convert(self):
        matrix = deepcopy(self.data)
        for i in range(self.n_row):
            matrix[i][-1] = self.data[i][-1] / self.data[i][i]
            for j in range(self.n_row):
                matrix[i][j] = 0 if i == j else -(self.data[i][j] / self.data[i][i])
        return matrix


data = [
    [1.7000 / 100, 0.0003, 0.0004, 0.0005, 0.6810],
    [0.0000, 0.8000 / 100, 0.0001, 0.0002, 0.4803],
    [-0.0003, -0.0002, -0.1000 / 100, 0.0000, -0.0802],
    [-0.0005, -0.0004, -0.0003, -1.0000 / 100, -1.0007],
]


def print_all_errors(x, _matrix: Matrix):
    norms = (1, 2, float("inf"))
    for i in norms:
        absolute, relative = _matrix.compute_errors(x, i)
        print(f"Абсолютная = {absolute}, Относительная = {relative}, по норме {i}")


# data = [[2, 2, 10, 14], [10, 1, 1, 12], [2, 10, 1, 13]]

mat = Matrix(data)
for i in mat.data:
    print(*i, sep=" ")
print("Гаусс")
print(mat.solution_vector)
print("Сходимость (сходится или нет)")
print(mat.check_convergence())
for i in (1e-8, 1e-12, 1e-15):
    print("Epsilon = ", i)
    print("Метод простых итераций")
    x, k = mat.simple_iteration_method(i)
    print(f"Сошлось за {k+1} итераций.")
    print(*x)
    print("Сверка Простых итераций с Гауссом")
    print_all_errors(x, mat)
    print("Метод Зейделя")
    x, k = mat.seidel_method(i)
    print(f"Сошлось за {k+1} итераций.")
    print(*x)
    print("Сверка Зейделя с Гауссом")
    print_all_errors(x, mat)
    print("-" * 50)
