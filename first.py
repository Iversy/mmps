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

    def norm(self, vector: list[float], p: int) -> float:
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


data = [[2.58, 2.93, 3.13, -6.66], [1.32, 1.55, 1.58, -3.58], [2.09, 2.25, 2.34, -5.01]]
# matrix = [
# [2.18, 2.44, 2.49, -4.34],
# [2.17, 2.31, 2.49, -3.91],
# [3.15, 3.22, 3.17, -5.27],
# ]
b = 0.5
data_b = deepcopy(data)
for row in data_b:
    row[-1] -= b


mat = Matrix(data)

print("Матрица прямого хода метода Гаусса:")
print(*mat.forwards, sep="\n")
print("Вектор прямого метода:", *mat.forward_vector)
print("Матрица обратного хода метода Гаусса:")
print(*mat.backwards, sep="\n")
print("Вектор решения:", *mat.vector)


print("Обратная матрица:")
inverse_matrix = mat.inverse()
print(*inverse_matrix, sep="\n")
print("А*А^-1:")
print(*(mat * inverse_matrix), sep="\n")
print("Вектор невязки:", mat.residual_vector())
print("Норма ||r||1:", mat.norm(mat.residual_vector(), 1))
print("Норма ||r||∞:", mat.norm(mat.residual_vector(), float("inf")))
print("Норма ||r||2:", mat.norm(mat.residual_vector(), 2))
print("Определитель:", mat.determinant())
print("Число обусловленности:", mat.condition_number())

mat_b = Matrix(data_b)

print("Вектор с погрешностью", *mat_b.vector)
print(
    "Вектор относительных погрешностей",
    *(abs(mat.vector[i] - el) for i, el in enumerate(mat_b.vector))
)
