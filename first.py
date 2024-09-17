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
        augmented_matrix = [row + [0] * n for row in self.data]
        for i in range(n):
            augmented_matrix[i][i + n] = 1

        for i in range(n):
            for k in range(i + 1, n):
                multiplier = augmented_matrix[k][i] / augmented_matrix[i][i]
                for j in range(i, 2 * n):
                    augmented_matrix[k][j] -= multiplier * augmented_matrix[i][j]

        for i in range(n - 1, -1, -1):
            for j in range(i + 1, n):
                augmented_matrix[i][n + j] -= (
                    augmented_matrix[i][j]
                    * augmented_matrix[j][n + j]
                    / augmented_matrix[j][j]
                )
            for j in range(n, 2 * n):
                augmented_matrix[i][j] /= augmented_matrix[i][i]

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
print("Вектор невязки:", mat.residual_vector())
print("Норма ||r||1:", mat.norm(mat.residual_vector(), 1))
print("Норма ||r||∞:", mat.norm(mat.residual_vector(), float("inf")))
print("Норма ||r||2:", mat.norm(mat.residual_vector(), 2))
print("Определитель:", mat.determinant())
print("Число обусловленности:", mat.condition_number())