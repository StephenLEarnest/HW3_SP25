import math


def is_symmetric(A):
    """Check if matrix A is symmetric (A = A^T)."""
    rows = len(A)
    for i in range(rows):
        for j in range(i, rows):
            if A[i][j] != A[j][i]:
                return False
    return True


def is_positive_definite(A):
    """Check if matrix A is positive definite."""
    n = len(A)
    # Try Cholesky decomposition to check positive definiteness
    try:
        L = cholesky_decomposition(A)
        return True
    except ValueError:
        return False


def cholesky_decomposition(A):
    """Perform Cholesky decomposition of A into LL^T."""
    n = len(A)
    L = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i + 1):
            sum_value = A[i][j]
            for k in range(j):
                sum_value -= L[i][k] * L[j][k]
            if i == j:
                if sum_value <= 0:
                    raise ValueError("Matrix is not positive definite")
                L[i][j] = math.sqrt(sum_value)
            else:
                L[i][j] = sum_value / L[j][j]

    return L


def forward_substitution(L, b):
    """Solve the system L * y = b using forward substitution."""
    n = len(b)
    y = [0] * n
    for i in range(n):
        sum_value = b[i]
        for j in range(i):
            sum_value -= L[i][j] * y[j]
        y[i] = sum_value / L[i][i]
    return y


def backward_substitution(L, y):
    """Solve the system L^T * x = y using backward substitution."""
    n = len(y)
    x = [0] * n
    for i in range(n - 1, -1, -1):
        sum_value = y[i]
        for j in range(i + 1, n):
            sum_value -= L[j][i] * x[j]
        x[i] = sum_value / L[i][i]
    return x


def lu_decomposition(A):
    """Perform LU decomposition (A = LU) using Doolittle's method."""
    n = len(A)
    L = [[0] * n for _ in range(n)]
    U = [[0] * n for _ in range(n)]

    for i in range(n):
        # Upper triangular matrix U
        for j in range(i, n):
            U[i][j] = A[i][j]
            for k in range(i):
                U[i][j] -= L[i][k] * U[k][j]

        # Lower triangular matrix L
        for j in range(i, n):
            if i == j:
                L[i][i] = 1
            else:
                L[j][i] = A[j][i]
                for k in range(i):
                    L[j][i] -= L[j][k] * U[k][i]
                L[j][i] /= U[i][i]

    return L, U


def solve_system(A, b):
    """Solve the system Ax = b using Cholesky or LU decomposition."""
    if is_symmetric(A) and is_positive_definite(A):
        print("Using Cholesky Decomposition")
        L = cholesky_decomposition(A)
        y = forward_substitution(L, b)
        x = backward_substitution(L, y)
    else:
        print("Using LU Decomposition (Doolittle's Method)")
        L, U = lu_decomposition(A)
        y = forward_substitution(L, b)
        x = backward_substitution(U, y)

    return x


def print_solution(x):
    """Nicely print the solution vector x."""
    print("\nSolution vector x:")
    for i, val in enumerate(x):
        print(f"x[{i + 1}] = {val:.6f}")


def main():
    # Define matrices and right-hand side vectors for two problems
    A1 = [[4, 1], [1, 3]]
    b1 = [1, 2]

    A2 = [[2, 3, 1], [3, 4, 2], [1, 2, 3]]
    b2 = [9, 9, 6]

    print("Problem 1:")
    x1 = solve_system(A1, b1)
    print_solution(x1)

    print("\nProblem 2:")
    x2 = solve_system(A2, b2)
    print_solution(x2)


if __name__ == "__main__":
    main()
