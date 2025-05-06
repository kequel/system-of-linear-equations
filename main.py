import numpy as np
import matplotlib.pyplot as plt
import time

#Generacja macierzy
def generate_matrix_A(N, a1, a2, a3):
    A = np.zeros((N, N))
    for i in range(N):
        A[i, i] = a1
        if i > 0:
            A[i, i-1] = a2
        if i < N-1:
            A[i, i+1] = a2
        if i > 1:
            A[i, i-2] = a3
        if i < N-2:
            A[i, i+2] = a3
    return A

def generate_vector_b(N, f):
    return np.array([np.sin(n * (f + 1)) for n in range(1, N+1)])

#Metody iteracyjne
def jacobi(A, b, tol=1e-9, max_iter=10000):
    N = len(b)
    x = np.zeros(N)
    residuals = []
    for k in range(max_iter):
        x_new = np.zeros(N)
        for i in range(N): # wzor (1)
            sigma = np.dot(A[i, :], x) - A[i, i] * x[i]
            x_new[i] = (b[i] - sigma) / A[i, i]
        residual = np.linalg.norm(A @ x_new - b)
        residuals.append(residual)
        if residual < tol:
            return x_new, k+1, residuals
        x = x_new.copy()
    return x, max_iter, residuals

def gauss_seidel(A, b, tol=1e-9, max_iter=10000):
    N = len(b)
    x = np.zeros(N)
    residuals = []
    for k in range(max_iter):
        x_old = x.copy()
        for i in range(N): # wzor (2)
            sigma = np.dot(A[i, :], x) - A[i, i] * x[i]
            x[i] = (b[i] - sigma) / A[i, i]
        residual = np.linalg.norm(A @ x - b)
        residuals.append(residual)
        if residual < tol:
            return x, k+1, residuals
    return x, max_iter, residuals

#Metoda LU
def lu_decomposition(A):
    N = A.shape[0]
    L = np.eye(N)
    U = A.copy()
    for k in range(N-1):
        for i in range(k+1, N):
            L[i, k] = U[i, k] / U[k, k] # wzor (3)
            U[i, k:] -= L[i, k] * U[k, k:] # wzor (4)
    return L, U

def solve_lu(L, U, b):
    y = np.zeros(len(b))
    for i in range(len(b)):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])
    x = np.zeros(len(b))
    for i in range(len(b)-1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
    return x

def lu_solve(A, b):
    L, U = lu_decomposition(A)
    return solve_lu(L, U, b)

def calculate_residual(A, x, b):
    return np.linalg.norm(A @ x - b)

#Czas wykonania
def performance_test(methods, N_values, a1, a2, a3, f):
    times = {method: [] for method in methods}
    for N in N_values:
        A = generate_matrix_A(N, a1, a2, a3)
        b = generate_vector_b(N, f)
        for method in methods:
            start = time.time()
            if method == 'Jacobi':
                x, _, _ = jacobi(A, b, tol=1e-9)
            elif method == 'Gauss-Seidel':
                x, _, _ = gauss_seidel(A, b, tol=1e-9)
            elif method == 'LU':
                x = lu_solve(A, b)
            end = time.time()
            times[method].append(end - start)
    return times

N = 1293
f = 8
a2 = -1
a3 = -1

a1_1 = 6
A_1 = generate_matrix_A(N, a1_1, a2, a3)
b_1 = generate_vector_b(N, f)

x_jacobi, iter_jacobi, res_jacobi = jacobi(A_1, b_1, tol=1e-9)
x_gs, iter_gs, res_gs = gauss_seidel(A_1, b_1, tol=1e-9)

plt.figure()
plt.semilogy(res_jacobi, label='Jacobi')
plt.semilogy(res_gs, label='Gauss-Seidel')
plt.axhline(y=1e-9, color='red', linestyle='--', label=r'$\varepsilon = 10^{-9}$')
plt.xlabel('Iteracja')
plt.ylabel('Norma residuum (log)')
plt.legend()
plt.title('Zmiana normy residuum (a1=6)')
plt.savefig('wykresy/zmiana_normy_6.png')
plt.close()

#Nowe parametry a1=3
a1_2 = 3
A_2 = generate_matrix_A(N, a1_2, a2, a3)
b_2 = generate_vector_b(N, f)

#Test zbieżności
x_jacobi_c, iter_jacobi_c, res_jacobi_c = jacobi(A_2, b_2, tol=1e-9, max_iter=1000)
x_gs_c, iter_gs_c, res_gs_c = gauss_seidel(A_2, b_2, tol=1e-9, max_iter=1000)

plt.figure()
plt.semilogy(res_jacobi_c, label='Jacobi')
plt.semilogy(res_gs_c, label='Gauss-Seidel')
plt.xlabel('Iteracja')
plt.ylabel('Norma residuum (log)')
plt.legend()
plt.title('Zmiana normy residuum (a1=3)')
plt.savefig('wykresy/zmiana_normy_3.png')
plt.close()

x_lu = lu_solve(A_2, b_2)
residual_lu = calculate_residual(A_2, x_lu, b_2)
print(f"Norma residuum LU = {residual_lu}")

N_values = [100, 500, 1000, 2000, 3000]
methods = ['Jacobi', 'Gauss-Seidel', 'LU']
times = performance_test(methods, N_values, a1_1, a2, a3, f)

plt.figure()
for method in methods:
    plt.plot(N_values, times[method], marker='o', label=method)
plt.xlabel('Rozmiar N')
plt.ylabel('Czas (s)')
plt.legend()
plt.title('Czas rozwiązania (skala liniowa)')
plt.savefig('wykresy/czas_lin.png')
plt.close()

plt.figure()
for method in methods:
    plt.semilogy(N_values, times[method], marker='o', label=method)
plt.xlabel('Rozmiar N')
plt.ylabel('Czas (s) - log')
plt.legend()
plt.title('Czas rozwiązania (skala logarytmiczna)')
plt.savefig('wykresy/czas_log.png')
plt.close()