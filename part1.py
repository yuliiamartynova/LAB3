import numpy as np

def SVD(A):
    m, n = A.shape
    ATA = np.dot(A.T, A)
    eigenvalues, V = np.linalg.eigh(ATA)

    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    V = V[:, sorted_indices]

    sigma = np.sqrt(eigenvalues)

    S = np.zeros((m, n), dtype=float)
    min_dimension = min(m, n)
    S[:min_dimension, :min_dimension] = np.diag(sigma)

    U = np.dot(A, V) / sigma

    return U, S, V.T


A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

U, S, VT = SVD(A)
print(f'U:\n'
     f'{U}')

print(f'S:\n'
     f'{S}')

print(f'VT:\n'
     f'{VT}')

check = np.dot(U, np.dot(S, VT))

print(np.allclose(A, np.dot(U, np.dot(S, VT))))