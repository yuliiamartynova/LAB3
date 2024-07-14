import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt

df = pd.read_csv('ratings.csv')

ratings_matrix = df.pivot(index='userId', columns='movieId', values='rating')

ratings_matrix = ratings_matrix.dropna(thresh=20, axis=0)
ratings_matrix = ratings_matrix.dropna(thresh=60, axis=1)

ratings_matrix_filled = ratings_matrix.fillna(ratings_matrix.mean().mean())

R = ratings_matrix_filled.values
user_ratings_mean = np.mean(R, axis=1)
R_demeaned = R - user_ratings_mean.reshape(-1, 1)

U, sigma, Vt = svds(R_demeaned, k=3)

U_cut = U[:50, :]
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(U_cut[:, 0], U_cut[:, 1], U_cut[:, 2])

plt.title('Users')
plt.show()


V_cut = Vt[:, :50].T

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(V_cut[:, 0], V_cut[:, 1], V_cut[:, 2])

plt.title('Movies')
plt.show()