import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt

file_path = 'ratings.csv'
df = pd.read_csv(file_path)

ratings_matrix = df.pivot(index='userId', columns='movieId', values='rating')

ratings_matrix = ratings_matrix.dropna(thresh=20, axis=0)
ratings_matrix = ratings_matrix.dropna(thresh=60, axis=1)

ratings_matrix_filled = ratings_matrix.fillna(ratings_matrix.mean().mean())

R = ratings_matrix_filled.values
user_ratings_mean = np.mean(R, axis=1)
R_demeaned = R - user_ratings_mean.reshape(-1, 1)

U, sigma, Vt = svds(R_demeaned, k=3)

sigma = np.diag(sigma)

rating_predict_everyone = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
preds_df = pd.DataFrame(rating_predict_everyone, columns=ratings_matrix.columns, index=ratings_matrix.index)
print(preds_df)

U_plot = U[:50]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(U_plot[:, 0], U_plot[:, 1], U_plot[:, 2])

plt.title('Users')
plt.show()

V_plot = Vt.T[:50]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(V_plot[:, 0], V_plot[:, 1], V_plot[:, 2])

plt.title('Movies')
plt.show()

predicted_ratings_only = preds_df.copy()

for row in ratings_matrix.index:
    for column in ratings_matrix.columns:
        if not np.isnan(ratings_matrix.loc[row, column]):
            predicted_ratings_only.loc[row, column] = np.nan


def recommend_movies(user_id, number=10):
    row_numb = user_id
    sorted = preds_df.iloc[row_numb].sort_values(ascending=False)
    recommendations = sorted.head(number)
    movies_df = pd.read_csv('movies.csv')
    movie_reccomendations = movies_df[movies_df['movieId'].isin(recommendations.index)]
    return movie_reccomendations[['movieId', 'title', 'genres']]


user_id = 1
print(f"ъ{user_id}\n{recommend_movies(user_id)}")
