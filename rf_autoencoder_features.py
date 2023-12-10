import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.svm import SVC

train_X = np.load("train_features.npy")
valid_X = np.load("valid_features.npy")

train_metadata = pd.read_csv("pnw_splits/train/metadata.csv")
valid_metadata = pd.read_csv("pnw_splits/valid/metadata.csv")

train_y = (train_metadata.source_type == "surface event").to_numpy(dtype=int)
valid_y = (valid_metadata.source_type == "surface event").to_numpy(dtype=int)

clf = RandomForestClassifier(n_estimators=1000, criterion="entropy")
# clf.fit(train_X, train_y)
# clf.score(valid_X, valid_y)

# Perform hyper-parameter optimization.

rf_param_grid = {
    # "bootstrap": [True, False],
    # "ccp_alpha": [0.0, 0.01, 0.05],
    "criterion": ["gini", "entropy", "log_loss"],
    # "max_depth": None,
    # "max_features": ["sqrt", "log2", 0.1, 0.5, 1.0],
    # "max_leaf_nodes": None,
    # "max_samples": [0.1, 0.5, 1.0],
    # "min_impurity_decrease": 0.0,
    # "min_samples_leaf": 1,
    # "min_samples_split": 2,
    # "min_weight_fraction_leaf": 0.0,
    "n_estimators": [100, 500, 1000, 1500],
}

# (* 2 3 3 5 3 4)


def hyper_parameter_opt():
    gs = GridSearchCV(RandomForestClassifier(), rf_param_grid, n_jobs=-1, cv=4)
    gs.fit(train_X, train_y)
    return gs


# clf2 = GaussianNB() # Doesn't seem to work very well.

# TODD: scale data?
# clf2 = make_pipeline(StandardScaler(), SVC(C=10, gamma=0.001))
# clf2 = make_pipeline(StandardScaler(), SVC(C=1000, gamma=0.0001))
# clf2 = SVC(C=10, gamma=0.001)
# clf2 = SVC(C=50, gamma=0.0005)
# clf2 = make_pipeline(StandardScaler(), SVC(C=50, gamma=0.0005))

svc_param_grid = {
    "svc__gamma": [0.0001, 0.0005, 0.001, 0.005, 0.01],
    "svc__C": [0.1, 0.5, 1, 5, 10, 50, 100],
}


def svc_hyper_opt():
    gs = GridSearchCV(make_pipeline(SVC()), svc_param_grid, n_jobs=-1, cv=4)
    gs.fit(train_X, train_y)
    return gs


def plot_embedded_features():
    fig, axs = plt.subplots(ncols=2)
    axs[0].set_title("earthquake")
    axs[1].set_title("surface event")
    V = StandardScaler().fit_transform(valid_X)
    axs[0].imshow(V[valid_y == 0][:800])
    axs[1].imshow(V[valid_y == 1][:800])
    plt.show()


# Copied from autoencoder.py.
def plot_t_sne(X, y):
    X_emb = TSNE().fit_transform(X)
    plt.scatter(X_emb[:, 0][y == 0], X_emb[:, 1][y == 0], alpha=0.3, label="earthquake")
    plt.scatter(
        X_emb[:, 0][y == 1], X_emb[:, 1][y == 1], alpha=0.3, label="surface event"
    )
    plt.legend()
    plt.show()


def plot_kmeans(X, y, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters)
    y_pred = kmeans.fit_predict(X)
    for i in range(n_clusters):
        y_in_cluster = y[y_pred == i]
        ys = np.sum(y_in_cluster)
        print(f"{len(y_in_cluster) - ys} earthquakes, {ys} surface events")
    # Use t-sne to scatter.
    X_emb = TSNE().fit_transform(X)
    plt.scatter(
        X_emb[:, 0][y == 0],
        X_emb[:, 1][y == 0],
        alpha=0.3,
        c=y_pred[y == 0],
        marker="s",
        label="earthquake",
    )
    plt.scatter(
        X_emb[:, 0][y == 1],
        X_emb[:, 1][y == 1],
        alpha=0.3,
        c=y_pred[y == 1],
        marker="o",
        label="surface event",
    )
    plt.legend()
    plt.show()


def plot_feature_correlations(X):
    C = np.corrcoef(X, rowvar=False)
    plt.imshow(C, vmin=-1, vmax=1)
    plt.show()
