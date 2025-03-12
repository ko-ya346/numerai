import matplotlib.pyplot as plt


def plot_feature_dist(df, feature):
    _, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].hist(df[feature], bins=5)
    df.groupby("era_int")[feature].mean().plot(kind="line", marker="o", ax=ax[1])

    plt.title(feature)
    plt.show()
