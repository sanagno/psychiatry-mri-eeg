import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_categories(embedding, classes, most_common_disorders, small_size=20, large_size=40):
    fig, ax = plt.subplots(nrows=1, ncols=len(most_common_disorders) + 1, figsize=(25, 7))

    def have_only_disorder(cls, dis):
        res = set(list(range(cls.shape[0])))

        for k in range(cls.shape[1]):
            if k == dis:
                res = res & set(np.where(cls[:, k] == 1)[0])
            else:
                res = res & set(np.where(cls[:, k] == 0)[0])

        return np.array(list(res))

    blue = sns.color_palette("Blues")[-2]
    red = sns.color_palette("Reds")[-2]
    green = sns.color_palette("Greens")[-2]
    for i in range(len(most_common_disorders)):
        ax[i].scatter(embedding[:, 0], embedding[:, 1],
                      c=[green if x == 1 else blue for x in classes[:, i]],
                      s=[large_size if x == 1 else small_size for x in classes[:, i]])
        only_disorder = have_only_disorder(classes, i)
        ax[i].scatter(embedding[only_disorder, 0], embedding[only_disorder, 1],
                      c=[red for _ in range(len(only_disorder))], s=large_size)
        # ax[i].set_gca().set_aspect('equal', 'datalim')
        ax[i].set_title('Existence of disorder: ' + most_common_disorders[i], fontsize=10)

    indx = np.where(np.sum(classes, axis=1) > 0)[0]
    ax[-1].scatter(embedding[indx, 0], embedding[indx, 1], c=[blue for _ in range(len(indx))], label='no disorder',
                   s=small_size)
    indx = np.where(np.sum(classes, axis=1) == 0)[0]
    ax[-1].scatter(embedding[indx, 0], embedding[indx, 1], c=[green for _ in range(len(indx))], label='disorder',
                   s=large_size)
    # dummy to get no indices
    indx = np.where(np.sum(classes, axis=1) == len(most_common_disorders) + 1)[0]
    ax[-1].scatter(embedding[indx, 0], embedding[indx, 1], c=[red for _ in range(len(indx))],
                   label='only this disorder', s=large_size)
    ax[-1].set_title('No disorder diagnosed', fontsize=10)

    ax[-1].legend()

    plt.tight_layout()
    plt.show()
