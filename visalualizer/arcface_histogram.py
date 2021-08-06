import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Palatino Linotype'

def add_jpg(x):
    x = x[:-3]
    return x + "jpg"


def add_png(x):
    x = x[:-3]
    return x + "png"

# this file generates a histogram when given a dataframe of clustered identities (df_arcface)
# and a dataframe of labeled identities
df_arcface = pd.read_csv("../ArcFace/arcface_validation_v2.csv")
df_arcface = df_arcface.rename(columns={"Unnamed: 0": "Image"})
df_arcface["Image"] = df_arcface["Image"].apply(add_jpg)

df_identities = pd.read_csv("../data/txt_files/identity_CelebA.txt", sep='\s+', names=["Image", "Id"])
images = df_arcface["Image"]

images = images.apply(add_jpg)
res = df_identities.loc[df_identities["Image"].isin(images)]
ids = np.unique(res.Id.tolist())


def compare_ids(id1, id2):
    img_same_id = df_identities.loc[df_identities["Id"] == id1].Image.values
    # taking first image as reference picture
    x = img_same_id[0]
    # all pictures of same id
    img_same_id = np.delete(img_same_id, 0)
    # getting vector and removing first element (name of the image) and converting it into array
    cosine_x = df_arcface.loc[df_arcface["Image"] == x].values[0]
    cosine_x = np.array(cosine_x[1:])
    scores_same_id = []

    for img in img_same_id:
        y = img
        cosine_y = df_arcface.loc[df_arcface["Image"] == y].values[0]
        cosine_y = np.array(cosine_y[1:])

        score = cosine_similarity([cosine_x], [cosine_y])
        scores_same_id.append(score[0][0])
        if score < 0.2:
            print(id1, x, img)

    identity_2 = id2
    img_diff_id = df_identities.loc[df_identities["Id"] == identity_2].Image.values
    scores_diff_id = []

    for img in img_diff_id:
        y = img
        cosine_y = df_arcface.loc[df_arcface["Image"] == y].values[0]
        cosine_y = np.array(cosine_y[1:])

        score = cosine_similarity([cosine_x], [cosine_y])
        scores_diff_id.append(score[0][0])

    return scores_same_id, scores_diff_id


scores_same_id = []
scores_diff_id = []

# change sample size
sample_size = 300
rand_ids = np.random.choice(ids, size=sample_size)
rand_ids = np.unique(rand_ids)
sample_size = len(rand_ids)

for s in range(int(sample_size/2)):
    # comparing first and last id, then second to second last etc.
    id1 = rand_ids[s]
    id2 = rand_ids[sample_size-s-1]
    same_id, diff_id = compare_ids(id1, id2)
    scores_diff_id += diff_id
    scores_same_id += same_id

fig = plt.figure(figsize=(8, 4))
plt.hist(scores_same_id, bins="auto", alpha=0.5, label="Scores of Same Identity")
plt.hist(scores_diff_id, bins="auto", alpha=0.5, label="Scores of Different Identity")
plt.xlabel("Cosine Similarity Score", size=14)
plt.ylabel("Count", size=14)
plt.title("Score Comparison between Same and Different Identities")
plt.legend(loc='upper center')
plt.show()
fig.savefig("../resources/histogram.pdf")