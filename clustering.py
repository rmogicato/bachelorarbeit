from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd

df_arcface = pd.read_csv("arc_face/arcface_validation.csv")
df_arcface = df_arcface.rename(columns={"Unnamed: 0":"Image"})

df_identities = pd.read_csv("identity_CelebA.txt", sep='\s+', names=["Image", "Id"])
print(df_identities)
images = df_arcface["Image"]


def suffixizer(x):
    x = x[:-3]
    return x+"jpg"


images = images.apply(suffixizer)

res = df_identities.loc[df_identities["Image"].isin(images)]
ids = res.Id.tolist()
unique_ids = np.unique(ids)
print(len(unique_ids))


