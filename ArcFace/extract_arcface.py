import bob.io.base
import bob.io.image
import mxnet as mx
import numpy
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils import check_array
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
import pkg_resources
import os
import sys
from bob.extension import rc
from skimage import io
from matplotlib import pyplot as plt

SOURCE_DIR_PATH = "../data/img_arcface/img_validation/"

internal_path = pkg_resources.resource_filename(
    __name__, os.path.join("data", "arcface_insightface"),
)

checkpoint_path = (
    internal_path
    if rc["bob.bio.face.models.ArcFaceInsightFace"] is None
    else rc["bob.bio.face.models.ArcFaceInsightFace"]
)

sym, arg_params, aux_params = mx.model.load_checkpoint(
    os.path.join(checkpoint_path, "model"), 0
)

all_layers = sym.get_internals()
sym = all_layers["fc1_output"]

# LOADING CHECKPOINT
ctx = mx.cpu()
model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
data_shape = (1, 3, 112, 112)
model.bind(data_shapes=[("data", data_shape)])
model.set_params(arg_params, aux_params)

# warmup
data = mx.nd.zeros(shape=data_shape)
db = mx.io.DataBatch(data=(data,))
model.forward(db, is_train=False)
embedding = model.get_outputs()[0].asnumpy()
model = model

files = os.listdir(SOURCE_DIR_PATH)

arcs = {}

for i, name in enumerate(files):
    p = str(round(i/len(files) * 100, 0))
    sys.stdout.write("\rProgress: " + p + "% - current file: " + name)
    sys.stdout.flush()
    image = bob.io.base.load(SOURCE_DIR_PATH + name)
    """
    # convert to the required data type
    X = image.astype(numpy.float32) / 255.
    X = check_array(X, allow_nd=True)
    """
    # adding 4th dimension?
    X = np.array([image])

    def _transform(X):
        X = mx.nd.array(X)
        db = mx.io.DataBatch(data=(X,))
        model.forward(db, is_train=False)
        return model.get_outputs()[0].asnumpy()

    vector = _transform(X)
    arcs[name] = vector[0]

df = pd.DataFrame.from_dict(arcs, orient="index")
df.to_csv("arcface_validation_v2.csv")