import os
import sys

import numpy

import torch
import pandas as pd
import imp
import bob.io.base
import bob.io.image

# load network model
# either AFFACT_balanced.py and AFFACT_balanced.pth or AFFACT_E.py and AFFACT_E.pth
MainModel = imp.load_source('MainModel', "./AFFACT_E.py")
network = torch.load("./AFFACT_E.pth")

# setup network
network.eval()
# OPTIONAL: set to cuda environment if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
network.to(device)  # /home/raffael/miniconda3/envs/bob_env1

SOURCE_DIR_PATH = "../data/img_celeba/img_validation/"
files = os.listdir(SOURCE_DIR_PATH)
# print the result to console
attribute_names = [
  '5_o_Clock_Shadow',
  'Arched_Eyebrows',
  'Attractive',
  'Bags_Under_Eyes',
  'Bald',
  'Bangs',
  'Big_Lips',
  'Big_Nose',
  'Black_Hair',
  'Blond_Hair',
  'Blurry',
  'Brown_Hair',
  'Bushy_Eyebrows',
  'Chubby',
  'Double_Chin',
  'Eyeglasses',
  'Goatee',
  'Gray_Hair',
  'Heavy_Makeup',
  'High_Cheekbones',
  'Male',
  'Mouth_Slightly_Open',
  'Mustache',
  'Narrow_Eyes',
  'No_Beard',
  'Oval_Face',
  'Pale_Skin',
  'Pointy_Nose',
  'Receding_Hairline',
  'Rosy_Cheeks',
  'Sideburns',
  'Smiling',
  'Straight_Hair',
  'Wavy_Hair',
  'Wearing_Earrings',
  'Wearing_Hat',
  'Wearing_Lipstick',
  'Wearing_Necklace',
  'Wearing_Necktie',
  'Young',
  'Image'
]

df = pd.DataFrame(columns=attribute_names)

for i, name in enumerate(files):
    p = str(round(i/len(files) * 100, 0))
    sys.stdout.write("\rProgress: " + p + "% - current file: " + name)
    sys.stdout.flush()

    # load example image (already preprocessed)
    image = bob.io.base.load(SOURCE_DIR_PATH + name)
    # convert to the required data type
    image = image.astype(numpy.float32) / 255.
    # turn it into torch data structure
    tensor = torch.Tensor(image).unsqueeze(0)
    tensor = tensor.to(device)

    with torch.no_grad():
        # extract feature vector
        attributes = network(tensor)
        # transform it into 1D numpy array
        attributes = attributes.cpu().numpy().flatten()
    # we save the image name as   jpg, which allows simpler id identification
    attributes = numpy.append(attributes, name[:-3]+"jpg")
    series = pd.Series(attributes, index=df.columns)
    df = df.append(series, ignore_index=True)

df.to_csv(path_or_buf="./AFFACT-B_validation.txt")

