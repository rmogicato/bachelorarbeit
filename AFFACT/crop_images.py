import bob.io.base
import bob.ip.facedetect as fd
import bob.io.image
import bob.ip.color
import bob.ip.base
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os.path, sys

SOURCE_DIR_PATH = "../data/img_celeba/img_celeba/"

# this file contains manually labeled facial landmarks
flandmark = pd.read_csv("../data/txt_files/list_landmarks_celeba.txt", sep="\s+", header=0)
eval_partition = pd.read_csv("../data/txt_files/list_eval_partition.txt", sep="\s+", names=["Image", "Partition"])
files = os.listdir(SOURCE_DIR_PATH)

res = (224, 224)


geom = bob.ip.base.GeomNorm(
    rotation_angle = 0.,
    scaling_factor=1.,
    crop_size=res,
    crop_offset=(res[0]/2, res[1]/2)
)

# this crops images for AFFACT (224x224)
def crop(image, annotations):
    # get coordinates
    re = [annotations["righteye_x"].values[0], annotations["righteye_y"].values[0]]
    le = [annotations["lefteye_x"].values[0], annotations["lefteye_y"].values[0]]
    rm = [annotations["rightmouth_x"].values[0], annotations["rightmouth_y"].values[0]]
    lm = [annotations["leftmouth_x"].values[0], annotations["leftmouth_y"].values[0]]

    # eye and mouth center
    ec = np.array([(re[0]+le[0])/2, (re[1]+le[1])/2])
    mc = np.array([(rm[0]+lm[0])/2, (rm[1]+lm[1])/2])

    med = np.linalg.norm(mc-ec)

    geom.rotation_angle = 90 - bob.ip.base.angle_to_horizontal(re, le)

    # bbox
    fac = 5.5*med
    top = ec[1] - 0.45*fac
    left = ec[0] - 0.5 * fac

    center = (top+fac/2, left+fac/2)
    geom.scaling_factor = res[0]/fac

    inmask = np.ones(image.shape[-2:], dtype=np.bool)
    outimage = np.ndarray((3,) + res)
    outmask = np.ones(res, dtype=np.bool)

    for i in range(3):
        geom(image[i], inmask, outimage[i], outmask, center)
    bob.ip.base.extrapolate_mask(outmask, outimage)
    return outimage

# this function crops images for arcface (112x112)
def crop_arcface(image, annotations):
    # get coordinates
    re = [annotations["righteye_x"].values[0], annotations["righteye_y"].values[0]]
    le = [annotations["lefteye_x"].values[0], annotations["lefteye_y"].values[0]]
    rm = [annotations["rightmouth_x"].values[0], annotations["rightmouth_y"].values[0]]
    lm = [annotations["leftmouth_x"].values[0], annotations["leftmouth_y"].values[0]]

    # eye and mouth center
    ec = np.array([(re[0]+le[0])/2, (re[1]+le[1])/2])
    mc = np.array([(rm[0]+lm[0])/2, (rm[1]+lm[1])/2])

    # cross(image, np.flip(ec.astype(int)), 5, (255, 255, 0))
    # cross(image, np.flip(mc.astype(int)), 5, (255, 255, 0))

    med = np.linalg.norm(mc-ec)

    geom.rotation_angle = 90 - bob.ip.base.angle_to_horizontal(re, le)

    # bbox

    # 112/38 is the new relation between eye-mouth distance and the image size
    # source: calculations of bachelor thesis
    fac = 112/38 * med
    top = ec[1] - 0.45*fac
    left = ec[0] - 0.5 * fac

    center = (top+fac/2, left+fac/2)
    geom.scaling_factor = res[0]/fac

    inmask = np.ones(image.shape[-2:], dtype=np.bool)
    outimage = np.ndarray((3,) + res)
    outmask = np.ones(res, dtype=np.bool)

    for i in range(3):
        geom(image[i], inmask, outimage[i], outmask, center)
    bob.ip.base.extrapolate_mask(outmask, outimage)
    return outimage


# if res is (224, 224) then all images in source_dir are cropped to a size of 224x224 pixels and saved them in dir "img_celeba"
# if res is (112, 112) then all images in source_dir are cropped to a size of 112x112 pixels and saved them in dir "img_arcface"
for i, name in enumerate(files):

    p = str(round(i / len(files) * 100, 0))
    sys.stdout.write("\rProgress: " + p + "% - current file: " + name)
    sys.stdout.flush()

    # print("current file:", name)
    # import image
    image_color = bob.io.base.load(SOURCE_DIR_PATH + name)

    # creating image for grayscale for facial landmark detection
    image_gray = bob.ip.color.rgb_to_gray(image_color)

    # we get the facial landmarks of this person
    keypoints = flandmark.loc[flandmark["id"] == name]

    if res == (112, 112):
        image_out = crop_arcface(image_color, keypoints)
        dest = "img_arcface"
    else:
        image_out = crop(image_color, keypoints)
        dest = "img_celeba"

    # setting the correct destination folder
    partition = eval_partition.loc[eval_partition["Image"] == str(name)]["Partition"].values[0]
    if partition == 0:
        # training images
        DEST_DIR_PATH = "../data/" + dest + "/img_training/"
    elif partition == 1:
        # validation images
        DEST_DIR_PATH = "../data/" + dest + "/img_validation/"
    else:
        DEST_DIR_PATH = "../data/" + dest + "/img_testing/"

    # converting image to be compatible with matplot
    img_view_for_matplotlib = bob.io.image.to_matplotlib(image_out)

    # uncomment to print images
    """
    imgplot = plt.imshow(img_view_for_matplotlib.astype(np.uint8))
    plt.title(name)
    plt.show()
    """

    # saving images in dest_dir with the same name as png (to avoid noise)
    plt.imsave(DEST_DIR_PATH + name[:-3] + "png", img_view_for_matplotlib.astype(np.uint8))
