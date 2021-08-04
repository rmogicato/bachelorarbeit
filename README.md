# Bachelor thesis
## Correlation Analysis of Facial Attributes with Respect To Face Identity

This repository contains the code for my Bachlor Thesis. This readme aims to describe how to reproduce the results of my thesis.

### CelebA Data
It is of course possible to use your own project structure (e.g., if the large amount of data requires an external hard drive).
For simplicity’s sake, I describe the project structure that I used.

* First, download the images of CelebA and place them in `bachelorarbeit/data/img_celeba/img_celeba/`.
* place the files `identity_CelebA.txt`, `list_attr_celeba.txt`, `list_landmarks_celeba.txt` and `list_eval_parition.txt` in `bachelorarbeit/data/txt_files`

The data can be downloaded here: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

### CNNs
This paper uses three CNNs, two AFFACT CNNs for facial attribute classification and an ArcFace CNN for facial recognition.

#### AFFACT preprocessing and extractions
In a first step we crop the images and sort them by their partition.
Execute `crop_images.py` to achieve this and make sure that SOURCE_DIR_PATH is set to your image directory.
The default output folders are set to be `bachelorarbeit/data/img_celeba/img_validation`, `/img_testing`, and `/img_training` according to the set the images are in.
To get the correct crop for AFFACT, the resolution variable `res` should be set to (224, 224).

To extract attribute information, execute `extract.py`. Make sure you set the correct model and network (unbalanced/balanced).
To change which source directory the images come from, simply edit the variable SOURCE_DIR_PATH to be the desired location.


#### Arcface preprocessing and extractions

### Arcface clustering

### Reweighting

Relevant papers

* [MOON](http://arxiv.org/abs/1603.07027)
* [AFFACT](http://arxiv.org/abs/1611.06158)
* [ECLIPSE](https://www.researchgate.net/publication/324999538_ECLIPSE_Ensembles_of_Centroids_Leveraging_Iteratively_Processed_Spatial_Eclipse_Clustering)


 
