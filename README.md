# Bachelor thesis
## Correlation Analysis of Facial Attributes with Respect To Face Identity

This repository contains the code for my Bachlor Thesis. This readme aims to describe how to reproduce the results of my thesis.

### 1 CelebA Data
It is of course possible to use your own project structure (e.g., if the large amount of data requires an external hard drive).
For simplicity’s sake, I describe the project structure that I used.

* First, download the images of CelebA and place them in `data/img_celeba/img_celeba/`.
* place the files `identity_CelebA.txt`, `list_attr_celeba.txt`, `list_landmarks_celeba.txt` and `list_eval_parition.txt` in `data/txt_files`

The data can be downloaded here: [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

### 2 CNNs
This paper uses three CNNs, two AFFACT CNNs for facial attribute classification and an ArcFace CNN for facial recognition.
Download the balanced AFFACT network here: [AFFACT-B](https://seafile.ifi.uzh.ch/d/58644ee482d34425b5a1/)
The unbalanced AFFACT network here: [AFFACT-U](https://seafile.ifi.uzh.ch/d/e3b27cd497c34639b082/)
And the ArcFace network here: [LResNet100E-IR,ArcFace@ms1m-refine-v2](https://github.com/deepinsight/insightface/wiki/Model-Zoo)

#### 2.1 AFFACT preprocessing and extractions
In a first step we crop and align the images and sort them by their partition.
Execute `AFFACT/crop_images.py` to achieve this and make sure that SOURCE_DIR_PATH is set to your image directory.
The default output folders are set to be `data/img_celeba/img_validation`, `/img_testing`, and `/img_training` according to the set the images are in.
To get the correct crop for AFFACT, the resolution variable `res` should be set to (224, 224).

To extract attribute information, execute `extract.py`. Make sure you set the correct model and network (unbalanced/balanced).
To change which source directory the images come from, simply edit the variable SOURCE_DIR_PATH to be the desired location.

#### 2.2 Arcface preprocessing and extractions
Again, we first need to crop and align the images by executing Execute `AFFACT/crop_images.py`, make sure to set the res to (112x112).
The images are now saved in `data/img_arcface` according to their partition.
To extract the facial information, execute `data/extract_arcface.py`. Make sure you have set the path to the arcface CNN and entered the desired partition. 

### 3 Arcface clustering
For reweighting with automatically generated identity labels, rather than just ground truth ones, we cluster the arcface extractions.
To receive clustered labels, execute `clustering.py` with the desired arcface extractions received in 2.2.
This returns a dataframe with two columns, one for the image name and one for the identity label.
To determine the number of clusters _n_, we calculate the silhouette score for a range of different _n_, based on the tuple `est_id`.
In the thesis we calculated these based on the true number of identites _tn_ with `est_ids = (tn*0.5, tn*1.5)`.
Of course you can also directly insert the number of estimated ids based on the results of the thesis, e.g. `est_ids = (1142, 1142)` for the test partition.


We can evaluate how accurate these automatically generated labels are with the file `evaluate_clustering.py`,
giving us the purity and NMI score of our clustered identities compared to the ground truth labels.

### 4 Reweighting
An example of this whole process can be seen in `master_file.py`, where each step is performed consecutively with extractions
from AFFACT-B using the reweighting formulas "square_sign" and both ground truth and clustered identity labels.
Below is a more detailed explanation for each step.

#### 4.1 Statistical measures
In a first step we calculate the mean and standard deviation of the extractions from a partition.
To do this, call the function `calculate_statistics()` in `calculate_std_mean.py`.
As a parameter you need give the path to the extractions, a dataframe of Ids (which specifies which image belongs to which id),
and a boolean that determines if the mean and standard deviation should consider class imbalance.

If balanced is set to True, the mean and standard deviation consider the class distribution of the training partition.
This is achieved by first calculating the distribution and then assigning a probabily to each class of all attributes.
The extracted values are then multiplied by this probability for a mean and standard deviation that consider class imbalance.

The identity dataframe can either be obtained from the ground truth labels (remember to only give the identity labels of the right partition)
or automatically clustering the arcface extractions and saving the labels as a txt file.
The arcface labels that were generated and used in the thesis can be found in  `ids/arcface_testing` and `ids/arcface_validation` for the test and validation partition.

#### 4.2 Reweighting extractions
To reweight the extractions of a partition, call the function `reweight_attributes()`
with the file location of the extractions, the dataframes of the mean, standard deviation and ids, and a string for the reweighting formula as parameters.

The reweighting formulas used in the thesis were "square_sign", "square_mean", "cube_sign", "cube_mean", "sigmoid1" and "cosine_mean".
In this step you should make sure that all dataframes use the same ids, either ground truth labels or clustered labels.

#### 4.3 Calculating error rates
Finally, we can call the function `calculate_accuracy` in `calculate_accuracy.py` which returns two dataframe, a less detailed one containing the balanced
and unbalanced error rate, and a more detailed one, which also contains the false positive rate and false negative rate.

### 5 Visualization Tools
A number of visualization scripts can be found in the folder `/visualizer`.

* `arcface_histogram.py` generates a histogram that compares the similarity of image pairs of the same and different identity
* `attribute_extraction_range.py` generates a boxplot that shows the range of the extracted values for each attribute.
* `pca_analysis.py` generates pca plots of the mean and standard deviation.
* `pca_analysis_by_attribute.py` generates pca plots  of the mean and standard deviation colorized by the values of each attribute.
* `probability_to_csv.py` saves a csv file of the probability values of the training partition.
* `properties_of_celebA.py` generates a bar plot of the class imbalance and a histogram of how many images are associated with the identites.

### 6 Relevant papers

* [MOON](http://arxiv.org/abs/1603.07027)
* [AFFACT](http://arxiv.org/abs/1611.06158)


 
