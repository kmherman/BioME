[![Build Status](https://travis-ci.com/kmherman/BioME.svg?token=ohrxcRT21DKp2pFP6NqQ&branch=main)](https://travis-ci.com/kmherman/BioME) [![Coverage Status](https://coveralls.io/repos/github/kmherman/BioME/badge.svg?branch=main)](https://coveralls.io/github/kmherman/BioME?branch=main)

<img src="https://github.com/kmherman/BioME/blob/main/doc/Biomelogo.png" width="200" />
CSE583 Final Project


|[Project Homepage](https://kmherman.github.io/BioME/)|
|---|


### Installation Instructions
##### This package assumes you have [Anaconda/Miniconda](https://docs.conda.io/en/latest/miniconda.html) and [git](https://github.com/git-guides/install-git) installed prior to starting!

##### Step 1: create a new environment.
In your terminal you will type:
```
$ conda create -n biome_env python=3
```
In the above code, I have created a new environment and named it **biome_env** while also specifying that I want to be using python version 3 (incase you have both installed).

##### Step 2: Cloning the BioME repository.
In the terminal we will first active our new environment **biome_env**, change our directory, and then clone the repo! For this tutorial, I will be cloning the BioME to my Desktop. If you aren't sure where you are in your computer, don't worry! type **pwd** (this means *print working directory*) into the terminal. This will tell you where you are! Now that might not be very helpful if you don't know whats in the directory! Type **ls** (*list*) in the terminal. This will tell you what is *in* your terminal. Once you have figured out where you are you can now use **cd** which means *change directory*. In this case, I'm just moving "forward" into a folder in this directory (my desktop!). We will then move into the
```
$ conda activate BioME_env
$ cd Desktop
$ git clone https://github.com/kmherman/BioME.git
$ cd BioME
```
Now take a look on your desktop screen. You'll now see the BioME file! __*Magic*__
##### Step 3: Finish the install.
Now we just need to run the "setup .py" and you are good to go!
###### Note: once this installs, you should be able to run the command from anywhere on your computer as long as **biome_env** is activated!
```
$ python setup.py install
```
###### You're now ready to go!



### BioME Tutorial:
For this tutorial we will be using the data from provided in BioME. You can get an over view of the directory structure below &downarrow;&downarrow;&downarrow;&downarrow;&downarrow;&downarrow;&downarrow;


>Note:
Your dataset should have a **minimum** of **~50** sample for [machine learning classifiers](https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html). Metadata columns that are used as the classifying variable should have a **minimum** of **10** samples per unique value. *Smaller counts will result in inaccurate models*.

##### Step 1: Start up BioME
We are going to assume that you are still in the BioME directory from above, but if not you'll just have to change the file path to the dataset!
```
$ biome_run.py
```
It might take a minute, but soon you should see something like this:
<img src="https://github.com/kmherman/BioME/blob/main/doc/images/biome.png" width="600" />

*Beautiful*
>Note:
Below all prompts are **_CASE-SENSITIVE_!!!**
##### Step 2: Please enter the relative path to the OTU data:
(Write the path to the "bug" (OTU) table)
If you look in the data folder, you'll see three files:
  * "bug_OTU_rel.tsv" an OTU table with OTU counts in *relative abundance*
  * "bug_OTU_raw.tsv" an OTU table with OTU counts in *raw abundance*
  * "query_point.tsv" a single sample's OTU abundance (just for demonstration)
  * "FecesMeta.txt" a metadata file with our categorical data.
Take a look at the [Functional Specifications](https://github.com/kmherman/BioME/blob/main/doc/FunctionalSpec.md) doc to learn more about the data!

Write the path to the OTU table you choose to use. I'll be using the *relative abundance*
In the prompt write:
```
: biome/Data/bug_OTU_rel.tsv
```
>Tip: Instead of writing it all out you can drag the folder into the terminal and finish by writing the exact file.. at least on a mac...

##### Step 2: Please enter the relative path to the categorical data:
(Write the path to the meta data file)
```
: biome/Data/FecesMeta.txt
```
##### Step 3: Please list the categorical variables of interest:
(Provide the categories that will be used for classification)
In this instance, we will be using the "ML_diagnosis" column in the metadata file.
Healthy Humans (**HC**) and individuals with either Crohn's Disease (**CD**), Ulcerative Colitis (**UC**), Collagenous Colitis (**CC**), or Ileal Crohn's disease (**IC**).
>Lists need to be comma-separated with no spaces

```
: HC,CD,UC,CC,IC
```
##### Step 4: What models would you like to test?
###### For more information on these ML algorithms, see the [Project Homepage](https://kmherman.github.io/BioME/)
  * Machine learning algorithm abbreviations:
    * **mlp1**: Multilayer perceptron with 1 hidden layer
    * **mlp3**: Multilayer perceptron with 3 hidden layers
    * **lr**: logistic regression
    * **rr**: ridge classifier (L2 regularizer)
    * **dtree**: decision tree
    * **svc**: support vector classifier
    * **knn**: k-nearest neighbors algorithm (implemented with PCA)
    * **forest**: random forest
    * **gnb**: Gaussian Naive-Bayes
    * **all**: train and evaluate every machine learning algorithm available (all above)

Some of these of models can take a long-time, which isn't to mean they aren't excellent options to run! But for the tutorial we will only run a few. Remember, no spaces!
```
: dtree,mlp1,mlp3
```
###### This will take a few minutes....

##### Step 5: Choose the trained model you would like to predict with:
<img src="https://github.com/kmherman/BioME/blob/main/doc/images/biome2.png" width="600" />

Looks like the best model was the **mlp1**: Multilayer perceptron with 1 hidden layers. Now you just need to decide if you want to use it to predict.
If you do type "yes" or "Yes":
```
? yes
```
##### Step 6: Please enter the path to the data that you would like to make a prediction for:
Here we will be using the "query_point.tsv" file!
```
: biome/Data/query_point.tsv
```
>Looks like CD, which looking at our FeceMeta file, is right!

Don't want the tutorial to end?
Check out the Demo at the [Project Homepage](https://kmherman.github.io/BioME/)!

>Enjoy!
### Directory Structure:
<img src="https://github.com/kmherman/BioME/blob/main/doc/images/biome_repo_structure.PNG" width="600" />
