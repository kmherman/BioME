[![Build Status](https://travis-ci.com/kmherman/BioME.svg?token=ohrxcRT21DKp2pFP6NqQ&branch=main)](https://travis-ci.com/kmherman/BioME) [![Coverage Status](https://coveralls.io/repos/github/kmherman/BioME/badge.svg?branch=main)](https://coveralls.io/github/kmherman/BioME?branch=main)

<img src="https://github.com/kmherman/BioME/blob/main/doc/Biomelogo.png" width="200" />
CSE583 Final Project

___
### Brief Background
#### The Problem Addressed:
Making supervised machine learning accessible to non-data scientists interested in microbiome research.

#### The Data:
Four open source microbiome datasets were obtained and processed in Qiita [[1]](#1). The full bioinformatic pipeline was conducted in Qiita with QIIME2 [[2]](#2) and can be  found [here](https://qiita.ucsd.edu/analysis/description/32520/), and more information on the datasets as a whole can be found in the [Functional Specifications](https://github.com/kmherman/BioME/blob/main/doc/FunctionalSpec.md), [Component Specifications](https://github.com/kmherman/BioME/blob/main/doc/ComponentSpec.md), and our Homepage!

&downarrow;&downarrow;&downarrow;&downarrow;&downarrow;&downarrow;&downarrow;&downarrow;&downarrow;&downarrow;&downarrow;&downarrow;&downarrow;&downarrow;
|[Project Homepage](https://kmherman.github.io/BioME/)|
|---|
___

### Installation Instructions
##### This package assumes you have [Anaconda/Miniconda](https://docs.conda.io/en/latest/miniconda.html) and [git](https://github.com/git-guides/install-git) installed prior to starting!

#### Step 1: Create a new environment.
```
$ conda create -n biome_env python=3
```
In the above code, I have created a new environment and named it **biome_env** while also specifying that I want to be using python version 3 (incase you have both installed).

#### Step 2: Cloning the BioME repository.
In the terminal we will first active our new environment **biome_env**, change our directory, and then clone the repo! For this tutorial, I will be cloning the BioME to my Desktop. If you aren't sure where you are in your computer, don't worry! Type **pwd** (this means *print working directory*) into the terminal. This will tell you where you are! Now that might not be very helpful if you don't know whats in the directory! Type **ls** (*list*) in the terminal. This will tell you what is *in* your terminal. Once you have figured out where you are you can now use **cd** which means *change directory*. In this case, I'm just moving "forward" into a folder in this directory (my desktop!).
```
$ conda activate BioME_env
$ cd Desktop
$ git clone https://github.com/kmherman/BioME.git
$ cd BioME
```
Now take a look on your desktop screen. You'll now see the BioME file!
#### Step 3: Finish the install.
Now we just need to run the "setup .py" and you are good to go!
>Note: once this installs, you should be able to run the command from anywhere on your computer as long as **biome_env** is activated!
```
$ python setup.py install
```
###### You're now ready to go!


___
### BioME Tutorial:
For this tutorial we will be using the data from provided in BioME. You can get an over view of the directory structure below &downarrow;&downarrow;&downarrow;&downarrow;&downarrow;&downarrow;&downarrow;


>Note:
Your dataset should have a **minimum** of **~50** sample for [machine learning classifiers](https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html). Metadata columns that are used as the classifying variable should have a **minimum** of **10** samples per unique value. *Smaller counts will result in inaccurate models*.

#### Step 1: Start up BioME
We are going to assume that you are still in the BioME directory from above, but if not you'll just have to change the file path to the dataset!
```
$ biome_run.py
```
###### It might take a minute, but soon you should see something like this:
<img src="https://github.com/kmherman/BioME/blob/main/doc/images/biome.png" width="600" />

*Beautiful*

#### Step 2: Please enter the relative path to the OTU data:
(Write the path to the "bug" (OTU) table)
If you look in the data folder, you'll see three files:
  * "bug_OTU_rel.tsv" an OTU table with OTU counts in *relative abundance*
  * "bug_OTU_raw.tsv" an OTU table with OTU counts in *raw abundance*
  * "query_point.tsv" a single sample's OTU abundance (just for demonstration)
  * "FecesMeta.txt" a metadata file with our categorical data.
Take a look at the [Functional Specifications](https://github.com/kmherman/BioME/blob/main/doc/FunctionalSpec.md) doc to learn more about the data!

Write the path to the OTU table you choose to use. I'll be using the *relative abundance*

>Note:
Below all prompts are **_CASE-SENSITIVE_!!!**
In the prompt write:
```
: biome/Data/bug_OTU_rel.tsv
```
>Tip: Instead of writing it all out you can drag the folder into the terminal and finish by writing the exact file.. at least on a mac...

#### Step 2: Please enter the relative path to the categorical data:
(Write the path to the meta data file)
```
: biome/Data/FecesMeta.txt
```
#### Step 3: Please list the categorical variables of interest:
(Provide the categories that will be used for classification)
In this instance, we will be using the "ML_diagnosis" column in the metadata file.
Healthy Humans (**HC**) and individuals with either Crohn's Disease (**CD**), Ulcerative Colitis (**UC**), Collagenous Colitis (**CC**), or Ileal Crohn's disease (**IC**). Full collection, DNA extraction, and 16sRNA amplicon sequencing methodologies for each study can be found in the provided papers [[3-5]](#3-5).
>Lists need to be comma-separated with no spaces

```
: HC,CD,UC,CC,IC
```
#### Step 4: What models would you like to test?

Some of these of models can take a long-time, which isn't to mean they aren't excellent options to run! But for the tutorial we will only run a few. Remember, no spaces!
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

  <em>How fast does each algorithm run?</em>
  Algorithm | Rank<sup>*
  ----------|-----
  mlp1   | 7
  mlp3   | 8
  lr     | 3
  rr     | 1
  dtree  | 9
  svc    | 6
  knn    | 2
  forest | 5
  gnb    | 4

###### *For more information on these ML algorithms, see the [Project Homepage](https://kmherman.github.io/BioME/)

```
: dtree,mlp1,mlp3
```
###### This will take a few minutes....

#### Step 5: Choose the trained model you would like to predict with:
<img src="https://github.com/kmherman/BioME/blob/main/doc/images/biome2.png" width="600" />

Looks like the best model was the **mlp1**: Multilayer perceptron with 1 hidden layers. Now you just need to decide if you want to use it to predict.
If you do type "yes" or "Yes":
```
? yes
```
#### Step 6: Please enter the path to the data that you would like to make a prediction for:
Here we will be using the "query_point.tsv" file!
```
: biome/Data/query_point.tsv
```
>Looks like CD, which looking at our FeceMeta file, is right!

Don't want the tutorial to end?
Check out the Demo at the [Project Homepage](https://kmherman.github.io/BioME/)!

>Enjoy!
___
### Directory Structure:
<img src="https://github.com/kmherman/BioME/blob/main/doc/images/biome_repo_structure.PNG" width="600" />


### References:
<a id="1">[1]</a> Antonio Gonzalez, Jose A. Navas-Molina, Tomasz Kosciolek, Daniel McDonald, Yoshiki Vázquez-Baeza, Gail Ackermann, Jeff DeReus, Stefan Janssen, Austin D. Swafford, Stephanie B. Orchanian, Jon G. Sanders, Joshua Shorenstein, Hannes Holste, Semar Petrus, Adam Robbins-Pianka, Colin J. Brislawn, Mingxun Wang, Jai Ram Rideout, Evan Bolyen, Matthew Dillon, J. Gregory Caporaso, Pieter C. Dorrestein & Rob Knight. **Qiita: rapid, web-enabled microbiome meta-analysis**. Nature Methods, volume 15, pages 796–798 (2018); https://doi.org/10.1038/s41592-018-0141-9
[Qiita website](https://qiita.ucsd.edu/)

<a id="2">[2]</a>  Bolyen E, Rideout JR, Dillon MR, Bokulich NA, Abnet CC, Al-Ghalith GA, Alexander H, Alm EJ, Arumugam M, Asnicar F, Bai Y, Bisanz JE, Bittinger K, Brejnrod A, Brislawn CJ, Brown CT, Callahan BJ, Caraballo-Rodríguez AM, Chase J, Cope EK, Da Silva R, Diener C, Dorrestein PC, Douglas GM, Durall DM, Duvallet C, Edwardson CF, Ernst M, Estaki M, Fouquier J, Gauglitz JM, Gibbons SM, Gibson DL, Gonzalez A, Gorlick K, Guo J, Hillmann B, Holmes S, Holste H, Huttenhower C, Huttley GA, Janssen S, Jarmusch AK, Jiang L, Kaehler BD, Kang KB, Keefe CR, Keim P, Kelley ST, Knights D, Koester I, Kosciolek T, Kreps J, Langille MGI, Lee J, Ley R, Liu YX, Loftfield E, Lozupone C, Maher M, Marotz C, Martin BD, McDonald D, McIver LJ, Melnik AV, Metcalf JL, Morgan SC, Morton JT, Naimey AT, Navas-Molina JA, Nothias LF, Orchanian SB, Pearson T, Peoples SL, Petras D, Preuss ML, Pruesse E, Rasmussen LB, Rivers A, Robeson MS, Rosenthal P, Segata N, Shaffer M, Shiffer A, Sinha R, Song SJ, Spear JR, Swafford AD, Thompson LR, Torres PJ, Trinh P, Tripathi A, Turnbaugh PJ, Ul-Hasan S, van der Hooft JJJ, Vargas F, Vázquez-Baeza Y, Vogtmann E, von Hippel M, Walters W, Wan Y, Wang M, Warren J, Weber KC, Williamson CHD, Willis AD, Xu ZZ, Zaneveld JR, Zhang Y, Zhu Q, Knight R, and Caporaso JG. **Reproducible, interactive, scalable and extensible microbiome data science using QIIME 2**. 2019. Nature Biotechnology 37: 852–857. https://doi.org/10.1038/s41587-019-0209-9
[QIIME2 website](qiime2.org)

<a id="3">[3]</a> Gevers D, Kugathasan S, Denson LA, Vázquez-Baeza Y, Van Treuren W, Ren B, Schwager E, Knights D, Song SJ, Yassour M, Morgan XC, Kostic AD, Luo C, González A, McDonald D, Haberman Y, Walters T, Baker S, Rosh J, Stephens M, Heyman M, Markowitz J, Baldassano R, Griffiths A, Sylvester F, Mack D, Kim S, Crandall W, Hyams J, Huttenhower C, Knight R, Xavier RJ.**The treatment-naive microbiome in new-onset Crohn's disease. Cell Host Microbe.**. 2014 Mar 12;15(3):382-392. doi: 10.1016/j.chom.2014.02.005. PMID: 24629344; PMCID: PMC4059512. https://pubmed.ncbi.nlm.nih.gov/24629344/

<a id="4">[4]</a> Daniel McDonald, Embriette Hyde, Justine W. Debelius, James T. Morton, Antonio Gonzalez, Gail Ackermann, Alexander A. Aksenov, Bahar Behsaz, Caitriona Brennan, Yingfeng Chen, Lindsay DeRight Goldasich, Pieter C. Dorrestein, Robert R. Dunn, Ashkaan K. Fahimipour, James Gaffney, Jack A. Gilbert, Grant Gogul, Jessica L. Green, Philip Hugenholtz, Greg Humphrey, Curtis Huttenhower, Matthew A. Jackson, Stefan Janssen, Dilip V. Jeste, Lingjing Jiang, Scott T. Kelley, Dan Knights, Tomasz Kosciolek, Joshua Ladau, Jeff Leach, Clarisse Marotz, Dmitry Meleshko, Alexey V. Melnik, Jessica L. Metcalf, Hosein Mohimani, Emmanuel Montassier, Jose Navas-Molina, Tanya T. Nguyen, Shyamal Peddada, Pavel Pevzner, Katherine S. Pollard, Gholamali Rahnavard, Adam Robbins-Pianka, Naseer Sangwan, Joshua Shorenstein, Larry Smarr, Se Jin Song, Timothy Spector, Austin D. Swafford, Varykina G. Thackray, Luke R. Thompson, Anupriya Tripathi, Yoshiki Vázquez-Baeza, Alison Vrbanac, Paul Wischmeyer, Elaine Wolfe, Qiyun Zhu, The American Gut Consortium, Rob Knight. **American Gut: an Open Platform for Citizen Science Microbiome Research**. mSystems May 2018, 3 (3) e00031-18; DOI: 10.1128/mSystems.00031-18 https://msystems.asm.org/content/3/3/e00031-18

<a id="5">[5]</a> Halfvarson, J., Brislawn, C., Lamendella, R. et al. **Dynamics of the human gut microbiome in inflammatory bowel disease**.  Nat Microbiol 2, 17004 (2017). https://doi.org/10.1038/nmicrobiol.2017.4
