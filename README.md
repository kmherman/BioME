[![Build Status](https://travis-ci.com/kmherman/BioME.svg?token=ohrxcRT21DKp2pFP6NqQ&branch=main)](https://travis-ci.com/kmherman/BioME) [![Coverage Status](https://coveralls.io/repos/github/kmherman/BioME/badge.svg?branch=main)](https://coveralls.io/github/kmherman/BioME?branch=main)

<img src="https://github.com/kmherman/BioME/blob/main/doc/Biomelogo.png" width="200" />
CSE583 Final Project


|[Project Homepage](https://kmherman.github.io/BioME/)|
|---| 


### Installation Instructions  
* Clone the BioME repository: git clone https://github.com/kmherman/BioME.git
* Run the setup.py file to install software: python setup.py install
* Create a conda environment to be used with this software: conda env create -q -n biome --file BioME_environment.yml
* Activate this new conda environment: conda activate biome
  

### How to utilize the BioME software:
1. Install software (above)
2. Move execution script to desired directory: "cp biome/scripts/biome_run.py dir_location"
3. Execute the biome_run.py file by typing "python3 biome_run.py"
4. Provide user input for prompts (input is case-sensitive; lists need to be comma-separated with no space)  
  * Machine learning algorithm abbreviations:
    * mlp1: Multilayer perceptron with 1 hidden layer
    * mlp3: Multilayer perceptron with 3 hidden layers
    * lr: logistic regression
    * rr: ridge classifier (L2 regularizer)
    * dtree: decision tree
    * svc: support vector classifier
    * knn: k-nearest neighbors algorithm (implemented with PCA)
    * forest: random forest
    * gnb: Gaussian Naive-Bayes
    * all: train and evaluate every machine learning algorithm available (all above)  
    (*for more information on these ML algorithms, see the attached GitHub page.)  

### Directory structure:
<img src="https://github.com/kmherman/BioME/blob/main/doc/images/repo_structure.PNG" width="700" />
