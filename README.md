# AUTOMATIC MODEL SELECTION USING CLUSTER INDICES

Model selection is an important building block in any machine learning pipeline. Typically, a best model is chosen from a list of model choices based on cross validation. The initial model choices are predominantly made based on personal preferences. With the abundance of classification models in the literature, it becomes difficult to construct a selection set without such a preference. We propose a method for model selection based on clustering indices.




## Prerequisites

***Clustering *** 


Clustering is the task of grouping a set of objects in such a way that objects in the same group (called a cluster) are more similar (in some sense) to each other than to those in other groups (clusters).

***Cluster indices***  

Clustering Indices are the metrics used for  validation of clusters, formed by a clustering algorithm. Clustering indices are grouped as internal and external indices. The internal indices characterizes the data distribution directly by evaluating the clustering structure exclusively from dataset. For example, an internal index would use the proximity matrix of dataset to assess the validity of clusters in terms of diameter and scattering. An external index is a measure of agreement between two partitions where the first partition is  a prior known clustering structure  and the second results from the clustering procedure.




***AutoMS pipeline*** 


1. The dataset given to the system is pre-processed and either sub-sampled into bags or run on it's entirety(oneshot setting).
2. The bags are clustered using 4 clustering algorithms such as Spectral, Kmeans , Agglomerative & HDBSCAN. 
3. Result of each clustering algorithm is used for generating internal and external clustering indices  
4. The generated cluster indices are used by our AutoMS  regressor models to predict F1 scores for different classifier methods.
5. The predicted F1 scores are returned. 




## Requirements

**python version: Python 3**
**Must have python dev tools installed** - sudo apt-get install python3.6-dev

Package                           Version
--------------------------------- -------
* cython==0.29.13
* gap-stat==1.0.1
* h5py==2.9.0
* hdbscan==0.8.22
* importlib-metadata==0.20
* joblib==0.13.2
* lightgbm==2.2.3
* matplotlib==3.1.1
* mistune==0.8.4
* more-itertools==7.2.0
* numpy==1.17.1
* packaging==19.1
* pandas==0.25.1
* py==1.8.0
* pyamg==4.0.0
* python-dateutil==2.8.0
* pytz==2019.2
* PyYAML==5.1.2
* pyzmq==18.1.0
* scikit-learn==0.21.3
* scipy==1.3.1
* seaborn==0.9.0
* six==1.12.0
* sklearn==0.0
* statsmodels==0.10.1
* xgboost==0.90


### Installing requirements 

**python version: Python 3**
**Must have python dev tools installed** - sudo apt-get install python3.6-dev


The required packages can be installed using the command below:
'''
	cmd: pip install -r requirements.txt
'''


## Getting Started 






### Files  & Folders 



**FILES**
* AutoMS.py - Main driver script for model selection using cluster indices.
* eda.py - Driver script containing basic functionalities for  the experiment. 
* preprocess.py - Helper script for data preprocessing. 
* gen_indices.py - Driver script for generation cluster indices using.
* gen_f1_scores.py - Driver script for generating F1 scores(Ground Truth).
* external_indices.py -  Implementation of selected external cluster indices.
* internal_indices.py - Implementation of selected internal cluster indices. 
* gap.py ,gap_h.py, gap_s.py - Helper scripts for GAP statistics. 

**FOLDERS**
* data - Directory where the Datasets are stored.
* bags - Directory where the sub-sampled bags are stored.
* indices - Directory where the generated cluster indices are stored.
* model - Directory where the  system's models are stored.  
* true_f1_scores - Directory where the generated  ground truth F1-scores are stored.[uncomment "calculating ground truth" section in AutoMS.py] (optional)



### Dataset configuration

The Dataset for running the experiment should follow the below configurations:
*  Must be in **CSV**(Comma Separated Values) format.
*  The Header row must be first row of the dataset.
*  The Target column must be the last column of the dataset.
*  The dataset must not contain NA/NaN values. 
*  The dataset must be binary classifiable. i.e. The target column must contain only 2 classes.3
*  If the dataset is to be run in  bags setting(Random sub sampling into bags). Then pass 1 when"Enter setting {0:Oneshot , 1:Sub-sampled bags}".
*  If the dataset is to be run in  oneshot setting(dataset in it's entirety as one bag). Then pass 0 when"Enter setting {0:Oneshot , 1:Sub-sampled bags}".

**Datasets for testing  must be present  in "AutoMS/data/".**


## Parameters 

* If the dataset is to be run in  bags setting(Random sub sampling into bags). Then pass 1 when "Enter setting {0:Oneshot , 1:Sub-sampled bags}" 
* If the dataset is to be run in  oneshot setting(dataset in it's entirety as one bag). Then pass 0 when "Enter setting {0:Oneshot , 1:Sub-sampled bags}" 
* The system works on the assumption that the dataset in binary classifiable. That is, target column has only 2 labels.




## Running AutoMS

* Download the repository 
* Make sure the datasets are present in the data folder.
* Make sure the dataset follows the Dataset configuration mentioned above 
* Run AutoMS.py using the command 

	'''
	cmd: python3 AutoMS.py <dataset_name>.csv 


	Example:
			python3 AutoMS.py sonar.csv 
	''' 
* The predicted F1 scores file is created in the "AutoMS/" folder.




## Example Run 

An example run for sonar dataset can be found in "AutoMS/sonar.log".
Use command "python3 AutoMS.py sonar.csv " for an example run.
There are other toy datasets given in the data folder to play around with. 



## Authors
* Sudarsun Santhiappan
* Nitin Shravan


## Acknowledgments

* Thank you, R.Mukesh[IIITDM]
* Thank you, Jeshuren Chelladurai[IITM]

