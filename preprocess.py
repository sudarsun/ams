# -*- coding: utf-8 -*-

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning) 


import pandas as pd 
import numpy as np

# For stratified sampling
from sklearn.model_selection import StratifiedShuffleSplit
import eda
import glob 
import collections
from multiprocessing import Pool       
import subprocess
import pathlib
import os 
import sys
import shutil
from pandas import DataFrame

import h5py
import pickle
import random

from time import time
from collections import Counter






def split(file_name):

	# Uncomment the following code segment for CSV data file -- start
	''' column (aka. feature) seperator, i.e, sequence of characters (or a regex) that seperate two features of a data sample '''

	seperator = ',' # default
	# seperator = '\s+' # combination of spaces and tabs
	# seperator = None # autodetect delimiter (warning:  may not work with certain combinations of parameters)
	# seperator = '<replace with a character or regular expression>' # examples: '\r\t'


	''' number of initial lines <int> or list of line numbers <list> of the data file to skip before start of data samples 
	Note: include the commented and blank lines in the count of intial lines to skip. Don't skip over header line (with column names) if any '''

	skiplines = None # default: no lines to skip (aka. skiplines = 0)
	# skiplines = <replace with number of initial lines to skip>
	# skiplines = [<replace with list of line numbers to skip>]


	''' relative zero-index based line number of header line containing column names to use
	Note: 
		* The indexing starts (from line number=0) for lines immediately following the skipped lines.
		* Blank lines are ignored in the indexing.
		* All lines following the skipped lines until the header line are ignored.
	'''

	# headerline = None # default: No header lines (containing column names) to use.
	headerline = 0 # the first non-blank line following skipped lines contains column names to use.
	# headerline = <replace with relative zero-index of header line>


	''' List of columns to use from the data file 
	Note: 
		* Use list of column names (as inferred from header line) or zero-based column indices (only if no header line)
		* Include the 'target' value column in the list of columns to use
	'''

	usecolumns = None # default: Use all columns from data file
	# usecolumns = [<replace with list of columns to use>]


	''' relative zero-based index (among selected columns) or column name (as inferred from header line) of 'target' values column '''

	targetcol = -1 # default: last column among list of selected columns
	# targetcol = 0 # first column among list of selected columns
	# targetcol = '<replace with name of target column as inferred from header line (if any)>'


	''' Should target values be treated as nominal values and be encoded (with zero-indexed integers) '''

	encode_target = True # default: Encode target values
	# encode_target = False # Target values represent a non-nominal (or continous) feature and should not be encoded


	''' List of column names (as inferred from header line) or absolute column indices (index of column as in data file) if no header line
	of nominal categorical columns to encode. 

	Note:
		* nominal_cols='infer' infers all string dtype columns with relatively large number of unique entries as 'string' or 'date' features and drops them from the dataset.
	'''

	nominal_cols = 'infer' # default: infer all string dtype columns with reasonably few unique entries as nominal columns. See Note (1).
	# nominal_cols = 'all' # All columns are nominal categorical
	#nominal_cols = None # No nominal categorical columns
	# nominal_cols = [<list of nominal categorical columns to encode>]

	''' List of strings to be inferred as NA (Not Available) or NaN (Not a Number) in addition to the default NA string values.
	Default NA values :  ‘’, ‘#N/A’, ‘#N/A N/A’, ‘#NA’, ‘-1.#IND’, ‘-1.#QNAN’, ‘-NaN’, ‘-nan’, ‘1.#IND’, ‘1.#QNAN’, ‘N/A’, ‘NA’, ‘NULL’, ‘NaN’, ‘n/a’, ‘nan’, ‘null’
	'''

	na_values = None # default: no additional strings (asides default strings) to infer as NA/NaN
	# na_values = [<list of additional strings to infer as NA/NaN>]
	#na_values = ['?']

	''' Dictionary of other keyword arguments accepted by :func:`pandas.read_csv` (Keyword Arguments: comment, lineterminator, ...) '''
	kargs = {}

	print("--------------------------------------------------------------------------")




	main = eda.eda()

	main.read_data_csv('/data/nitin/craved_f1_2_clusters/data/files/'+file_name, sep=seperator, skiprows=skiplines, header_row=headerline, usecols=usecolumns, target_col=targetcol, encode_target=encode_target, categorical_cols=nominal_cols, na_values=na_values, nrows=None, **kargs)


	dataset_name,file_type = file_name.split('.')


	print(dataset_name)

	if main.target is not None and main.data.shape[0] >= 700 :

		sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3)
		sss.get_n_splits(main.data, main.target)


		for train_index, test_index in  sss.split(main.data, main.target):


			train = pd.DataFrame()
			train_target = pd.DataFrame()
			train = main.data[train_index]
			train_target = main.target[train_index]
			train = np.append(train, train_target[:, None], axis=1)
			train_df =pd.DataFrame(data= train)
			print("train size")
			print(train_df.shape)
			print('train dist')
			print(collections.Counter(main.target[train_index]))
			train_filename = str(dataset_name+'_train.csv')
			os.chdir("/data/nitin/craved_f1_2_clusters/data/train_test_splits") 
			train_df.to_csv(train_filename,header=False,index=False,index_label=False)

			test = pd.DataFrame()
			test_target = pd.DataFrame()
			test = main.data[test_index]
			test_target = main.target[test_index]
			test = np.append(test, test_target[:, None], axis=1)
			test_df =pd.DataFrame(data= test)
			print("test size")
			print(test_df.shape)
			test_filename = str(dataset_name+'_test.csv')            
			test_df.to_csv(test_filename,header=False,index=False,index_label=False)
			#print(np.bincount(main.target[test_index]))
			print('test dist')
			print(collections.Counter(main.target[test_index]))
			#print('if repetation in train and test')
			#print(train_index in test_index)
			os.chdir('/data/nitin/craved_f1_2_clusters/data/') 
			print("--------------------------------------------------------------------------")
			return train_df,test_df
	else:
				print("error: ignored due to small size of dataset {} ".format(main.data.shape[0]))
				print("--------------------------------------------------------------------------")
				#sys.exit(1)




def random_sampling(file_name):
	main = eda.eda()

	# Path and file name of the data file
	# Uncomment the following code segment for CSV data file -- start
	''' column (aka. feature) seperator, i.e, sequence of characters (or a regex) that seperate two features of a data sample '''

	seperator = ',' # default
	''' number of initial lines <int> or list of line numbers <list> of the data file to skip before start of data samples 
	Note: include the commented and blank lines in the count of intial lines to skip. Don't skip over header line (with column names) if any '''

	skiplines = None # default: no lines to skip (aka. skiplines = 0)

	''' relative zero-index based line number of header line containing column names to use
	Note: 
		* The indexing starts (from line number=0) for lines immediately following the skipped lines.
		* Blank lines are ignored in the indexing.
		* All lines following the skipped lines until the header line are ignored.
	'''

	headerline = 0 # default: No header lines (containing column names) to use.
	''' List of columns to use from the data file 
	Note: 
		* Use list of column names (as inferred from header line) or zero-based column indices (only if no header line)
		* Include the 'target' value column in the list of columns to use
	'''

	usecolumns = None # default: Use all columns from data file
	''' relative zero-based index (among selected columns) or column name (as inferred from header line) of 'target' values column '''

	targetcol = -1 # default: last column among list of selected columns

	''' Should target values be treated as nominal values and be encoded (with zero-indexed integers) '''

	encode_target = True # default: Encode target values

	''' List of column names (as inferred from header line) or absolute column indices (index of column as in data file) if no header line
	of nominal categorical columns to encode. 

	Note:
		* nominal_cols='infer' infers all string dtype columns with relatively large number of unique entries as 'string' or 'date' features and drops them from the dataset.
	'''
	nominal_cols='infer'
	#nominal_cols = None # No nominal categorical columns
	''' List of strings to be inferred as NA (Not Available) or NaN (Not a Number) in addition to the default NA string values.
	Default NA values :  ‘’, ‘#N/A’, ‘#N/A N/A’, ‘#NA’, ‘-1.#IND’, ‘-1.#QNAN’, ‘-NaN’, ‘-nan’, ‘1.#IND’, ‘1.#QNAN’, ‘N/A’, ‘NA’, ‘NULL’, ‘NaN’, ‘n/a’, ‘nan’, ‘null’
	'''
	na_values = None # default: no additional strings (asides default strings) to infer as NA/NaN

	''' Dictionary of other keyword arguments accepted by :func:`pandas.read_csv` (Keyword Arguments: comment, lineterminator, ...) '''
	kargs = {}
	curr_pwd = str(os.getcwd())
	data_pwd = curr_pwd + str('/data/')
	main.read_data_csv(data_pwd+file_name, sep=seperator, skiprows=skiplines, header_row=headerline, usecols=usecolumns, target_col=targetcol, encode_target=encode_target, categorical_cols=nominal_cols, na_values=na_values, nrows=None, **kargs)
	
	#main.load_data(data,target)

	""" Perform dummy coding of nominal columns (or features) """
	nominal_columns = 'infer' # Default: Use list of nominal columns supplied to (or inferred by) :func:`read_data_csv` or :func:`read_data_arff`
	#print(main.dat)
	main.dummy_coding(nominal_columns=nominal_columns)
	#main.standardize_data()
	""" Perform repeated stratified  sampling (with replacement across samplings) of dataset into bags """
	# Name of folder that contains sampled bags and metadata files





	bags_setting  = int(input("Enter setting {0:Oneshot , 1:Sub-sampled bags} :"))

	if(bags_setting == 0):
		print("Experiment carried out in oneshot setting ")
		sample_size = main.n_samples

	else:
		print("Experiment carried out in Random sub-sampled bags setting ")
		if(main.n_samples > 1000):
			print('choosing default bags_size=500')
			sample_size = 500
		else:
			if(main.n_samples >500 ):
				print("dataset size less than 1000")	
				print("bag size is fixed to be half of dataset size ")
				sample_size = int(main.n_samples/2)
			else:
				print("the dataset size is less than 500")
				sample_size  = int(input("please enter the desired bag size:")) 




	# """applying thumb rule for n_bags and sample_size"""
	# if("oneshot" in file_name):
	# 	sample_size = main.n_samples
	# else:
	# 	if(main.n_samples > 1000):
	# 			print('choosing default bags_size=500')
	# 			sample_size = 500
	# 	else:
	# 		print("dataset size less than 1000")	
	# 		print("bag size is fixed to be half of dataset size ")
	# 		sample_size = int(main.n_samples/2)	
				





	"""Determining number of bags to be drawn from the dataset """
	# print("bag n_samples :{}".format(main.n_samples))
	unique_data_pts = 0.63 * sample_size 
	no_partitions = main.n_samples / unique_data_pts
	n_bags = np.round(5 * no_partitions)
	# if('oneshot' in file_name):
	# 	n_bags=1

	if(bags_setting==0):
		n_bags = 1




	# Display bag size and number of bags
	print("Bag size :{}".format(sample_size))
	print("number of bags:{}".format(np.round(n_bags)))


	# unique bag name for bags (after sampling)
	bag_name= file_name.split('.')[0] + '_' + 'stratified'
	
	try:
		bags_pwd  = curr_pwd + str('/bags/') 
		os.chdir(str(bags_pwd))        
		os.mkdir(file_name.split('.')[0])

	except OSError as err:
		print("error: Unable to write sampled data bags to disk.\n{0}".format(err))
		print("Over-writing to  the same folder ",file_name.split('.')[0])
		shutil.rmtree(bags_pwd+ file_name.split('.')[0])
		
		os.mkdir(file_name.split('.')[0])

		# sys.exit(1)
	

	# Validating the sample size parameter
	if isinstance(sample_size, int) and (sample_size>0 and sample_size<=main.n_samples):
		pass

	elif isinstance(sample_size, float) and (sample_size>0.0 and sample_size<=1.0):
		pass

	else:
		print("error: Invalid sampling size encountered")
		sys.exit(1)



	file_prefix = None

	if file_prefix is None:
		file_prefix = ''

	else:
		file_prefix = bag_name + '/'



	# Random sampling  procedure 
	bincount = np.bincount(main.target)
	# print("bincount:{}".format(bincount))
	
	#count of data points belonging to classes
	class0_count = bincount[0]
	class1_count = bincount[1]
	# no.of.random samples to be derived from each class
	n_class0 = round((bincount[0]/np.sum(bincount))*sample_size)
	n_class1 = round((bincount[1]/np.sum(bincount))*sample_size)

	#getting all index of datapoints belonging to class
	class0_index = np.where(main.target == 0)
	class1_index = np.where(main.target == 1)
	bag_number = 0
	tot_rep = 0 
	sample_bags =[]
	for i in range(int(n_bags)):
	#creating random index numbers
		random.seed()
		prob0 = 1 / class0_count
		prob1 = 1 / class1_count

		rand_class0_index=[]
		for j in range (int(n_class0)):
				rand_num = random.uniform(0,1)
				cdf_val = int(rand_num * class0_count) 
				rand_class0_index.append(cdf_val)


		#  repetations in class 0 
		rep0= [item for item, count in Counter(rand_class0_index).items() if count > 1]

		rand_class1_index=[]
		for j in range (int(n_class1)):
			rand_num = random.uniform(0,1)
			cdf_val = int(rand_num * class1_count) 
			rand_class1_index.append(cdf_val)

		#  repetations in class 1
		rep1= [item for item, count in Counter(rand_class1_index).items() if count > 1]

		sampled_class0 = [class0_index[0][x] for x in rand_class0_index]
		sampled_class1 = [class1_index[0][x] for x in rand_class1_index]

		#print( sampled_class0 in sampled_class1)

		final_indices = sampled_class0 + sampled_class1

		rep= [item for item, count in Counter(final_indices).items() if count > 1]
		tot_rep =  tot_rep + len(rep)

		#fetching datapoints to the corresponding index numbers
		sampled_data = dict.fromkeys(['data', 'target','classes','n_samples','n_features','column_names','column_categories','dataset_name','bag_number'])
		sampled_data['data'], sampled_data['target'] = main.data[final_indices], main.target[final_indices]	
		sampled_data['classes']=label_cnt_dict(main.target) if main.target is not None else None
		sampled_data['n_samples']=main.n_samples # Not inferrable from classes, if target=None
		sampled_data['n_features']=main.n_features,
		sampled_data['column_names']=main.columns_,
		sampled_data['column_categories']=main.columns_categories_ if hasattr(main, 'columns_categories_') else None	
		sampled_data['dataset_name']= file_name
		sampled_data['bag_number']= bag_number
        
		os.chdir(bags_pwd+file_name.split('.')[0])


		df = DataFrame.from_records(main.data[final_indices])
		df[''] = main.target[final_indices]


		# saving bags in the data folder
		df.to_csv(str(file_prefix +file_name.split('.')[0] +"_bag"+str(bag_number+1)+".csv"),header=None,index=False,index_label=False)
		bag_number = bag_number + 1
		sample_bags.append(sampled_data) 
		
	os.chdir(curr_pwd)
	return sample_bags



def label_cnt_dict(labels):
		unique, counts = np.unique(labels, return_counts=True)
		return dict(zip(unique, counts))
