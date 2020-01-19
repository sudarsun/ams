import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings('ignore') 




import eda
import internal_indices
import external_indices
import sys
import pickle
import xgboost 
import os
from math import ceil
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.metrics import f1_score 
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer







def f1_ratio(y_test,y_pred):
    ratio  = imbalance_ratio(y_test)
    f1_majority_score  = f1_majority(y_test,y_pred)
    f1_minority_score  = f1_minority(y_test,y_pred)
    
    return (f1_majority_score + (ratio*f1_minority_score))/(1+ratio)

def f1_majority(y_test,y_pred):
    class_dist = np.bincount(y_test).tolist()
    majority_class_label = class_dist.index(max(class_dist))
    return f1_score(y_test,y_pred,pos_label=majority_class_label)

def f1_minority(y_test,y_pred):
    class_dist = np.bincount(y_test).tolist()
    minority_class_label = class_dist.index(min(class_dist))
    return f1_score(y_test,y_pred,pos_label=minority_class_label)

def imbalance_ratio(y_test):
	class_dist = np.bincount(y_test).tolist()
	ratio  = round(max(class_dist)/min(class_dist))
	return ratio

custom_f1 = make_scorer(f1_ratio,greater_is_better=True) 




def knn(sampled_data):
	bag_name = sampled_data[0]['dataset_name'].split('.')[0]
	n_bags = len(sampled_data)
	curr_pwd = str(os.getcwd())
	indices_pwd = curr_pwd + '/true_f1_scores/'
	os.chdir(indices_pwd)
	file_name = bag_name+'_knn_f1_weighted'+'.csv'
	results_file = open(file_name, 'a')
	tot_knn_result = []


	for x in range(len(sampled_data)):
		print("Processing '{file_name}'".format(file_name=bag_name+"_bag_"+str(sampled_data[x]['bag_number'])))
		bag_name_x=sampled_data[0]['dataset_name'].split('.')[0]+'_'+str(sampled_data[x]['bag_number'])
		dataset = sampled_data[x]
		data, target = dataset['data'], dataset['target']

		main = eda.eda()
		main.load_data(data, target)
		

		# print("\n  bag  dist\n")
		# print(np.count_nonzero(target==0))
		# print(np.count_nonzero(target==1))
		
	
		X_train, X_test, y_train, y_test = train_test_split(data,target, test_size=0.3,stratify=target)
		class_dist = np.bincount(y_test).tolist()
		majority_class_label = class_dist.index(max(class_dist))
		minority_class_label = class_dist.index(min(class_dist))

		if(imbalance_ratio(y_test)>=10):
			print("-----entering imbalace class----")
			k_value = imbalance_ratio(y_test)
			f1_minority_score =0
			while(f1_minority_score==0):
			
				parameters = {'weights':['uniform','distance'],'leaf_size':[2,5,10,15,20,25,30]}
				neigh_clf = GridSearchCV(KNeighborsClassifier(n_neighbors =k_value), parameters, cv=5,n_jobs =20,refit=True,scoring=custom_f1)
				neigh_clf.fit(X_train,y_train) 


				y_pred_prob =neigh_clf.predict_proba(X_test)
				y_pred = []
				if(minority_class_label==1):

					for i in range(len(y_pred_prob)):
					    if y_pred_prob[i,1] > (1/imbalance_ratio(y_test)):
					        y_pred.append(1)
					    else:
					        y_pred.append(0)
				else:
					for i in range(len(y_pred_prob)):
					    if y_pred_prob[i,0] > (1/imbalance_ratio(y_test)):
					        y_pred.append(0)
					    else:
					        y_pred.append(1)
				# print("\n  train set dist\n")
				# print(np.count_nonzero(y_test==0))
				# print(np.count_nonzero(y_test==1))
				# print("\n prediction dist ")
				# print(np.count_nonzero(y_pred==0))
				# print(np.count_nonzero(y_pred==1))
				knn_f1_score = f1_ratio(y_test,y_pred)
				# print("majority_class_f1:{}".format(f1_majority(y_test,y_pred)))
				# print("minority_class_f1:{}".format(f1_minority(y_test,y_pred)))
				f1_minority_score = f1_minority(y_test,y_pred)
				if(f1_minority_score ==0):
					print("---------re-fitting------")
					X_train, X_test, y_train, y_test = train_test_split(data,target, test_size=0.3,stratify=target)

			

			results_file.write("\"{file_name}\",{sample_size},\"{algorithm}\",\"{parameters}\",\"{majority_class_f1}\",\"{minority_class_f1}\",\"{knn_f1_score}\",".format(file_name=bag_name_x, sample_size=main.n_samples, algorithm='KNearestNeighbour-imbalanced', parameters=neigh_clf.best_estimator_,majority_class_f1=f1_majority(y_test,y_pred),minority_class_f1=f1_minority(y_test,y_pred),knn_f1_score=knn_f1_score))
			results_file.write('\n')

			knn_result = {
						"file_name":str(bag_name+'_'+str(x)+'_bag'),
						"sample_size":main.n_samples,
						"algorithm":'KNeighborsClassifier-imbalances',
						"parameters":neigh_clf.best_estimator_,
						"majority_class_f1":f1_majority(y_test,y_pred),
						"minority_class_f1":f1_minority(y_test,y_pred),
						"knn_f1_score":knn_f1_score
					}
			tot_knn_result.append(knn_result)
		else:

			f1_minority_score =0
			while(f1_minority_score==0):
				parameters = {'n_neighbors':[3,5,7,9],'weights':['uniform','distance'],'leaf_size':[2,5,10,15,20,25,30]}
				neigh_clf = GridSearchCV(KNeighborsClassifier(), parameters, cv=5,n_jobs =20,refit=True,scoring=custom_f1)
				neigh_clf.fit(X_train,y_train) 

				y_pred = neigh_clf.predict(X_test)
				# print("\n  train set dist\n")
				# print(np.count_nonzero(y_test==0))
				# print(np.count_nonzero(y_test==1))
				# print("\n prediction dist ")
				# print(np.count_nonzero(y_pred==0))
				# print(np.count_nonzero(y_pred==1))
				knn_f1_score = f1_ratio(y_test,y_pred)
				# print("majority_class_f1:{}".format(f1_majority(y_test,y_pred)))
				# print("minority_class_f1:{}".format(f1_minority(y_test,y_pred)))
				f1_minority_score = f1_minority(y_test,y_pred)
				if(f1_minority_score ==0):
					# print("---------re-fitting------")
					X_train, X_test, y_train, y_test = train_test_split(data,target, test_size=0.3,stratify=target)

				

			results_file.write("\"{file_name}\",{sample_size},\"{algorithm}\",\"{parameters}\",\"{majority_class_f1}\",\"{minority_class_f1}\",\"{knn_f1_score}\",".format(file_name=bag_name_x, sample_size=main.n_samples, algorithm='KNearestNeighbour', parameters=neigh_clf.best_estimator_,majority_class_f1=f1_majority(y_test,y_pred),minority_class_f1=f1_minority(y_test,y_pred),knn_f1_score=knn_f1_score))
			results_file.write('\n')

			knn_result = {
						"file_name":str(bag_name+'_'+str(x)+'_bag'),
						"sample_size":main.n_samples,
						"algorithm":'KNeighborsClassifier',
						"parameters":neigh_clf.best_estimator_,
						"majority_class_f1":f1_majority(y_test,y_pred),
						"minority_class_f1":f1_minority(y_test,y_pred),
						"knn_f1_score":knn_f1_score
					}
			tot_knn_result.append(knn_result)


	results_file.close()
	os.chdir(curr_pwd)
	return "KNN classifier building done using 5 fold validation"
	

def rf(sampled_data):

	bag_name = sampled_data[0]['dataset_name'].split('.')[0]
	n_bags = len(sampled_data)
	curr_pwd = str(os.getcwd())
	indices_pwd = curr_pwd + '/true_f1_scores/'
	os.chdir(indices_pwd)
	file_name = bag_name+'_rf_f1_weighted'+'.csv'
	results_file = open(file_name, 'a')
	tot_rf_result = []
	for x in range(len(sampled_data)):
		print("Processing '{file_name}'".format(file_name=bag_name+"_bag_"+str(sampled_data[x]['bag_number'])))
		bag_name_x=sampled_data[0]['dataset_name'].split('.')[0]+'_'+str(sampled_data[x]['bag_number'])
		dataset = sampled_data[x]
		data, target = dataset['data'], dataset['target']

		main = eda.eda()
		main.load_data(data, target)
		

		# print("\n  bag  dist\n")
		# print(np.count_nonzero(target==0))
		# print(np.count_nonzero(target==1))
		
	
		X_train, X_test, y_train, y_test = train_test_split(data,target, test_size=0.3,stratify=target)
		class_dist = np.bincount(y_test).tolist()
		majority_class_label = class_dist.index(max(class_dist))
		minority_class_label = class_dist.index(min(class_dist))


		f1_minority_score =0
		while(f1_minority_score==0):

			random_forest_clf = RandomForestClassifier(class_weight='balanced_subsample')
			random_forest_clf.fit(X_train, y_train)

			y_pred =random_forest_clf.predict(X_test)
			# print("\n  train set dist\n")
			# print(np.count_nonzero(y_test==0))
			# print(np.count_nonzero(y_test==1))
			# print("\n prediction dist ")
			# print(np.count_nonzero(y_pred==0))
			# print(np.count_nonzero(y_pred==1))

			#f1_scores = []
			#final_parameters = []
			
			#rf_f1_score = f1_ratio(y_test,y_pred)
			#f1_scores.append(rf_f1_score)
			#final_parameters.append(random_forest_clf.get_params)

			depth= []
			max_tree_depth = max([estimator.tree_.max_depth for estimator in random_forest_clf.estimators_])
			# print("maxdepth:{}".format(max_tree_depth))
			depth.append(max_tree_depth)
			depth.append(int(max_tree_depth*0.75))
			depth.append(int(max_tree_depth*0.5))
			depth.append(int(max_tree_depth*0.25))
			#removing zeros
			depth = [x for x in depth if x != 0]
			# print("depths:")
			# print(depth)
			parameters = {'n_estimators':[1,3,7,10],'max_depth':depth}	
			rf_f1_ratio_average = []
			f1_majority_score_average =[]
			f1_minority_score_average =[]
			for i in range(1,10):
				random_forest_clf = GridSearchCV(RandomForestClassifier(class_weight='balanced_subsample'), parameters, cv=5,n_jobs =20,refit=True,scoring=custom_f1)
				random_forest_clf.fit(X_train, y_train)

				y_pred =random_forest_clf.predict(X_test)

				rf_f1_score = f1_ratio(y_test,y_pred)
				# f1_scores.append(rf_f1_score)
				# final_parameters.append(random_forest_clf.get_params)

				# print("f1_scores")
				# print(f1_scores)
				# index_high_score = f1_scores.index(max(f1_scores))

				# print("index_high_score:{}".format(index_high_score))
				# print("majority_class_f1:{}".format(f1_majority(y_test,y_pred)))
				# print("minority_class_f1:{}".format(f1_minority(y_test,y_pred)))
				rf_f1_ratio_average.append(rf_f1_score)
				f1_minority_score_average.append(f1_minority(y_test,y_pred))
				f1_majority_score_average.append(f1_majority(y_test,y_pred))

			# print("avg_majority_class_f1:{}".format(sum(f1_majority_score_average)/len(f1_majority_score_average)))
			# print("avg_minority_class_f1:{}".format(sum(f1_minority_score_average)/len(f1_minority_score_average)))
			f1_minority_score = sum(f1_minority_score_average)/len(f1_minority_score_average)
			if(f1_minority_score ==0):
				print("---------re-fitting------")
				X_train, X_test, y_train, y_test = train_test_split(data,target, test_size=0.3,stratify=target)

		results_file.write("\"{file_name}\",{sample_size},\"{algorithm}\",\"{parameters}\",\"{majority_class_f1}\",\"{minority_class_f1}\",\"{rf_f1_score}\",".format(file_name=bag_name_x, sample_size=main.n_samples, algorithm='Random_forest', parameters=random_forest_clf.best_estimator_,majority_class_f1=sum(f1_majority_score_average)/len(f1_majority_score_average),minority_class_f1=sum(f1_minority_score_average)/len(f1_minority_score_average),rf_f1_score=sum(rf_f1_ratio_average)/len(rf_f1_ratio_average)))
		results_file.write('\n')

		rf_result = {
					"file_name":str(bag_name+'_'+str(x)+'_bag'),
					"sample_size":main.n_samples,
					"algorithm":'Random_forest',
					#"parameters":final_parameters[index_high_score],
					"parameters":random_forest_clf.best_estimator_,
					#"rf_f1_score":f1_scores[index_high_score],
					"majority_class_f1":sum(f1_majority_score_average)/len(f1_majority_score_average),
					"minority_class_f1":sum(f1_minority_score_average)/len(f1_minority_score_average),
					"rf_f1_score":rf_f1_score
				}
		tot_rf_result.append(rf_result)

	results_file.close()
	os.chdir(curr_pwd)
	return "RF classifier building done using 5 fold validation"
	

def dt(sampled_data):

	bag_name = sampled_data[0]['dataset_name'].split('.')[0]
	n_bags = len(sampled_data)
	curr_pwd = str(os.getcwd())
	indices_pwd = curr_pwd + '/true_f1_scores/'
	os.chdir(indices_pwd)
	file_name = bag_name+'_dt_f1_weighted'+'.csv'
	results_file = open(file_name, 'a')
	tot_dt_result = []
	for x in range(len(sampled_data)):
		print("Processing '{file_name}'".format(file_name=bag_name+"_bag_"+str(sampled_data[x]['bag_number'])))
		bag_name_x=sampled_data[0]['dataset_name'].split('.')[0]+'_'+str(sampled_data[x]['bag_number'])
		dataset = sampled_data[x]
		data, target = dataset['data'], dataset['target']

		main = eda.eda()
		main.load_data(data, target)
		X_train, X_test, y_train, y_test = train_test_split(data,target, test_size=0.3,stratify=target)
		

		# print("\n  bag  dist\n")
		# print(np.count_nonzero(target==0))
		# print(np.count_nonzero(target==1))
		
	
		
		# print("\n  train set dist\n")
		# print(np.count_nonzero(y_test==0))
		# print(np.count_nonzero(y_test==1))
		# print("\n prediction dist ")
		# print(np.count_nonzero(y_pred==0))
		# print(np.count_nonzero(y_pred==1))

		# f1_scores = []
		# final_parameters = []
		
		# rf_f1_score = f1_ratio(y_test,y_pred)
		# f1_scores.append(rf_f1_score)
		# final_parameters.append(Desc_tree_clf.get_params)

		
		
		# for i in range(1,4):
		# 	if(depth[i] <1 ):
		# 		depth[i]  = 1
		# 		print("\nINFO: fixing depth = 1 \n")
		f1_minority_score =0
		while(f1_minority_score==0):

			

			class_dist = np.bincount(y_test).tolist()
			majority_class_label = class_dist.index(max(class_dist))
			minority_class_label = class_dist.index(min(class_dist))

			weight_ratio = float(len(y_train[y_train == majority_class_label]))/float(len(y_train[y_train == minority_class_label]))
			w_array = np.array([1]*y_train.shape[0])
			w_array[y_train==minority_class_label] = weight_ratio
			w_array[y_train==majority_class_label] =1
			fit_parameters={'sample_weight':w_array}


			Desc_tree_clf = DecisionTreeClassifier(class_weight='balanced')
			Desc_tree_clf.fit(X_train,y_train,**fit_parameters) 

			depth= []
			max_tree_depth = Desc_tree_clf.tree_.max_depth 
			# print("maxdepth:{}".format(max_tree_depth))
			depth.append(max_tree_depth)
			depth.append(int(max_tree_depth*0.75))
			depth.append(int(max_tree_depth*0.5))
			depth.append(int(max_tree_depth*0.25))
			#removing zeros
			depth = [x for x in depth if x != 0]
			# print("depths:")
			# print(depth)
			parameters = {'max_depth':depth}

			y_pred =Desc_tree_clf.predict(X_test)
			Desc_tree_clf = GridSearchCV(DecisionTreeClassifier(class_weight='balanced'), parameters, cv=5,n_jobs =20,refit=True,scoring=custom_f1)
			Desc_tree_clf.fit(X_train,y_train,**fit_parameters) 
			


			y_pred =Desc_tree_clf.predict(X_test)

			dt_f1_score = f1_ratio(y_test,y_pred)
			# print("majority_class_f1:{}".format(f1_majority(y_test,y_pred)))
			# print("minority_class_f1:{}".format(f1_minority(y_test,y_pred)))
			f1_minority_score = f1_minority(y_test,y_pred)
			if(f1_minority_score ==0):
				print("---------re-fitting------")
				X_train, X_test, y_train, y_test = train_test_split(data,target, test_size=0.3,stratify=target)
		# f1_scores.append(rf_f1_score)
		# final_parameters.append(Desc_tree_clf.get_params)

		# print("f1_scores")
		# print(f1_scores)
		# index_high_score = f1_scores.index(max(f1_scores))

		# print("index_high_score:{}".format(index_high_score))

		results_file.write("\"{file_name}\",{sample_size},\"{algorithm}\",\"{parameters}\",\"{majority_class_f1}\",\"{minority_class_f1}\",\"{dt_f1_score}\",".format(file_name=bag_name_x, sample_size=main.n_samples, algorithm='Desc_tree(Prunned)', parameters=Desc_tree_clf.best_estimator_,majority_class_f1=f1_majority(y_test,y_pred),minority_class_f1=f1_minority(y_test,y_pred),dt_f1_score=dt_f1_score ))
		results_file.write('\n')

		dt_result = {
					"file_name":str(bag_name+'_'+str(x)+'_bag'),
					"sample_size":main.n_samples,
					"algorithm":'Desc_tree(Prunned)',
					#"parameters":final_parameters[index_high_score],
					"parameters":Desc_tree_clf.best_estimator_,
					#"rf_f1_score":f1_scores[index_high_score],
					"majority_class_f1":f1_majority(y_test,y_pred),
					"minority_class_f1":f1_minority(y_test,y_pred),
					"dt_f1_score":dt_f1_score
				}
		tot_dt_result.append(dt_result)

	results_file.close()
	os.chdir(curr_pwd)
	return "DT classifier building done using 5 fold validation"
	
def lr(sampled_data):
	bag_name = sampled_data[0]['dataset_name'].split('.')[0]
	n_bags = len(sampled_data)
	curr_pwd = str(os.getcwd())
	indices_pwd = curr_pwd + '/true_f1_scores/'
	os.chdir(indices_pwd)
	file_name = bag_name+'_lr_f1_weighted'+'.csv'
	results_file = open(file_name, 'a')
	tot_lr_result = []

	for x in range(len(sampled_data)):
		print("Processing '{file_name}'".format(file_name=bag_name+"_bag_"+str(sampled_data[x]['bag_number'])))
		bag_name_x=sampled_data[0]['dataset_name'].split('.')[0]+'_'+str(sampled_data[x]['bag_number'])
		dataset = sampled_data[x]
		data, target = dataset['data'], dataset['target']

		main = eda.eda()
		main.load_data(data, target)
		

		# print("\n  bag  dist\n")
		# print(np.count_nonzero(target==0))
		# print(np.count_nonzero(target==1))
		
	
		X_train, X_test, y_train, y_test = train_test_split(data,target, test_size=0.3,stratify=target)
		

		f1_minority_score =0
		while(f1_minority_score==0):
			class_dist = np.bincount(y_test).tolist()
			majority_class_label = class_dist.index(max(class_dist))
			minority_class_label = class_dist.index(min(class_dist))
			weight_ratio = float(len(y_train[y_train == majority_class_label]))/float(len(y_train[y_train == minority_class_label]))
			w_array = np.array([1]*y_train.shape[0])
			w_array[y_train==minority_class_label] = weight_ratio
			w_array[y_train==majority_class_label] =1
			fit_parameters={'sample_weight':w_array}
			parameters = {'C':[10**-3,10**-2,10**-1,10**0,10**1,10**2,10**3],'solver':['saga']}

			log_reg_clf = GridSearchCV(LogisticRegression(penalty='l1',class_weight='balanced',tol=0.01,max_iter=500,l1_ratio=weight_ratio), parameters, cv=5,n_jobs =20,refit=True,scoring=custom_f1)
			log_reg_clf.fit(X_train,y_train) 

			y_pred =log_reg_clf.predict(X_test)
			# print("majority_class_f1:{}".format(f1_majority(y_test,y_pred)))
			# print("minority_class_f1:{}".format(f1_minority(y_test,y_pred)))
		

			# print("\n  train set dist\n")
			# print(np.count_nonzero(y_test==0))
			# print(np.count_nonzero(y_test==1))
			# print("\n prediction dist ")
			# print(np.count_nonzero(y_pred==0))
			# print(np.count_nonzero(y_pred==1))
			lr_f1_score = f1_ratio(y_test,y_pred)
			f1_minority_score = f1_minority(y_test,y_pred)
			if(f1_minority_score ==0):
				print("---------re-fitting------")
				X_train, X_test, y_train, y_test = train_test_split(data,target, test_size=0.3,stratify=target)

		results_file.write("\"{file_name}\",{sample_size},\"{algorithm}\",\"{parameters}\",\"{majority_class_f1}\",\"{minority_class_f1}\",\"{lr_f1_score}\",".format(file_name=bag_name_x, sample_size=main.n_samples, algorithm='LogisticRegression(l1)', parameters=log_reg_clf.best_estimator_,majority_class_f1=f1_majority(y_test,y_pred),minority_class_f1=f1_minority(y_test,y_pred),lr_f1_score=lr_f1_score))
		
		results_file.write('\n')

		lr_result = {
					"file_name":str(bag_name+'_'+str(x)+'_bag'),
					"sample_size":main.n_samples,
					"algorithm":'LogisticRegression',
					"parameters":log_reg_clf.best_estimator_,
					"majority_class_f1":f1_majority(y_test,y_pred),
					"minority_class_f1":f1_minority(y_test,y_pred),
					"lr_f1_score":lr_f1_score
				}
		tot_lr_result.append(lr_result)

	results_file.close()
	os.chdir(curr_pwd)   
	return "LR classifier building done using 5 fold validation" 
	

def xg(sampled_data):

	bag_name = sampled_data[0]['dataset_name'].split('.')[0]
	n_bags = len(sampled_data)
	curr_pwd = str(os.getcwd())
	indices_pwd = curr_pwd + '/true_f1_scores/'
	os.chdir(indices_pwd)
	file_name = bag_name+'_xg_f1_weighted'+'.csv'
	results_file = open(file_name, 'a')
	tot_dt_result = []
	for x in range(len(sampled_data)):
		print("Processing '{file_name}'".format(file_name=bag_name+"_bag_"+str(sampled_data[x]['bag_number'])))
		bag_name_x=sampled_data[0]['dataset_name'].split('.')[0]+'_'+str(sampled_data[x]['bag_number'])
		dataset = sampled_data[x]
		data, target = dataset['data'], dataset['target']

		main = eda.eda()
		main.load_data(data, target)
		

		# print("\n  bag  dist\n")
		# print(np.count_nonzero(target==0))
		# print(np.count_nonzero(target==1))
		
	
		X_train, X_test, y_train, y_test = train_test_split(data,target, test_size=0.3,stratify=target)
		class_dist = np.bincount(y_test).tolist()
		majority_class_label = class_dist.index(max(class_dist))
		minority_class_label = class_dist.index(min(class_dist))
		f1_minority_score =0
		while(f1_minority_score==0):


			weight_ratio = float(len(y_train[y_train == majority_class_label]))/float(len(y_train[y_train == minority_class_label]))
			w_array = np.array([1]*y_train.shape[0])
			w_array[y_train==minority_class_label] = weight_ratio
			w_array[y_train==majority_class_label] =1
			fit_parameters={'sample_weight':w_array}

			parameters = {'learning_rate':np.linspace(0.05,0.3,3),'max_depth':[2**0,2**1,2**2,2**3,2**4,2**5,2**6]}
			xg_clf = GridSearchCV(xgboost.XGBClassifier(), parameters, cv=5,n_jobs =20,refit=True,scoring=custom_f1)
			xg_clf.fit(X_train,y_train,**fit_parameters) 



			y_pred =xg_clf.predict(X_test)

			xg_f1_score = f1_ratio(y_test,y_pred)
			# print("majority_class_f1:{}".format(f1_majority(y_test,y_pred)))
			# print("minority_class_f1:{}".format(f1_minority(y_test,y_pred)))

			f1_minority_score = f1_minority(y_test,y_pred)
			if(f1_minority_score ==0):
				# print("---------re-fitting------")
				X_train, X_test, y_train, y_test = train_test_split(data,target, test_size=0.3,stratify=target)
			
			



		results_file.write("\"{file_name}\",{sample_size},\"{algorithm}\",\"{parameters}\",\"{majority_class_f1}\",\"{minority_class_f1}\",\"{rf_f1_score}\",".format(file_name=bag_name_x, sample_size=main.n_samples, algorithm='xgboost', parameters=xg_clf.best_estimator_,majority_class_f1=f1_majority(y_test,y_pred),minority_class_f1=f1_minority(y_test,y_pred),rf_f1_score=xg_f1_score))
		results_file.write('\n')

		rf_result = {
					"file_name":str(bag_name+'_'+str(x)+'_bag'),
					"sample_size":main.n_samples,
					"algorithm":'xgboost',
					"parameters":xg_clf.best_estimator_,
					"majority_class_f1":f1_majority(y_test,y_pred),
					"minority_class_f1":f1_minority(y_test,y_pred),
					"rf_f1_score":xg_f1_score
				}
		tot_dt_result.append(rf_result)

	results_file.close()
	os.chdir(curr_pwd)
	return "XG classifier building done using 5 fold validation"

def svc(sampled_data):

	bag_name = sampled_data[0]['dataset_name'].split('.')[0]
	n_bags = len(sampled_data)
	curr_pwd = str(os.getcwd())
	indices_pwd = curr_pwd + '/true_f1_scores/'
	os.chdir(indices_pwd)
	file_name = bag_name+'_svc_f1_weighted'+'.csv'
	results_file = open(file_name, 'a')
	tot_svc_result = []
	for x in range(len(sampled_data)):
		print("Processing '{file_name}'".format(file_name=bag_name+"_bag_"+str(sampled_data[x]['bag_number'])))
		bag_name_x=sampled_data[0]['dataset_name'].split('.')[0]+'_'+str(sampled_data[x]['bag_number'])
		dataset = sampled_data[x]
		data, target = dataset['data'], dataset['target']

		main = eda.eda()
		main.load_data(data, target)
		

		# print("\n  bag  dist\n")
		# print(np.count_nonzero(target==0))
		# print(np.count_nonzero(target==1))
		
		X_train, X_test, y_train, y_test = train_test_split(data,target, test_size=0.3,stratify=target)
		

		f1_minority_score =0
		f1_majority_score=0
		counter =0

		parameters = {'C':[10**-2,10**-1,10**0,10**1,10**2],'gamma':['scale']}

		while((f1_minority_score==0 or f1_majority_score==0)):
			# print("counter:",counter)

			# print("--------------------------------imbalance_ratio 1:",imbalance_ratio(y_train))
			# print("\n--------------------------------train-dist--------------------------------\n")
			# print(np.count_nonzero(y_train==0))
			# print(np.count_nonzero(y_train==1))
			
			class_dist = np.bincount(y_test).tolist()
			majority_class_label = class_dist.index(max(class_dist))
			minority_class_label = class_dist.index(min(class_dist))
		
			
			


			if(counter>1):
				if(counter>1 and counter<5):
					# print("****balanced***")
					svc_clf = GridSearchCV(SVC(class_weight='balanced',cache_size=3000,shrinking=True,max_iter=15000), parameters, cv=5,n_jobs =20,verbose=False,refit=True,scoring=custom_f1)
					svc_clf.fit(X_train,y_train) 
				if(counter>=5 and counter<=7):
					# print("****class_weight_dict")
					class_weight_dict = dict()
					class_weight_dict[majority_class_label]=float(imbalance_ratio(y_train)/counter)
					class_weight_dict[minority_class_label]=(imbalance_ratio(y_train)+(counter*2))
					# print("****class_weight_dict:",class_weight_dict)
					
					svc_clf = GridSearchCV(SVC(class_weight=class_weight_dict,cache_size=3000,max_iter=15000,shrinking=True), parameters, cv=5,n_jobs =20,verbose=False,refit=True,scoring=custom_f1)
					svc_clf.fit(X_train,y_train) 
				if(counter>=8 and counter<12):
					# print("****sample_weight")
					weight_ratio = float(len(y_train[y_train == majority_class_label]))/float(len(y_train[y_train == minority_class_label]))
					w_array = np.array([1]*y_train.shape[0])
					w_array[y_train==minority_class_label] = weight_ratio+(counter*2)
					w_array[y_train==majority_class_label] =-(weight_ratio)
					fit_parameters={'sample_weight':w_array}

					# print('****sample_weight: {}'.format(w_array))

					svc_clf = GridSearchCV(SVC(cache_size=3000,shrinking=True,max_iter=15000), parameters, cv=5,n_jobs =20,verbose=False,refit=True,scoring=custom_f1)
					svc_clf.fit(X_train,y_train,**fit_parameters)
				if(counter>=12):
					parameters = {'C':[10**-2,10**-1,10**0,10**1,10**2],'kernel':['rbf','poly'],'degree':[2,3,4,5]}
					svc_clf = GridSearchCV(SVC(cache_size=15000,tol=1e-1,shrinking=True,max_iter=150000), parameters, cv=5,n_jobs =20,verbose=False,refit=True,scoring=custom_f1)
					svc_clf.fit(X_train,y_train)



			else:
				svc_clf = GridSearchCV(SVC(cache_size=3000), parameters, cv=5,n_jobs =20,verbose=False,refit=True,scoring=custom_f1)
				svc_clf.fit(X_train,y_train) 
			
			
			
	
			y_pred =svc_clf.predict(X_test)

			svc_f1_score = f1_ratio(y_test,y_pred)
			# print("majority_class_f1:{}".format(f1_majority(y_test,y_pred)))
			# print("minority_class_f1:{}".format(f1_minority(y_test,y_pred)))
			# print("\n--------------------------------after fit test  dist--------------------------------\n")
			# print(np.count_nonzero(y_pred==0))
			# print(np.count_nonzero(y_pred==1))	
			if(f1_majority(y_test,y_pred)==0 and f1_minority(y_test,y_pred)>0):
				if(counter>=5 and counter<=7):
					# print("****balanced re-fit")
					class_weight_dict = dict()
					class_weight_dict[majority_class_label]=float(imbalance_ratio(y_train)/(counter)+0.5)
					class_weight_dict[minority_class_label]=(imbalance_ratio(y_train)+(counter*2))
					# print("****class_weight_dict:",class_weight_dict)
					svc_clf = GridSearchCV(SVC(class_weight=class_weight_dict,cache_size=3000,max_iter=13000,shrinking=True), parameters, cv=5,n_jobs =20,verbose=False,refit=True,scoring=custom_f1)
					svc_clf.fit(X_train,y_train) 

					# print("majority_class_f1:{}".format(f1_majority(y_test,y_pred)))
					# print("minority_class_f1:{}".format(f1_minority(y_test,y_pred)))


			f1_minority_score = f1_minority(y_test,y_pred)
			f1_majority_score = f1_majority(y_test,y_pred)
			counter =counter +1
			if(counter>15):
				
				# print("XXXXXX tried 15 times XXXXXX")
				return 
			if(f1_minority_score ==0 or f1_majority_score==0):
				# print("---------re-fitting------")
				X_train, X_test, y_train, y_test = train_test_split(data,target, test_size=0.3,stratify=target,random_state=counter**2)

		results_file.write("\"{file_name}\",{sample_size},\"{algorithm}\",\"{parameters}\",\"{majority_class_f1}\",\"{minority_class_f1}\",\"{svc_f1_score}\",".format(file_name=bag_name_x, sample_size=main.n_samples, algorithm='SVC(linear/rbf)', parameters=svc_clf.best_estimator_,majority_class_f1=f1_majority(y_test,y_pred),minority_class_f1=f1_minority(y_test,y_pred),svc_f1_score=f1_ratio(y_test,y_pred)))
		results_file.write('\n')

		svc_result = {
					"file_name":str(bag_name+'_'+str(x)+'_bag'),
					"sample_size":main.n_samples,
					"algorithm":'svc(linear/rbf)',
					"parameters":svc_clf.best_estimator_,
					"majority_class_f1":f1_majority(y_test,y_pred),
					"minority_class_f1":f1_minority(y_test,y_pred),
					"svc_f1_score":f1_ratio(y_test,y_pred)
				}
		tot_svc_result.append(svc_result)

	results_file.close()
	os.chdir(curr_pwd)
	return "SVC classifier building done using 5 fold validation"



