import warnings



warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning) 






import os
import numpy as np
import pandas as pd
import gen_f1_score as gen_f1


def split_sample(file_name):
	#splitting datasets into test and train 
	import preprocess as sp
	train_df,test_df = sp.split(file_name)
	#(random sampling)bagging train data
	dataset_name,file_type = file_name.split('.')
	dataset_name_train = dataset_name + "_train.csv"
	#(random sampling)bagging test data
	dataset_name,file_type = file_name.split('.')
	dataset_name_test = dataset_name + "_test.csv"
	return dataset_name_train,dataset_name_test,train_df,test_df

def  random_sampling_file(file_name):

	import preprocess as random_sampling
	sample_bags=random_sampling.random_sampling(file_name)
	return sample_bags

def gen_indices(sample_bags):
	import gen_indices as GI
	
	
	#spectral     
	tot_cluster_result_spectral= GI.spectral_indices(sample_bags)
	
	
	#kmeans 
	tot_cluster_result_kmeans=GI.kmeans_indices(sample_bags)
	
	
 
	#hierarchial
	tot_cluster_result_heirar= GI.hierarchial_indices(sample_bags)
	
	
	

	
	#hdbscan
	tot_cluster_result_hdbscan= GI.hdbscan_indices(sample_bags)
	
	return 0



def grid_knn(sample_bags):
	
	result= gen_f1.knn(sample_bags)
	return result


def grid_rf(sample_bags):
	 
	result= gen_f1.rf(sample_bags)
	return result


def grid_dt(sample_bags):
	
	result= gen_f1.dt(sample_bags)
	return result


def grid_lr(sample_bags):
	
	result= gen_f1.lr(sample_bags)
	return result


def grid_svc(sample_bags):
	
	result= gen_f1.svc(sample_bags)
	return result

def grid_xg(sample_bags):
	
	result= gen_f1.xg(sample_bags)
	return result




import sys 
dataset_name= sys.argv[1]
print("running : {}".format(dataset_name))

curr_pwd=str(os.getcwd())

sample_bags=random_sampling_file(dataset_name)
    
os.chdir(curr_pwd)

tot_cluster_result = gen_indices(sample_bags)
    
os.chdir(curr_pwd)





#indices names
indices_list = ['WGSS', 'BGSS', 'Ball-Hall', 'Banfeld-Raftery', 'Calinski-Harabasz', 'Det-Ratio', 'Ksq-DetW', 'Log-Det-Ratio', 'Log-SS-Ratio', 'Silhouette', 'Trace-WiB', 'C', 'Dunn', 'Davies-Bouldin', 'Ray-Turi', 'PBM', 'Score', 'Entropy', 'Purity', 'Precision', 'Recall', 'F', 'Weighted-F', 'Folkes-Mallows', 'Rand', 'Adjusted-Rand', 'Adjusted-Mutual-Info', 'Normalised-Mutual-Info', 'Homegeneity', 'Completeness', 'V-Measure', 'Jaccard', 'Hubert Γ', 'Kulczynski', 'McNemar', 'Phi', 'Russel-Rao', 'Rogers-Tanimoto','Sokal-Sneath1','Sokal-Sneath2']

indices_list_mixed = []
indices_list_kmeans = []
indices_list_heirar = []
indices_list_spectral = []
indices_list_hdb = []


for i in indices_list:    
    indices_list_mixed.append(i+'_kmeans')
    indices_list_kmeans.append(i+'_kmeans')
    
for i in indices_list:
    indices_list_mixed.append(i+'_hierar')
    indices_list_heirar.append(i+'_hierar')    

for i in indices_list:
    indices_list_mixed.append(i+'_spectral')
    indices_list_spectral.append(i+'_spectral')
for i in indices_list:
    indices_list_mixed.append(i+'_hdbscan')
    indices_list_hdb.append(i+'_hdbscan')
    
    
os.chdir(curr_pwd+'/indices/')
# print(dataset_name.split('.')[0])
# curr_pwd = str(os.getcwd())
# indices_pwd = curr_pwd + '/indices/'
# os.chdir(indices_pwd)
file_kmeans_train  =  pd.read_csv(str(dataset_name.split('.')[0]+'kmeans_indices.csv'),header=None).drop(columns=[0,1,2,3,4,5,6,47],axis=1)
file_heirarchial_train  =  pd.read_csv(str(dataset_name.split('.')[0]+'heirarchical_indices.csv'),header=None).drop(columns=[0,1,2,3,4,5,6,47],axis=1)
file_spectral_train  =  pd.read_csv(str(dataset_name.split('.')[0]+'spectral_indices.csv'),header=None).drop(columns=[0,1,2,3,4,5,6,47],axis=1)
file_hdbscan_train  =  pd.read_csv(str(dataset_name.split('.')[0]+'hdbscan_indices.csv'),header=None).drop(columns=[0,1,2,3,4,5,6,47],axis=1)
file_train_indices = pd.concat([file_kmeans_train,file_heirarchial_train,file_spectral_train,file_hdbscan_train],axis=1)


file_train_indices.columns = indices_list_mixed



rouge_indices =['WGSS_kmeans',
 'BGSS_kmeans',
 'Ball-Hall_kmeans',
 'Banfeld-Raftery_kmeans',
 'Calinski-Harabasz_kmeans',
 'Det-Ratio_kmeans',
 'Ksq-DetW_kmeans',
 'Log-Det-Ratio_kmeans',
 'Log-SS-Ratio_kmeans',
 'Trace-WiB_kmeans',
 'Davies-Bouldin_kmeans',
 'Ray-Turi_kmeans',
 'PBM_kmeans',
 'McNemar_kmeans',
 'WGSS_hierar',
 'BGSS_hierar',
 'Ball-Hall_hierar',
 'Banfeld-Raftery_hierar',
 'Calinski-Harabasz_hierar',
 'Det-Ratio_hierar',
 'Ksq-DetW_hierar',
 'Log-Det-Ratio_hierar',
 'Log-SS-Ratio_hierar',
 'Trace-WiB_hierar',
 'Davies-Bouldin_hierar',
 'Ray-Turi_hierar',
 'PBM_hierar',
 'McNemar_hierar',
 'WGSS_spectral',
 'BGSS_spectral',
 'Ball-Hall_spectral',
 'Banfeld-Raftery_spectral',
 'Calinski-Harabasz_spectral',
 'Det-Ratio_spectral',
 'Ksq-DetW_spectral',
 'Log-Det-Ratio_spectral',
 'Log-SS-Ratio_spectral',
 'Trace-WiB_spectral',
 'PBM_spectral',
 'McNemar_spectral',
 'WGSS_hdbscan',
 'BGSS_hdbscan',
 'Ball-Hall_hdbscan',
 'Banfeld-Raftery_hdbscan',
 'Calinski-Harabasz_hdbscan',
 'Det-Ratio_hdbscan',
 'Ksq-DetW_hdbscan',
 'Log-Det-Ratio_hdbscan',
 'Trace-WiB_hdbscan',
 'C_hdbscan',
 'Davies-Bouldin_hdbscan',
 'Ray-Turi_hdbscan',
 'PBM_hdbscan',
 'McNemar_hdbscan']

file_train_indices_rouge_dropped = file_train_indices.drop(columns = rouge_indices)

# print(file_train_indices_rouge_dropped.shape)


#loading models
import pickle 
    
os.chdir(curr_pwd)


os.chdir(curr_pwd+'/model/')
with open('full_train_full_dataset_xg_reg_dt.pkl', 'rb') as f:
    reg_dt = pickle.load(f)
with open('full_train_full_dataset_xg_reg_rf.pkl', 'rb') as f:
    reg_rf = pickle.load(f)
with open('full_train_full_dataset_xg_reg_lr.pkl', 'rb') as f:
    reg_lr = pickle.load(f)
with open('full_train_full_dataset_xg_reg_knn.pkl', 'rb') as f:
    reg_knn = pickle.load(f)
with open('full_train_full_dataset_xg_reg_xg.pkl', 'rb') as f:
    reg_xg = pickle.load(f)
with open('full_train_full_dataset_xg_reg_svc.pkl', 'rb') as f:
    reg_svc = pickle.load(f)
os.chdir(curr_pwd)

predicted_dt_score = reg_dt.best_estimator_.predict(file_train_indices_rouge_dropped)
predicted_rf_score = reg_rf.best_estimator_.predict(file_train_indices_rouge_dropped)
predicted_lr_score = reg_lr.best_estimator_.predict(file_train_indices_rouge_dropped)
predicted_knn_score = reg_knn.best_estimator_.predict(file_train_indices_rouge_dropped)
predicted_xg_score = reg_xg.best_estimator_.predict(file_train_indices_rouge_dropped)
predicted_svc_score = reg_svc.best_estimator_.predict(file_train_indices_rouge_dropped)


predicted_df   = pd.DataFrame(predicted_dt_score)
predicted_df.columns = ['pred_dt']
predicted_df['pred_rf']  = predicted_rf_score
predicted_df['pred_lr']  = predicted_lr_score
predicted_df['pred_knn']  = predicted_knn_score
predicted_df['pred_xg']  = predicted_xg_score
predicted_df['pred_svc']  = predicted_svc_score

predicted_df.to_csv(dataset_name+"_predicted_f1_scores.csv",index=None)


mean_pred_dt = np.mean(predicted_dt_score)
mean_pred_rf = np.mean(predicted_rf_score)
mean_pred_lr = np.mean(predicted_lr_score)
mean_pred_knn = np.mean(predicted_knn_score)
mean_pred_xg = np.mean(predicted_xg_score)
mean_pred_svc = np.mean(predicted_svc_score)

predicted_dict = {'dt':mean_pred_dt,
                  'rf':mean_pred_rf,
                  'lr':mean_pred_lr,
                  'knn':mean_pred_knn,
                  'xg':mean_pred_xg,
                  'svc':mean_pred_svc
                 }


generate_true_f1 = input("Do you want to generate True F1 (Best score after 5-fold cross-validation is displayed) scores(Y/N):")

if(generate_true_f1 == 'Y' or generate_true_f1 == 'y' ):
	# calculating ground truth F1 score  
	'''Uncomment to generate ground truth '''

	print("----------------------KNN-----------------")
	print(grid_knn(sample_bags))
	print("----------------------RF-----------------")
	print(grid_rf(sample_bags))
	print("----------------------DT-----------------")
	print(grid_dt(sample_bags))
	print("----------------------LR-----------------")
	print(grid_lr(sample_bags))
	print("----------------------SVC-----------------")
	print(grid_svc(sample_bags))
	print("----------------------XG-----------------")
	print(grid_xg(sample_bags))
	print(" Ground truth F1  generate [output written in 'true_f1_scores' folder]")

	#reading ground truth F1 score 

	df = pd.read_csv(curr_pwd + '/true_f1_scores/' + str(dataset_name.split('.')[0]) + '_dt_f1_weighted.csv' ,header=None,index_col=None)
	mean_true_dt = df.iloc[:,-2].mean()

	df = pd.read_csv(curr_pwd + '/true_f1_scores/' + str(dataset_name.split('.')[0]) + '_rf_f1_weighted.csv',header=None,index_col=None )
	mean_true_rf = df.iloc[:,-2].mean()

	df = pd.read_csv(curr_pwd + '/true_f1_scores/' + str(dataset_name.split('.')[0]) + '_lr_f1_weighted.csv' ,header=None,index_col=None)
	mean_true_lr = df.iloc[:,-2].mean()

	df = pd.read_csv(curr_pwd + '/true_f1_scores/' + str(dataset_name.split('.')[0]) + '_knn_f1_weighted.csv' ,header=None,index_col=None)
	mean_true_knn = df.iloc[:,-2].mean()

	df = pd.read_csv(curr_pwd + '/true_f1_scores/' + str(dataset_name.split('.')[0]) + '_xg_f1_weighted.csv' ,header=None,index_col=None)
	mean_true_xg = df.iloc[:,-2].mean()

	df = pd.read_csv(curr_pwd + '/true_f1_scores/' + str(dataset_name.split('.')[0]) + '_svc_f1_weighted.csv',header=None,index_col=None )
	mean_true_svc = df.iloc[:,-2].mean()
	true_f1_dict = {'dt':mean_true_dt,
	'rf':mean_true_rf,
	'lr':mean_true_lr,
	'knn':mean_true_knn,
	'xg':mean_true_xg,
	'svc':mean_true_svc
	}


	if(len(sample_bags)==1):

		print("---------------------------RESULT------------------------------")

		print("Predicted F1 scores: ",sorted(predicted_dict.items(), key=lambda x: x[1], reverse=True))           


		
		print("True F1 scores: ",sorted(true_f1_dict.items(), key=lambda x: x[1], reverse=True))           

		print("---------------------------END------------------------------")
		exit()
	else:

		print("Building Classifiers over the entire dataset")
		import eda as eda
		main =eda.eda()
		seperator = ','
		skiplines = None
		headerline = 0
		usecolumns = None
		targetcol = -1
		encode_target = True
		nominal_cols='infer'
		na_values = None

		curr_pwd = str(os.getcwd())
		data_pwd = curr_pwd + str('/data/')
		main.read_data_csv(data_pwd+dataset_name, sep=seperator, skiprows=skiplines, header_row=headerline, usecols=usecolumns, target_col=targetcol, encode_target=encode_target, categorical_cols=nominal_cols, na_values=na_values, nrows=None)
		nominal_columns = 'infer'
		main.dummy_coding(nominal_columns=nominal_columns)
		sample_bags =[]
		sampled_data = dict.fromkeys(['data', 'target','classes','n_samples','n_features','column_names','column_categories','dataset_name','bag_number'])
		sampled_data['data'], sampled_data['target'] = main.data, main.target	
		sampled_data['classes']=eda.label_cnt_dict(main.target) if main.target is not None else None
		sampled_data['n_samples']=main.n_samples # Not inferrable from classes, if target=None
		sampled_data['n_features']=main.n_features,
		sampled_data['column_names']=main.columns_,
		sampled_data['column_categories']=main.columns_categories_ if hasattr(main, 'columns_categories_') else None	
		sampled_data['dataset_name']= dataset_name.split('.')[0]+'_full.csv'
		sampled_data['bag_number']= 'full'


		sample_bags.append(sampled_data) 
		print("----------------------KNN-----------------")
		print(grid_knn(sample_bags))
		print("----------------------RF-----------------")
		print(grid_rf(sample_bags))
		print("----------------------DT-----------------")
		print(grid_dt(sample_bags))
		print("----------------------LR-----------------")
		print(grid_lr(sample_bags))
		print("----------------------SVC-----------------")
		print(grid_svc(sample_bags))
		print("----------------------XG-----------------")
		print(grid_xg(sample_bags))

		df = pd.read_csv(curr_pwd + '/true_f1_scores/' + str(sample_bags[0]['dataset_name'].split('.')[0]) + '_dt_f1_weighted.csv' ,header=None,index_col=None)
		mean_true_dt_full = df.iloc[:,-2].mean()

		df = pd.read_csv(curr_pwd + '/true_f1_scores/' + str(sample_bags[0]['dataset_name'].split('.')[0]) + '_rf_f1_weighted.csv',header=None,index_col=None )
		mean_true_rf_full = df.iloc[:,-2].mean()

		df = pd.read_csv(curr_pwd + '/true_f1_scores/' + str(sample_bags[0]['dataset_name'].split('.')[0]) + '_lr_f1_weighted.csv' ,header=None,index_col=None)
		mean_true_lr_full = df.iloc[:,-2].mean()

		df = pd.read_csv(curr_pwd + '/true_f1_scores/' + str(sample_bags[0]['dataset_name'].split('.')[0]) + '_knn_f1_weighted.csv' ,header=None,index_col=None)
		mean_true_knn_full = df.iloc[:,-2].mean()

		df = pd.read_csv(curr_pwd + '/true_f1_scores/' + str(sample_bags[0]['dataset_name'].split('.')[0]) + '_xg_f1_weighted.csv' ,header=None,index_col=None)
		mean_true_xg_full = df.iloc[:,-2].mean()

		df = pd.read_csv(curr_pwd + '/true_f1_scores/' + str(sample_bags[0]['dataset_name'].split('.')[0]) + '_svc_f1_weighted.csv',header=None,index_col=None )
		mean_true_svc_full = df.iloc[:,-2].mean()

		true_f1_dict_full = {'dt':mean_true_dt_full,
		'rf':mean_true_rf_full,
		'lr':mean_true_lr_full,
		'knn':mean_true_knn_full,
		'xg':mean_true_xg_full,
		'svc':mean_true_svc_full
		}

		print("---------------------------RESULT------------------------------")

		print("Predicted F1 scores: ",sorted(predicted_dict.items(), key=lambda x: x[1], reverse=True))           


		
		print("True F1 scores(built over bags): ",sorted(true_f1_dict.items(), key=lambda x: x[1], reverse=True)) 


		print("True F1 scores(built over entire dataset): ",sorted(true_f1_dict_full.items(), key=lambda x: x[1], reverse=True))           

		print("---------------------------END------------------------------")
		exit()





else:




	print("---------------------------RESULT------------------------------")

	print("Predicted F1 scores: ",sorted(predicted_dict.items(), key=lambda x: x[1], reverse=True))           

	print("Top classifier:",max(predicted_dict, key=predicted_dict.get))
	print("---------------------------END------------------------------")

	exit()
