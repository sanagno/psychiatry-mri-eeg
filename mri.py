import itertools

import numpy as np
import pandas as pd
# import seaborn as sn
# import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix 
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_fscore_support

# Fix random seed 
np.random.seed(13)


metrics = ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1_score']

def compute_scores(y_true, y_pred): 
    d = {}
    
    for metric in metrics: 
        if metric == 'accuracy': 
            d[metric] = accuracy_score(y_true, y_pred)
        elif metric == 'balanced_accuracy': 
            d[metric] = balanced_accuracy_score(y_true, y_pred)
        elif metric == 'precision': 
            d[metric] = precision_score(y_true, y_pred)
        elif metric == 'recall': 
            d[metric] = recall_score(y_true, y_pred)
        elif metric == 'f1_score': 
            d[metric] = f1_score(y_true, y_pred)
        else: 
            raise ValueError('You provided a non-supported evaluation metric.')
            
    return d

def binary_classification(features, labels, test_splits, disorders, scalers, estimator, 
                          param_grid, scale_features, scenario, train_mode, verbose=True):
    
    # Initialize results 
    results = {}
    predictions = {}

    for disorder in disorders: 
        if disorder == 'No Diagnosis Given': 
            continue

        if scenario == 'Disorder vs No Diagnosis':
            # Preserve subjects that belong to either class
            mask = (labels[disorder] == 1) | (labels['No Diagnosis Given'] == 1)
            
            X = features[mask.values]
            Y = labels[mask.values]
            
        elif scenario == 'Disorder vs Rest':
            # Preserve all subjects
            X = features
            Y = labels
        else: 
            raise ValueError('Incorrect scenario value!')

        if train_mode == 'Intersection': 
            datapoints = []
            for split in test_splits: 
                datapoints = datapoints + split
            # Binary labels for current disorder 
            y_true = Y[disorder].loc[datapoints]
        elif train_mode == 'Everything': 
            datapoints = X.index.values.tolist()
            # Binary labels for current disorder 
            y_true = Y[disorder].loc[datapoints]
        else: 
            raise ValueError('Incorrect training mode provided!')
            
        # Initialize predictions
        y_pred = pd.Series(data=np.empty(len(datapoints)),index=datapoints, dtype=int)
        
        for test_split in test_splits:
            # Train split
            train_split = list(set(datapoints) - set(test_split))
            
            if bool(set(train_split) & set(test_split)):
                raise ValueError('Training and Testing splits overlap!')
            
            X_train, X_test = X.loc[train_split].values, X.loc[test_split].values
            y_train, y_test = y_true.loc[train_split].values.astype(int), y_true.loc[test_split].values.astype(int)

            # Pre-processing 
            if scale_features:
                scaler = scalers[disorder]
                Z_train = scaler.fit_transform(X_train)
            else:
                Z_train = X_train

            N = len(y_train)
            N_0 = np.sum(y_train == 0)
            N_1 = np.sum(y_train == 1)
            
            param_grid['class_weight'] = [{0: N/N_0, 1: (N_0/N_1)*N/N_1}]
            
            # Hyper-parameter tuning 
            _classifier = estimator

            search = GridSearchCV(
                        estimator=_classifier,
                        param_grid=param_grid,
                        scoring='f1',
                        n_jobs=-1,
                        iid=False,
                        cv=StratifiedKFold(n_splits=5))

            search.fit(Z_train, y_train)

            best_params = search.best_params_

            # Train on the "best" configuration 
            classifier = estimator
            classifier.set_params(**best_params)
            classifier.fit(Z_train, y_train)

            # Testing 
            if scale_features: 
                Z_test = scaler.transform(X_test)
            else: 
                Z_test = X_test 
                
            y_pred.loc[test_split] = classifier.predict(Z_test)

        # Results 
        results[disorder] = compute_scores(y_true.values.astype(int), y_pred.values.astype(int))
        predictions[disorder] = y_pred.values.astype(int)

        if verbose: 
            print('================================= {0} ================================='.format(disorder))

            print('accuracy {:.3f} balanced_accuracy {:.3f} precision {:.3f} recall {:.3f} f1_score {:.3f}'                       .format(results[disorder]['accuracy'], results[disorder]['balanced_accuracy'], 
                              results[disorder]['precision'], results[disorder]['recall'], results[disorder]['f1_score']))
            
    return results, predictions


# # Behavioural Data 

# In[47]:


behaviour_data = pd.read_csv('data/Behavioral/cleaned/HBNFinalSummaries.csv', low_memory=False)

# Drop patients with incomplete diagnosis
initial_size = behaviour_data.shape[0]
behaviour_data = behaviour_data[behaviour_data['NoDX'].isin(['Yes', 'No'])]
new_size = behaviour_data.shape[0]
print('Removing', initial_size - new_size, 'patients as their evaluation was incomplete.')

most_common_disorders = ['Attention-Deficit/Hyperactivity Disorder', 'Anxiety Disorders', 'Specific Learning Disorder',
                         'Autism Spectrum Disorder', 'Disruptive', 'No Diagnosis Given', 'Communication Disorder',
                         'Depressive Disorders']

category_columns = ['DX_' + str(i).zfill(2) + '_Cat' for i in range(1, 11)] +                   ['DX_' + str(i).zfill(2) + '_Sub' for i in range(1, 11)]

# find users that have no diagnosis within these top diseases
# filtering should cahnge anything as this should also happen at a later stage
mask = None
for col in category_columns:
    mask_col = behaviour_data[col].isin(most_common_disorders)
    if mask is None:
        mask = mask_col
    else:
        mask = mask | mask_col

initial_size = behaviour_data.shape[0]
behaviour_data = behaviour_data[mask]
behaviour_data = behaviour_data.reset_index(drop=True)
new_size = behaviour_data.shape[0]
behaviour_data.rename(columns={'EID': 'ID'}, inplace=True) 

print('Removing', initial_size - new_size, 'patients as their diagnoses were very uncommon.')


classes = np.zeros((len(most_common_disorders), behaviour_data.shape[0]), dtype=np.int32)

df_disorders = behaviour_data[category_columns]

for i, disorder in enumerate(most_common_disorders):
    mask = df_disorders.select_dtypes(include=[object]).applymap(lambda x: disorder in x if pd.notnull(x) else False)
    
    disorder_df = df_disorders[mask.any(axis=1)]
    
    np.add.at(classes[i], disorder_df.index.values, 1)

classes = np.transpose(classes)
classes = np.column_stack((behaviour_data['ID'], classes))

labels = pd.DataFrame(data=classes, columns=['ID'] + most_common_disorders)

# # Disorder vs No Diagnosis Given

# ## FA Per Tract


fa_per_tract = pd.read_csv('data/MRI/MRI/DTI/FAPerTract.csv', low_memory=False)

# Remove "/" from the end some IDs 
fa_per_tract['ID'] = fa_per_tract['ID'].apply(lambda x: x[:-1] if "/" in x else x)

# Keep patients who have both behavioural records and MRI
fa_patient_IDs = np.array(list(set(behaviour_data['ID'].values) & set(fa_per_tract['ID'].values)))
fa_per_tract = fa_per_tract[fa_per_tract['ID'].isin(fa_patient_IDs)].sort_values(by='ID')
labels_fa_tr = labels[labels['ID'].isin(fa_patient_IDs)].sort_values(by='ID')

print('Number of patients: ', len(fa_patient_IDs))


# In[66]:


# Fill in missing values in MRI data
median_mri = fa_per_tract.median(axis=0, skipna=True)

fa_per_tract = fa_per_tract.fillna(value=median_mri, axis=0)

# Drop ID column
# fa_per_tract = fa_per_tract.drop(columns=['ID'])

# Convert site to integer ID
scan_sites = fa_per_tract['ScanSite'].unique()
map_site_to_ID = {site: i for i, site in enumerate(scan_sites)}
fa_per_tract['ScanSite'] = fa_per_tract['ScanSite'].apply(lambda x: map_site_to_ID[x])

# Add Sex and Age to features 
fa_per_tract = pd.merge(
                fa_per_tract, 
                behaviour_data[behaviour_data['ID'].isin(fa_patient_IDs)].sort_values(by='ID')[['ID','Sex', 'Age']],
                on='ID',
                how='inner')

fa_per_tract.set_index('ID', inplace=True)
labels_fa_tr.set_index('ID', inplace=True)


# ### Random Forest 

# ## Structural MRI

cort_thick_l = pd.read_csv('data/MRI/MRI/structuralMRI/CorticalThicknessLHROI.csv', low_memory=False)
cort_thick_r = pd.read_csv('data/MRI/MRI/structuralMRI/CorticalThicknessRHROI.csv', low_memory=False)
cort_vol_l = pd.read_csv('data/MRI/MRI/structuralMRI/CorticalVolumeLHROI.csv', low_memory=False)
cort_vol_r = pd.read_csv('data/MRI/MRI/structuralMRI/CorticalVolumeRHROI.csv', low_memory=False)
sub_cort_vol_l = pd.read_csv('data/MRI/MRI/structuralMRI/SubCorticalVolumeLHROI.csv', low_memory=False)
sub_cort_vol_r = pd.read_csv('data/MRI/MRI/structuralMRI/SubCorticalVolumeRHROI.csv', low_memory=False)
glob_thick = pd.read_csv('data/MRI/MRI/structuralMRI/GlobalCorticalThickness.csv', low_memory=False)

# Drop duplicated columns 
cort_thick_r = cort_thick_r.drop(columns=['eTIV', 'ScanSite'])
cort_vol_l = cort_vol_l.drop(columns=['eTIV', 'ScanSite'])
cort_vol_r = cort_vol_r.drop(columns=['eTIV', 'ScanSite'])
sub_cort_vol_l = sub_cort_vol_l.drop(columns=['eTIV', 'ScanSite'])
sub_cort_vol_r = sub_cort_vol_r.drop(columns=['eTIV', 'ScanSite'])
glob_thick = glob_thick.drop(columns=['ScanSite'])

# Join tables 
struct_mri = pd.merge(cort_thick_l, cort_thick_r, on='ID', how='inner')
struct_mri = pd.merge(struct_mri, cort_vol_l, on='ID', how='inner')
struct_mri = pd.merge(struct_mri, cort_vol_r, on='ID', how='inner')
struct_mri = pd.merge(struct_mri, sub_cort_vol_l, on='ID', how='inner')
struct_mri = pd.merge(struct_mri, sub_cort_vol_r, on='ID', how='inner')
struct_mri = pd.merge(struct_mri, glob_thick, on='ID', how='inner')

struct_mri.head()


# In[70]:


# Remove "/" from the end some IDs 
struct_mri['ID'] = struct_mri['ID'].apply(lambda x: x[:-1] if "/" in x else x)

# Keep patients who have both behavioural records and MRI
struct_patient_IDs = np.array(list(set(behaviour_data['ID'].values) & set(struct_mri['ID'].values)))
struct_mri = struct_mri[struct_mri['ID'].isin(struct_patient_IDs)].sort_values(by='ID')
labels_str = labels[labels['ID'].isin(struct_patient_IDs)].sort_values(by='ID')

print('Number of patients: ', len(struct_patient_IDs))

# Add Sex and Age to features 
struct_mri = pd.merge(
                struct_mri, 
                behaviour_data[behaviour_data['ID'].isin(struct_patient_IDs)].sort_values(by='ID')[['ID','Sex', 'Age']],
                on='ID',
                how='inner')


# Check that IDs from the features and labels tables match
if any(struct_mri['ID'].values != labels_str['ID'].values):
    raise ValueError('There is a mismatch in IDs!')
    
struct_mri.set_index('ID', inplace=True)
labels_str.set_index('ID', inplace=True)


# In[74]:


# Drop ID column
# struct_mri = struct_mri.drop(columns=['EID'])

# Convert site to integer ID
scan_sites = struct_mri['ScanSite'].unique()
map_site_to_ID = {site: i for i, site in enumerate(scan_sites)}
struct_mri['ScanSite'] = struct_mri['ScanSite'].apply(lambda x: map_site_to_ID[x])


import pickle

# load it again
with open('cross_validation_splits.pkl', 'rb') as fid:
    cross_validation_splits = pickle.load(fid)
    
N_rep, N_folds = cross_validation_splits.shape


# ## FA Per Track 

# In[22]:


fa_per_tract_res = {}


# ### Random Forest 

# In[ ]:


rf_param_grid = {'n_estimators': [100], 
                 'max_depth': np.linspace(start=4,stop=20,num=5),
                 'max_features': np.linspace(start=0.2,stop=1,num=5),
                 'n_jobs': [-1], 
                 'class_weight': ['balanced']}

scalers = {disorder: RobustScaler() for disorder in most_common_disorders}

results = []

for i in range(N_rep): 

    rf_tract_res_vs_rest, rf_tract_y_pred_vs_rest = binary_classification(fa_per_tract, 
                                                                          labels_fa_tr, 
                                                                          cross_validation_splits[i],
                                                                          most_common_disorders, 
                                                                          scalers,
                                                                          RandomForestClassifier(), 
                                                                          rf_param_grid,
                                                                          scale_features=False,
                                                                          scenario='Disorder vs Rest',
                                                                          train_mode='Intersection')
    
    results.append(rf_tract_res_vs_rest)
    
fa_per_tract_res['rf'] = results


# ### SVM

# In[ ]:


# svm_param_grid = {'C': np.logspace(-10, 10, 5), 
#                   'kernel': ['rbf'],
#                   'gamma': ['scale', 'auto'] + np.logspace(-2, 0, 5).tolist(),
#                   'class_weight': ['balanced']}

svm_param_grid = {'C': [1], 
                  'kernel': ['rbf'],
                  'gamma': ['scale'],
                  'class_weight': ['balanced']}

scalers = {disorder: RobustScaler() for disorder in most_common_disorders}

results = []

for i in range(N_rep): 

    svm_tract_res_vs_rest, svm_tract_y_pred_vs_rest = binary_classification(fa_per_tract, 
                                                                            labels_fa_tr, 
                                                                            cross_validation_splits[i],
                                                                            most_common_disorders, 
                                                                            scalers,
                                                                            SVC(), 
                                                                            svm_param_grid,
                                                                            scale_features=True,
                                                                            scenario='Disorder vs Rest',
                                                                            train_mode='Intersection')
    
    results.append(svm_tract_res_vs_rest)
    
fa_per_tract_res['svm'] = results


# In[ ]:


with open('fa_per_tract_res_inter.pickle', 'wb') as f:
    pickle.dump(fa_per_tract_res, f)


# ### Results

# In[ ]:


# fig, axs = plt.subplots(nrows= 2, ncols=4, figsize=(35,15))

# for ax, disorder in zip(axs.flat, rf_tract_res_vs_rest.keys()):
#     metrics = [m for m in rf_tract_res_vs_rest[disorder].keys()]
#     rf_scores = [v for v in rf_tract_res_vs_rest[disorder].values()]
#     svm_scores = [v for v in svm_tract_res_vs_rest[disorder].values()]
    
#     x = np.arange(len(metrics)) # label locations 
#     width = 0.4 # the width of the bars
    
#     rects_rf = ax.bar(x + width/2, rf_scores, width, label='RF')
#     rects_svm = ax.bar(x - width/2, svm_scores, width, label='SVM')
                             
#     # Add some text for labels, title and custom x-axis tick labels, etc.
#     ax.set_ylabel('Score')
#     ax.set_title(disorder)
#     ax.set_xticks(x)
#     ax.set_xticklabels(metrics)
#     ax.grid(True)
#     ax.legend()
                             
# plt.show()


# ## Structural MRI 

# In[ ]:


struct_mri_res = {}


# ### Random Forest

# In[ ]:


# rf_param_grid = {'n_estimators': [100], 
#                  'max_depth': np.linspace(start=4,stop=20,num=5),
#                  'max_features': np.linspace(start=0.2,stop=1,num=5),
#                  'n_jobs': [-1], 
#                  'class_weight': ['balanced']}

rf_param_grid = {'n_estimators': [100], 
                 'max_depth': [16],
                 'max_features': ["auto"],
                 'n_jobs': [-1], 
                 'class_weight': ['balanced']}

scalers = {disorder: RobustScaler() for disorder in most_common_disorders}


results = []

for i in range(N_rep): 

    rf_struct_res_vs_rest, rf_struct_y_pred_vs_rest = binary_classification(struct_mri, 
                                                                            labels_str, 
                                                                            cross_validation_splits[i],
                                                                            most_common_disorders, 
                                                                            scalers,
                                                                            RandomForestClassifier(), 
                                                                            rf_param_grid,
                                                                            scale_features=False,
                                                                            scenario='Disorder vs Rest',
                                                                            train_mode='Intersection')
    
    results.append(rf_struct_res_vs_rest)

struct_mri_res['rf'] = results


# ### SVM 

# In[ ]:


# svm_param_grid = {'C': np.logspace(-10, 10, 5), 
#                   'kernel': ['rbf'],
#                   'gamma': ['scale', 'auto'] + np.logspace(-2, 0, 5).tolist(),
#                   'class_weight': ['balanced']}

svm_param_grid = {'C': [1], 
                  'kernel': ['rbf'],
                  'gamma': ['scale'],
                  'class_weight': ['balanced']}

scalers = {}

for disorder in most_common_disorders: 
    if disorder == 'Attention-Deficit/Hyperactivity Disorder': 
        scalers[disorder] = RobustScaler()
    else: 
        scalers[disorder] = MinMaxScaler()
        
results = []

for i in range(N_rep): 

    svm_struct_res_vs_rest, svm_struct_y_pred_vs_rest = binary_classification(struct_mri, 
                                                                              labels_str, 
                                                                              cross_validation_splits[i],
                                                                              most_common_disorders, 
                                                                              scalers,
                                                                              SVC(), 
                                                                              svm_param_grid,
                                                                              scale_features=True,
                                                                              scenario='Disorder vs Rest',
                                                                              train_mode='Intersection')
    
    results.append(svm_struct_res_vs_rest)

struct_mri_res['svm'] = results


# In[ ]:


with open('struct_mri_res_inter.pickle', 'wb') as f:
    pickle.dump(struct_mri_res, f)


# ### Results

# In[ ]:


# fig, axs = plt.subplots(nrows= 2, ncols=4, figsize=(35,15))

# for ax, disorder in zip(axs.flat, rf_struct_res_vs_rest.keys()):
#     metrics = [m for m in rf_struct_res_vs_rest[disorder].keys()]
#     rf_scores = [v for v in rf_struct_res_vs_rest[disorder].values()]
#     svm_scores = [v for v in svm_struct_res_vs_rest[disorder].values()]
    
#     x = np.arange(len(metrics)) # label locations 
#     width = 0.4 # the width of the bars
    
#     rects_rf = ax.bar(x + width/2, rf_scores, width, label='RF')
#     rects_svm = ax.bar(x - width/2, svm_scores, width, label='SVM')
                             
#     # Add some text for labels, title and custom x-axis tick labels, etc.
#     ax.set_ylabel('Score')
#     ax.set_title(disorder)
#     ax.set_xticks(x)
#     ax.set_xticklabels(metrics)
#     ax.grid(True)
#     ax.legend()
                             
# plt.show()


# ## Combined MRI 

# In[80]:


combined_mri_patient_IDs = list(set(fa_per_tract.index.values.tolist()) & set(struct_mri.index.values.tolist()))
print('Number of patients having both MRI records: ', len(combined_mri_patient_IDs))


# In[118]:


struct_mri.reset_index(level=0, inplace=True)
labels_str.reset_index(level=0, inplace=True)
fa_per_tract.reset_index(level=0, inplace=True)
labels_fa_tr.reset_index(level=0, inplace=True)

# Consider only intersection 

# Structural MRI 
struct_mri_reduced = struct_mri[struct_mri['ID'].isin(combined_mri_patient_IDs)].sort_values(by='ID')
labels_str_reduced = labels_str[labels_str['ID'].isin(combined_mri_patient_IDs)].sort_values(by='ID')

# FA Per Tract 
fa_per_tract_reduced = fa_per_tract[fa_per_tract['ID'].isin(combined_mri_patient_IDs)].sort_values(by='ID')
labels_fa_tr_reduced = labels_fa_tr[labels_fa_tr['ID'].isin(combined_mri_patient_IDs)].sort_values(by='ID')

# Join Tables 
combined_mri = pd.merge(fa_per_tract_reduced, struct_mri_reduced, on='ID', how='inner')
combined_mri_labels = labels_str_reduced


# In[125]:


combined_mri.set_index('ID', inplace=True)
combined_mri_labels.set_index('ID', inplace=True)


# In[ ]:


combined_mri_res = {}


# In[ ]:


rf_param_grid = {'n_estimators': [100], 
                 'max_depth': np.linspace(start=4,stop=20,num=5),
                 'max_features': np.linspace(start=0.2,stop=1,num=5),
                 'n_jobs': [-1], 
                 'class_weight': ['balanced']}

scalers = {disorder: RobustScaler() for disorder in most_common_disorders}

results = []

for i in range(N_rep): 

    rf_combined_res_vs_rest, rf_combined_y_pred_vs_rest = binary_classification(combined_mri, 
                                                                                combined_mri_labels, 
                                                                                cross_validation_splits[i],
                                                                                most_common_disorders, 
                                                                                scalers,
                                                                                RandomForestClassifier(), 
                                                                                rf_param_grid,
                                                                                scale_features=False,
                                                                                scenario='Disorder vs Rest',
                                                                                train_mode='Intersection')
    
    results.append(rf_combined_res_vs_rest)
    
combined_mri_res['rf'] = results


# In[ ]:


# svm_param_grid = {'C': np.logspace(-10, 10, 5), 
#                   'kernel': ['rbf'],
#                   'gamma': ['scale', 'auto'] + np.logspace(-2, 0, 5).tolist(),
#                   'class_weight': ['balanced']}

svm_param_grid = {'C': [1], 
                  'kernel': ['rbf'],
                  'gamma': ['scale'],
                  'class_weight': ['balanced']}

scalers = {}

for disorder in most_common_disorders: 
    if disorder == 'Attention-Deficit/Hyperactivity Disorder': 
        scalers[disorder] = RobustScaler()
    else: 
        scalers[disorder] = MinMaxScaler()
        
results = []

for i in range(N_rep): 

    svm_combined_res_vs_rest, svm_combined_y_pred_vs_rest = binary_classification(combined_mri, 
                                                                                  combined_mri_labels, 
                                                                                  cross_validation_splits[i],
                                                                                  most_common_disorders, 
                                                                                  scalers,
                                                                                  SVC(), 
                                                                                  svm_param_grid,
                                                                                  scale_features=True,
                                                                                  scenario='Disorder vs Rest',
                                                                                  train_mode='Intersection')
    
    results.append(svm_combined_res_vs_rest)

combined_mri_res['svm'] = results


# In[ ]:


with open('combined_mri_res_inter.pickle', 'wb') as f:
    pickle.dump(combined_mri_res, f)

