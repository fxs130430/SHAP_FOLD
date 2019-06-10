import math
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics.classification import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
import pandas as pd
import shap
import xgboost
from sklearn.svm import SVC
from pandas.api.types import is_numeric_dtype
from sklearn.cluster import SpectralClustering
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import random
def generate_data(dataset_name,csv_url,numeric_cols,categorical_cols,class_names=['negative','positive'],drop_cols=[],target_label=1):
    df = pd.read_csv(csv_url, na_values=['?'])
    if dataset_name == 'emotions':
        df['label'] = [0 if x == 'NEGATIVE' else 1 if x == 'NEUTRAL' else 2 for x in df['label']]
    else:
        df['label'] = [0 if x == class_names[0] else 1 if x == class_names[1] else 2 for x in df['label']]


    if len(drop_cols) > 0:
        df = df.drop(drop_cols,1)

	numeric_columns = numeric_cols
    categorical_columns = categorical_cols
    cols_to_drop = []

    if dataset_name == 'cancer':
        clean_up = {'age': {'10_19':15 , '20_29':25, '30_39':35, '40_49':45, '50_59':55, '60_69':65, '70_79':75, '80_89':85, '90_99': 95},
                    'menopause': {'lt40':1, 'ge40':2, 'premeno':3},
                    'tumorsize': {'0_4':1, '5_9':2, '10_14':3, '15_19':4, '20_24':5, '25_29':6, '30_34':7, '35_39':8, '40_44':9,'45_49':10,'50_54':11,'55_59':12 },
                    'invnodes': {'0_2':1,'3_5':2,'6_8':3,'9_11':4,'12_14':5,'15_17':6,'18_20':7,'21_23':8,'24_26':9,'27_29':10,'30_32':11,'33_35':12,'36_39':13},
                    'nodecaps': {'yes':1, 'no':0},
                    'breast': {'left':1, 'right':2},
                    'breastquad': {'left_up':1,'left_low':2,'right_up':3,'right_low':4,'central':5}}
        df.replace(clean_up,inplace = True)


    for i in numeric_columns:
        df[i].fillna(df[i].mean(), inplace=True)



    if dataset_name == 'kidney':
        df['pc'].fillna('normal', inplace=True)
        df['pcc'].fillna('notpresent', inplace=True)
        df['ba'].fillna('notpresent', inplace=True)
        df['htn'].fillna('no', inplace=True)
        df['dm'].fillna('no', inplace=True)
        df['cad'].fillna('no', inplace=True)
        df['appet'].fillna('good', inplace=True)
        df['pe'].fillna('no', inplace=True)


    for col in categorical_columns:
        if is_numeric_dtype(df[col]):
            #df[col] = df[col].fillna('-1')
            if df[col].dtype != 'float64':
                df[col] = df[col].astype('Int8')
            #df[col] = df[col].astype(str)
            #df[col] = df[col].replace('-1', np.nan)
            lst = sorted(df[col].unique())
            if len(lst) == 2:
                cols_to_drop.append('{0}_{1}'.format(col, lst[0]))

    dic_distinct_vals = {}
    for c in categorical_columns:
        dic_distinct_vals[c] = df[c].unique()
    for col in numeric_columns:
        df[col] = pd.qcut(df[col],4,labels=False,duplicates='drop')
    df = dummy_df(df, categorical_columns)
    df = df.drop(cols_to_drop, 1)

    X = df.drop(['label'], 1)
    Y = df['label']

    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, stratify=Y, train_size=0.80,random_state = 1)
    #xgboost_gridCV(X_train.drop(['id'],axis=1).values,Y_train.values)

    clf = xgboost.XGBClassifier(objective='multi:softmax',n_estimators=2500)
    clf.fit(X_train.drop(['id'], axis=1).values, Y_train.values)

    #clf = SVC(kernel='poly', degree=1, gamma=0.01, C=1000, probability=True)
    #clf.fit(X_train.drop(['id'], 1).values, Y_train.values)

    print(sklearn.metrics.accuracy_score(Y_test.values, clf.predict(X_test.drop(['id'], axis=1).values)))

    Y_hat = clf.predict_proba(X.drop(['id'], axis=1).values)
    predict_fn = lambda x: clf.predict_proba(x).astype(float)


    print('Train', accuracy_score(Y_train.values, clf.predict(X_train.drop(['id'], 1).values)))
    print('Test', accuracy_score(Y_test.values, clf.predict(X_test.drop(['id'], 1).values)))
    print(classification_report(Y_test.values, clf.predict(X_test.drop(['id'], 1).values), target_names=['NEGATIVE', 'NEUTRAL','POSITIVE']))
    print(confusion_matrix(Y_test.values, clf.predict(X_test.drop(['id'], 1).values)))

    explainer_shap = shap.TreeExplainer(clf, X_train.drop(['id'], 1).values)
    shap_values = explainer_shap.shap_values(X_train.drop(['id'], 1).values)

    shap_vals_ids = np.append(X_train['id'].values.reshape(-1,1),shap_values,1)


    file_train = open('{0}_train.pl'.format(dataset_name), 'w')
    file_test = open('{0}_test.pl'.format(dataset_name), 'w')
    file_bias = open('{0}_bias.txt'.format(dataset_name), 'w')
    file_shap = open('{0}_shap_values.txt'.format(dataset_name), 'w')
    file_itemset_mining = open('{0}_itemset_mining.txt'.format(dataset_name), 'w')
    file_index_colname = open('{0}_index2colname.txt'.format(dataset_name), 'w')

    #### bias
    file_bias.write('#modeh(positive(var(case))) \n')
    file_bias.write('#constant(num0_3,0) \n')
    file_bias.write('#constant(num0_3,1) \n')
    file_bias.write('#constant(num0_3,2) \n')
    file_bias.write('#constant(num0_3,3) \n')
    ####


    file_train.write(':-style_check(-discontiguous).\n')
    file_test.write(':-style_check(-discontiguous).\n')
    file_train.write('foil_cwa(true).\n')
    file_train.write('foil_predicates([')
    columns = list(X_train.columns)
    bias = []
    index_2_colname = {}
    for i in range(1,len(columns)):
        if columns[i] in numeric_columns:
            bias.append('{0}/2'.format(columns[i]))
            file_bias.write('#modeb(1,{0}(var(case),const(num0_3))) \n'.format(columns[i]))
        else:
            ind = columns[i].rfind('_')
            pred_name = columns[i][:ind]
            entry = '{0}/2'.format(pred_name)
            if entry not in bias:
                bias.append(entry)
                file_bias.write('#modeb(1,{0}(var(case),const(args_{1}))) \n'.format(pred_name,pred_name))
                for v in dic_distinct_vals[pred_name]:
                    file_bias.write('#constant(args_{0},{1}) \n'.format(pred_name,str(v).lower().replace('.','')))

    for b in bias:
        file_train.write(b)
        file_train.write(',')

    file_train.write('positive/1]).\n')
    itemset_columns = []
    i = 0
    for row_index, row in X_train.iterrows():
        id = int(row['id'])
        label = ''
        if Y_train[row_index] == target_label:
            label = 'positive'
        #elif Y_train[row_index] == 0:
        else:
            label = 'negative'
        file_train.write('{0}(p{1}).\n'.format(label,id))
        sum_shap_neg,sum_shap_pos = (0,0)
        col_list = [[],[]]
        util_list = [[],[]]
        for col_index,col in enumerate(X_train.drop(['id'],1).columns):
            index_2_colname[col_index] = col
            if col in numeric_columns:
                file_train.write('{0}(p{1},{2}).\n'.format(col, id, row[col]))
                file_shap.write('data(p{0}):{1}(A,{2}):{3} \n'.format(id,col,row[col], shap_values[i][col_index]))
                num_col = '{0}_{1}'.format(col,row[col])
                if num_col not in itemset_columns:
                    itemset_columns.append(num_col)

                if label == 'positive' and shap_values[i][col_index] > 0:
                    sum_shap_pos += int(round(shap_values[i][col_index], 3) * 1000)
                    col_list[1].append(itemset_columns.index(num_col))
                    util_list[1].append(int(round(shap_values[i][col_index], 3) * 1000))
                elif label == 'negative' and shap_values[i][col_index] < 0:
                    sum_shap_neg += abs(int(round(shap_values[i][col_index], 3) * 1000))
                    col_list[0].append(itemset_columns.index(num_col))
                    util_list[0].append(int(abs(round(shap_values[i][col_index], 3)) * 1000))
            else:
                ind = col.rfind('_')
                second_arg = col[ind + 1:].lower().replace('.','')
                pred_name = col[:ind]
                if row[col] == 1:
                    file_train.write('{0}(p{1},{2}).\n'.format(pred_name,id,second_arg))
                    file_shap.write('data(p{0}):{1}(A,{2}):{3} \n'.format(id, pred_name, second_arg, shap_values[i][col_index]))
                    if col not in itemset_columns:
                        itemset_columns.append(col)
                    if label == 'positive' and shap_values[i][col_index] > 0:
                        sum_shap_pos += int(round(shap_values[i][col_index], 3) * 1000)
                        col_list[1].append(itemset_columns.index(col))
                        util_list[1].append(int(round(shap_values[i][col_index], 3) * 1000))
                    elif label == 'negative' and shap_values[i][col_index] < 0:
                        sum_shap_neg += abs(int(round(shap_values[i][col_index], 3) * 1000))
                        col_list[0].append(itemset_columns.index(col))
                        util_list[0].append(int(abs(round(shap_values[i][col_index], 3)) * 1000))
        i += 1
        zipped_0 = list(zip(col_list[0], util_list[0]))
        zipped_0.sort(key=lambda x: x[1],reverse = True)
        zipped_1 = list(zip(col_list[1],util_list[1]))
        zipped_1.sort(key=lambda x: x[1],reverse = True)
        zipped_0 = zipped_0[:min(len(zipped_0),100)]
        zipped_1 = zipped_1[:min(len(zipped_1), 100)]
        sum_shap_pos = sum([x[1] for x in zipped_1])
        sum_shap_neg = sum([x[1] for x in zipped_0])
        if label == 'positive':
            file_itemset_mining.write('positive(p{0})::{1}:{2}:{3}\n'.format(id," ".join(str(item) for item in [x[0] for x in zipped_1]),str(sum_shap_pos)," ".join(str(item) for item in [x[1] for x in zipped_1])))
        elif label == 'negative':
            file_itemset_mining.write('positive(p{0})::{1}:{2}:{3}\n'.format(id," ".join(str(item) for item in [x[0] for x in zipped_0]),str(round(sum_shap_neg,2))," ".join(str(item) for item in [x[1] for x in zipped_0])))





    for index, row in X_test.iterrows():
        id = int(row['id'])
        if Y_test[index] == target_label:
            file_test.write('positive(p{0}).\n'.format(id))
        #elif Y_test[index] == 0:
        else:
            file_test.write('negative(p{0}).\n'.format(id))

        for col in X_test.columns:
            if col == 'id':
                continue
            if col in numeric_columns:
                file_test.write('{0}(p{1},{2}).\n'.format(col, id, row[col]))
            else:
                ind = col.rfind('_')
                second_arg = col[ind + 1:].lower().replace('.','')
                pred_name = col[:ind]
                if row[col] == 1:
                    file_test.write('{0}(p{1},{2}).\n'.format(pred_name, id, second_arg))

    for ind,item in enumerate(itemset_columns):
        # replace for kidney column sc
        second_arg = item[item.rfind('_') + 1:].lower().replace('.','')
        pred_name = item[:item.rfind('_') ]
        file_index_colname.write('{0}->{1}(A,{2}) \n'.format(ind,pred_name,second_arg))


    file_train.close()
    file_test.close()
    file_bias.close()
    file_shap.close()
    file_itemset_mining.close()
    file_index_colname.close()



def xgboost_gridCV(X_train,Y_train):
    xgb_clf = xgboost.XGBClassifier()
    params = {
        'max_depth':  [6,7,8], # 5 is good but takes too long in kaggle env
        'subsample': [0.6],
        'colsample_bytree': [0.5],
        'n_estimators':  [1000,2000,3000],
        'reg_alpha':  [0.01, 0.02, 0.03, 0.04]
    }
    rs = GridSearchCV(xgb_clf,
                      params,
                      cv=5,
                      scoring="accuracy",
                      n_jobs=1,
                      verbose=2)
    rs.fit(X_train, Y_train)
    best_est = rs.best_estimator_
    print(best_est)
    exit(0)

def dummy_df(df, todummy_list):
    for x in todummy_list:
        dummies = pd.get_dummies(df[x], prefix=x, dummy_na=False)
        df = df.drop(x, 1)
        df = pd.concat([df, dummies], axis=1)
    return df


def main():

	# UNCOMMENT EACH DATASET TO GENERATE THE FOLD FILES
	# CREATE YOUR OWN LIKEWISE
	
	########################## UCI HEART DATASET ############################
    #dataset_name = 'heart'
    #csv_url = 'heart.csv'
    #numeric_cols = ['age', 'blood_pressure', 'serum_cholestoral', 'maximum_heart_rate_achieved', 'oldpeak']
    #categorical_cols = ['major_vessels','sex', 'slope', 'chest_pain', 'fasting_blood_sugar', 'resting_electrocardiographic_results',
    #                   'exercise_induced_angina', 'thal']
    #class_names = ['absent','present']
    #drop_cols = []
	########################## UCI HEART DATASET ############################

	
	########################## UCI BREAST CANCER WISCONSIN DATASET ############################
    #dataset_name = 'breastw'
    #csv_url = 'breastw.csv'
    #numeric_cols = []
    #categorical_cols = ['clump_thickness','cell_size_uniformity','cell_shape_uniformity','marginal_adhesion','single_epi_cell_size','bare_nuclei','bland_chromatin','normal_nucleoli','mitoses']
    #class_names = ['benign','malignant']
    #drop_cols = []
	########################## UCI BREAST CANCER WISCONSIN DATASET ############################
    
	########################## UCI AUTISM DATASET ############################
    #dataset_name = 'autism'
    #csv_url = 'autism.csv'
    #numeric_cols = ['age']
    #class_names = ['NO','YES']
    #categorical_cols = ['a1','a2','a3','a4','a5','a6','a7','a8',
    #                    'a9','a10','gender','ethnicity','jundice']
    #drop_cols = ['used_app_before', 'relation','autism']
	########################## UCI AUTISM DATASET ############################

	########################## UCI KIDNEY DATASET ############################
    #dataset_name = 'kidney'
    #csv_url = 'kidney.csv'
    #numeric_cols = ['age','bp','al','su','bgr','bu','sc','sod','pot','hemo','pcv','wbcc','rbcc']
    #class_names = ['notckd','ckd']
    #categorical_cols = ['sg','rbc','pc','pcc','ba','htn','dm','cad','appet','pe','ane']
    #drop_cols = []
	########################## UCI KIDNEY DATASET ############################

	########################## UCI CREDIT DATASET ############################
    #dataset_name = 'credit'
    #csv_url = 'credit.csv'
    #numeric_cols = ['a2','a3','a8','a11','a14','a15']
    #class_names = ['-','+']
    #categorical_cols = ['a1','a4','a5','a6','a7','a9','a10','a12','a13']
    #drop_cols = []
	########################## UCI CREDIT DATASET ############################

	########################## UCI CONGRESSIONAL VOTING DATASET ############################
    #dataset_name = 'voting'
    #csv_url = 'voting.csv'
    #numeric_cols = []
    #class_names = ['republican','democrat']
    #categorical_cols = ['handicapped_infants','water_project_cost_sharing',
    #                    'budget_resolution','physician_fee_freeze','el_salvador_aid',
    #                    'religious_groups_in_schools','anti_satellite_test_ban',
    #                    'aid_to_nicaraguan_contras','mx_missile','immigration',
    #                    'synfuels_corporation_cutback','education_spending','superfund_right_to_sue',
    #                    'crime','duty_free_exports','export_administration_act_south_africa']
    #drop_cols = []
	########################## UCI CONGRESSIONAL VOTING DATASET ############################

	########################## UCI MUSHROOM DATASET ############################
    #dataset_name = 'mushroom'
    #csv_url = 'mushroom.csv'
    #numeric_cols = []
    #class_names = ['e','p']
    #categorical_cols = ['cap_shape','cap_surface','cap_color','bruises','odor',
    #                    'gill_attachment','gill_spacing','gill_size','gill_color',
    #                    'stalk_shape','stalk_root','stalk_surface_above_ring',
    #                    'stalk_surface_below_ring','stalk_color_above_ring',
    #                    'stalk_color_below_ring','veil_type','veil_color',
    #                    'ring_number','ring_type','spore_print_color',
    #                    'population','habitat']
    #drop_cols = []
	########################## UCI MUSHROOM DATASET ############################
	
	########################## UCI SONAR DATASET ############################
    #dataset_name = 'sonar'
    #csv_url = 'sonar.csv'
    #numeric_cols = ['a{0}'.format(x) for x in range(1,61)]
    #class_names = ['Rock','Mine']
    #categorical_cols = []
    #drop_cols = []
	########################## UCI SONAR DATASET ############################

	########################## UCI KNIGHT-ROOK KNIGHT-PAWN DATASET ############################
    #dataset_name = 'krkp'
    #csv_url = 'krkp.csv'
    #numeric_cols = []
    #class_names = ['nowin', 'won']
    #categorical_cols = ['a{0}'.format(x) for x in range(1, 37)]
    #drop_cols = []
	########################## UCI KNIGHT-ROOK KNIGHT-PAWN DATASET ############################

	########################## UCI ACUTE DATASET ############################
    #dataset_name = 'acute'
    #csv_url = 'acute.csv'
    #numeric_cols = ['a1']
    #class_names = ['no','yes']
    #categorical_cols = ['a2','a3','a4','a5','a6']
    #drop_cols = []
	########################## UCI ACUTE DATASET ############################

	########################## UCI CARS DATASET ############################
    #dataset_name = 'cars'
    #csv_url = 'cars.csv'
    #numeric_cols = []
    #categorical_cols = ['buying','maint','doors','persons','lugboot','safety']
    #drop_cols = []
	#class_names = ['negative','positive']
	########################## UCI CARS DATASET ############################

    generate_data(dataset_name,csv_url=csv_url,numeric_cols=numeric_cols,
                  categorical_cols=categorical_cols,
                  class_names=class_names,
                  drop_cols=drop_cols,
                  target_label=1)

if __name__ == "__main__":
    main()