import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from tqdm import tqdm
from contextlib import contextmanager
import time
from nltk import word_tokenize
import re
import os
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from bayes_opt import BayesianOptimization


@contextmanager
def timer(name):
    """
    Taken from Konstantin Lopuhin https://www.kaggle.com/lopuhin
    in script named : Mercari Golf: 0.3875 CV in 75 LOC, 1900 s
    https://www.kaggle.com/lopuhin/mercari-golf-0-3875-cv-in-75-loc-1900-s
    """
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')
    
    
def emb_index(file):
    embeddings_index = {}
    f = open(file, encoding = 'utf-8')
    for line in tqdm(f):
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()    
    return embeddings_index


def sent2vec(s,emb):
    words = str(s).lower()
    words = word_tokenize(words)
    #words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(emb[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    if type(v) != np.ndarray:
        return np.zeros(300)
    return v / np.sqrt((v ** 2).sum())


def clean_text(s):
    # Remove punct except ! and ?
    s = re.sub(r"[,.:|(;@)-/^—#&%$<=>`~{}\[\]\'\"]+\ *", " ", s)
    # Separate out ! and ?
    s = re.sub("!", " !", s)
    s = re.sub("\?", " ?", s)
    # Remove addition space
    s = ' '.join(s.split())
    return s


def count_consec_caps(text): # Count incremented by 1 for 3 consecutive words in capitals
    counter = 0
    super_counter = 0
    for substr in text.split():
        if (substr.isupper() == True):
            counter += 1
            if counter >=3:
                super_counter = super_counter + 1
                counter = 0
        else:
            if counter >=3:
                super_counter = super_counter + 1
                counter = 0
            else:
                counter = 0
    return super_counter


def num_feats(data):
#    df = pd.DataFrame(train.iloc[1:2,1])  
#    df.iloc[0,0] = "Need to count how many fucks are present you asshole!! YOU AND YOUR GF ARE PRICKS OF THE NTH ORDER!"
    df = pd.DataFrame()
    # num bangs
    df['num_bangs'] = data['comment_text'].apply(lambda text: len(re.findall("!",text)))    
    # num question marks
    df['num_questions'] = data['comment_text'].apply(lambda text: len(re.findall("\?",text)))    
    # num capitals
    df['num_caps'] = data['comment_text'].apply(lambda text: sum(1 for word in text.split() if word.isupper()))    
    # num 3 consecutive capitals
    df['num_consec3_caps'] = data['comment_text'].apply(lambda text: count_consec_caps(text))   
    # num words
    df['num_words'] = data['comment_text'].apply(lambda text: sum(1 for word in text.split() if word != '!' and word != '?'))     
    # num chars
    df['num_chars'] = data['comment_text'].apply(lambda text: len(list(text))) 
    # perc capitals
    df['perc_caps'] = np.where(df['num_words'] == 0, 0, df['num_consec3_caps'] / df['num_words'])
    # num you, you're, you are (words or phrases containing you)
    df['num_you'] = data['comment_text'].apply(lambda text: len(re.findall("you",text.lower())))   
    # you, you're, you are present or not
    df['flag_you'] = np.where(df['num_you'] >= 1, 1, 0)     
    # num 'are you'
    #df['num_areyou'] = data['comment_text'].apply(lambda text: len(re.findall("are you",text.lower()))) 
    # fuck present or not
    df['num_fk'] = data['comment_text'].apply(lambda text: len(re.findall("fuck",text.lower())))    
    # suck present or not
    #df['num_sk'] = data['comment_text'].apply(lambda text: len(re.findall("suck",text.lower())))    
    # shit present or not
    #df['num_shit'] = data['comment_text'].apply(lambda text: len(re.findall("shit",text.lower())))    
    # bitch present or not
    #df['num_bitch'] = data['comment_text'].apply(lambda text: len(re.findall("bitch",text.lower())))    
    # fag present or not
    df['num_fag'] = data['comment_text'].apply(lambda text: len(re.findall("fag",text.lower()) + re.findall("gay",text.lower())))   
    # nigger present or not
    #df['num_nig'] = data['comment_text'].apply(lambda text: len(re.findall("nigg",text.lower())))   
    # Wiki present or not
    #df['num_wiki'] = data['comment_text'].apply(lambda text: len(re.findall("wiki",text.lower())))
    # mother present or not
    #df['num_mother'] = data['comment_text'].apply(lambda text: len(re.findall("mother",text.lower())))         
    # profane word count
    #df['num_profane'] = data['comment_text'].apply(lambda text: sum(1 for word in text.split() if word.lower() in profane))
    return df


if __name__ == '__main__':
    
    with timer("Loading training, testing data"):
        train = pd.read_csv("train.csv")
        test = pd.read_csv("test.csv")
        label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
        y = train[label_cols].values
 
    
    with timer("Cleaning text"):
        train['comment_text'] = train['comment_text'].apply(lambda x: clean_text(x))
        test['comment_text'] = test['comment_text'].apply(lambda x: clean_text(x))
        
       
    with timer("Creating numerical features"):
        num_train = num_feats(train)
        num_test = num_feats(test)
        
   
    with timer("Loading Fasttext word vectors into dictionary"):    
        embedding_file = 'crawl-300d-2M.vec'
        emb_fasttext = emb_index(embedding_file) 
        print('Found %s word vectors in fasttext.' % len(emb_fasttext))     
    
    
    with timer("Convert sentences to summed fasttest word vectors"):
        X_t = [sent2vec(x,emb_fasttext) for x in train['comment_text']]
        X_te = [sent2vec(x,emb_fasttext) for x in test['comment_text']]
        
    
    with timer("Applying LSA to reduce dimensionality"):
        svd = TruncatedSVD(50)
        lsa = make_pipeline(svd, Normalizer(copy=False))
        
        X_t = lsa.fit_transform(X_t)
        X_te = lsa.fit_transform(X_te)

  
    path = "./final ensemble files/"
    for keyword in ["sub", "val"]:           
        files = [filename for filename in os.listdir(path) if filename.startswith(keyword)]
        exec(keyword + "= {i:pd.read_csv(os.path.join(path,file)) for i,file in enumerate(files)}")
           
    # Drop 'ID' from sub frames
    for sb in sub.values():
        sb.drop('id', axis=1,inplace=True)
        
            
    """
    1. Logistic Regression Stacking
    Picking relevant files to stack with logistic regression without intercept
    Converting probabilities to log odds so that classifier related biases are removed
    """
    with timer("Stacking with Logistic regression"):
        
        file_nums = [0,1,2,5,6,7,9,10,11,12,13]
        train_meta = pd.DataFrame()
        test_meta = pd.DataFrame()
        
        for num in file_nums:
            train_meta = pd.concat([train_meta, val[num]],axis=1)
            test_meta = pd.concat([test_meta, sub[num]],axis=1)
        
        # Instantiate classifier
        classifier = LogisticRegression(fit_intercept = False)
                  
        # Log odds transformation of probabilities for better normalization of different classifier results
        almost_zero = 1e-12
        almost_one = 1 - almost_zero  # To avoid division by zero
        train_meta[train_meta>almost_one] = almost_one
        train_meta[train_meta<almost_zero] = almost_zero
        train_meta = np.log(train_meta/(1-train_meta))
        test_meta[test_meta>almost_one] = almost_one
        test_meta[test_meta<almost_zero] = almost_zero
        test_meta = np.log(test_meta/(1-test_meta))
                  
        n_splits = 10
        folds = StratifiedKFold(n_splits = n_splits, shuffle = True, random_state=25)
        oof = np.empty([len(train_meta),len(label_cols)])
        sub_preds = np.zeros([len(test_meta),len(label_cols)])
        submission = np.zeros([len(test_meta),len(label_cols)])
                   
        for fold_, (trn_idx, val_idx) in enumerate(folds.split(y[:,0], y[:,0])): 
       
            X_train, y_train = train_meta.iloc[trn_idx], y[trn_idx]
            X_val, y_val = train_meta.iloc[val_idx], y[val_idx]
            
            print("Training fold %d" % fold_)

            for i, class_name in enumerate(label_cols):
                
                cols = list(np.arange(i,X_train.shape[1],6))
                    
                classifier.fit(X_train.iloc[:,cols], y_train[:,i])
             
                oof[val_idx,i] = classifier.predict_proba(X_val.iloc[:,cols])[:,1]
                sub_preds[:,i] = classifier.predict_proba(test_meta.iloc[:,cols])[:,1]

            auc = 0    
            for j in range(len(label_cols)):
                auc += roc_auc_score(y_val[:,j], oof[val_idx,j]) / len(label_cols)
        
            print("Fold %d AUC: %.6f" % (fold_,auc))
            submission = submission + (sub_preds / n_splits)   
        
        full_auc=0
        for j in range(len(label_cols)):
            full_auc += roc_auc_score(y[:,j], oof[:,j]) / len(label_cols)
        
        print("Full AUC: $.6f" % full_auc)
                           
        validation = pd.DataFrame(oof, columns = label_cols)
        validation.to_csv('validation_stack_logreg.csv', index=False) 
        
        submission = pd.concat([test['id'], pd.DataFrame(submission, columns = label_cols)], axis=1)
        submission.to_csv('submission_stack_logreg.csv', index=False)   


    """
    2. LightGBM stacking
    Picking relevant files to stack with Lightgbm.
    Concatenating with numerical features generated from feature engineering.
    Doing so when splitting into train and val sets
    """
    with timer("Stacking with LightGBM"):
        
        file_nums = [0,1,2,3,4,5,6,7,9,10,11,12,13]
        train_meta = pd.DataFrame()
        test_meta = pd.DataFrame()
        
        for num in file_nums:
            train_meta = pd.concat([train_meta, val[num]],axis=1)
            test_meta = pd.concat([train_meta, sub[num]],axis=1)
        
        n_splits = 10
        folds = StratifiedKFold(n_splits = n_splits, shuffle = True, random_state=25)
        oof = np.empty([len(train_meta),len(label_cols)])
        sub_preds = np.zeros([len(test_meta),len(label_cols)])
        submission = np.zeros([len(test_meta),len(label_cols)])
        auc_list=[]
                                               
        for fold_, (trn_idx, val_idx) in enumerate(folds.split(y[:,0], y[:,0])): 
            
            X_train, X_tr, y_train = train_meta.iloc[trn_idx], num_train.iloc[trn_idx], y[trn_idx]
            X_val, X_v, y_val = train_meta.iloc[val_idx], num_train.iloc[val_idx], y[val_idx]
           
            print("Training fold {}".format(fold_))
        
            for i, class_name in enumerate(label_cols):
                               
                X_train_comb = pd.concat([X_train,X_tr], axis=1)
                X_val_comb = pd.concat([X_val,X_v], axis=1)
                X_test_comb = pd.concat([test_meta, num_test], axis=1)
                     
                clf = LGBMClassifier(n_estimators=1000,
                                max_depth=6,
                                min_child_weight = 1,
                                objective="binary:logistic",
                                learning_rate=.1, 
                                subsample=.8, 
                                colsample_bytree=.8,
                                min_child_samples=750,
                                #max_bin=10,
                                n_jobs=2)
         
                clf.fit(X_train_comb, y_train[:,i], 
                    eval_set=[(X_val_comb, y_val[:,i])],
                    eval_metric='auc',
                    early_stopping_rounds=20,
                    verbose=False)

                oof[val_idx,i] = clf.predict_proba(X_val_comb)[:,1]
                sub_preds[:,i] = clf.predict_proba(X_test_comb)[:,1]
                                                          
            auc = 0    
            for j in range(len(label_cols)):
                auc += roc_auc_score(y_val[:,j], oof[val_idx,j]) / len(label_cols)
        
            print("Fold {} AUC: {}".format(fold_,auc))
            
            auc_list.append(auc)

            submission = submission + (sub_preds / n_splits)   
        
        print("Full AUC: %.6f" % np.mean(auc_list))
        
        validation = pd.DataFrame(oof, columns = label_cols)
        validation.to_csv('validation_stack_lgb.csv', index=False) 
        
        submission = pd.concat([test['id'], pd.DataFrame(submission, columns = label_cols)], axis=1)
        submission.to_csv('submission_stack_lgb.csv', index=False)   
        

    """
    3. Finding weights with Bayesian Optimization
    Picking relevant files to stack with Lightgbm.
    Concatenating with numerical features generated from feature engineering.
    Doing so when splitting into train and val sets
    """
    with timer("Optimal weights with Bayesian optimization"):    

        # Initialize weights dictionary with 0s        
        weight_dict_all = {'W'+str(i):[0] for i in range(len(val))}
                
        for i, class_name in enumerate(label_cols):
              
            def opt(**kwargs): 
                kwargs = np.array([value for key,value in kwargs.items()])
                pred = np.zeros([train.shape[0],len(label_cols)])
                for j in range(len(val)):                  
                    pred[:,i] += kwargs[j]*scaler.fit_transform(val[j])[:,i]                                     
                    auc = roc_auc_score(y[:,i], pred[:,i])                  
                return auc
        
            gp_params = {"alpha": 1e-5}             
            
            wt = BayesianOptimization(opt,
                           {'W'+str(i):(0,1) for i in range(len(val))}
                          )
            
            wt.maximize(n_iter=10, **gp_params)        
            weight_dict_class = wt.res['max']['max_params']
            print("Bayes wts for class %s done. AUC is %.6f" % (class_name, wt.res['max']['max_val']))
    
            for key, value in weight_dict_class.items():
                weight_dict_all[key].append(value) 
        
        # Calculate auc after finding all weights across all classes
        trn_preds = np.zeros([train.shape[0],len(label_cols)])
        pred = np.zeros([train.shape[0],len(label_cols)])
        auc=0
        
        for i, class_name in enumerate(label_cols):          
            for j in range(len(val)):                  
                    pred[:,i] +=weight_dict_all['W'+str(j)][i+1]*scaler.fit_transform(val[j])[:,i]            
            trn_preds[:,i] = pred[:,i]
            auc += roc_auc_score(y[:,i], trn_preds[:,i]) / len(label_cols)
        
        print("AUC after full bayes opt class-wise process: %.6f" % auc)
            
        # apply weights on submission
        sub_pred = np.zeros([test.shape[0], len(label_cols)])
        pred = np.zeros([test.shape[0],len(label_cols)])
        
        for i, class_name in enumerate(label_cols):          
            for j in range(len(sub)):                  
                    pred[:,i] +=weight_dict_all['W'+str(j)][i+1]*scaler.fit_transform(sub[j])[:,i]            
            sub_pred[:,i] = pred[:,i]
                        
        submission = pd.concat([test['id'], pd.DataFrame(sub_pred, columns = label_cols)], axis=1)
        submission.to_csv('submission_stack_bayeswt.csv', index=False)   
    
    
    """
    4. Blending different stacks to create final submissions
    """            
    blend_1 = pd.read_csv("submission_stack_logreg.csv")
    blend_2 = pd.read_csv("submission_stack_lgb.csv")
    blend_3 = pd.read_csv("submission_stack_bayeswt.csv")
    
    superblend = pd.DataFrame(np.zeros([len(blend_1),len(label_cols)]), columns = label_cols)
    
    for i in range(len(label_cols)):
        
        superblend.iloc[:,[i]] = ((blend_1.iloc[:,[i+1]] +
                                   blend_2.iloc[:,[i+1]] +
                                   blend_3.iloc[:,[i+1]])/3)
            
    superblend = pd.concat([blend_1['id'], superblend], axis=1)
    superblend.to_csv("superblend.csv", index = False)