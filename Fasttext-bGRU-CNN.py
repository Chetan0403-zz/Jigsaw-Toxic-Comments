import time
start_time = time.time()
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import sys, os, re, csv, codecs, numpy as np, pandas as pd
np.random.seed(32)
os.environ["OMP_NUM_THREADS"] = "4"
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D
from keras.layers import Bidirectional, GlobalMaxPool1D, MaxPooling1D, Add, Flatten
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D
from keras.models import Model, load_model
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras import backend as K
from keras.engine import InputSpec, Layer
os.environ['OMP_NUM_THREADS'] = '4'
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.layers import GRU, BatchNormalization, Conv1D, MaxPooling1D
import warnings
warnings.filterwarnings('ignore')
from keras.callbacks import Callback


max_features = 100000
maxlen = 150
embed_size = 300
batch_size = 128
n_splits = 10
    

def text_preprocess(text): 
    
    # Emoticons
    text = text.replace(":/", " bad ")
    text = text.replace(":&gt;", " sad ")
    text = text.replace(":')", " sad ")
    text = text.replace(":-(", " frown ")
    text = text.replace(":(", " frown ")
    text = text.replace(":s", " frown ")
    text = text.replace(":-s", " frown ")
    text = text.replace("&lt;3", " heart ")
    text = text.replace(":d", " smile ")
    text = text.replace(":p", " smile ")
    text = text.replace(":dd", " smile ")
    text = text.replace("8)", " smile ")
    text = text.replace(":-)", " smile ")
    text = text.replace(":)", " smile ")
    text = text.replace(";)", " smile ")
    text = text.replace("(-:", " smile ")
    text = text.replace("(:", " smile ")
    text = text.replace(":/", " worry ")
    text = text.replace(":&gt;", " angry ")
    text = text.replace(":')", " sad ")
    text = text.replace(":-(", " sad ")
    text = text.replace(":(", " sad ")
    text = text.replace(":s", " sad ")
    text = text.replace(":-s", " sad ")
    
    # Shortforms   
    text = re.sub(r'[\w]*don\'t[\w]*','do not',text)
    text = re.sub(r'[\w]*i\'ll[\w]*','i will',text)
    text = re.sub(r'[\w]*wasn\'t[\w]*','was not',text)
    text = re.sub(r'[\w]*there\'s[\w]*','there is',text)
    text = re.sub(r'[\w]*i\'m[\w]*','i am',text)
    text = re.sub(r'[\w]*won\'t[\w]*','will not',text)
    text = re.sub(r'[\w]*let\'s[\w]*','let us',text)
    text = re.sub(r'[\w]*i\'d[\w]*','i would',text)
    text = re.sub(r'[\w]*they\'re[\w]*','they are',text)
    text = re.sub(r'[\w]*haven\'t[\w]*','have not',text)
    text = re.sub(r'[\w]*that\'s[\w]*','that is',text)
    text = re.sub(r'[\w]*couldn\'t[\w]*','could not',text)
    text = re.sub(r'[\w]*aren\'t[\w]*','are not',text)
    text = re.sub(r'[\w]*wouldn\'t[\w]*','would not',text)
    text = re.sub(r'[\w]*you\'ve[\w]*','you have',text)
    text = re.sub(r'[\w]*you\'ll[\w]*','you will',text)
    text = re.sub(r'[\w]*what\'s[\w]*','what is',text)
    text = re.sub(r'[\w]*we\'re[\w]*','we are',text)
    text = re.sub(r'[\w]*doesn\'t[\w]*','does not',text)
    text = re.sub(r'[\w]*can\'t[\w]*','can not',text)
    text = re.sub(r'[\w]*shouldn\'t[\w]*','should not',text)
    text = re.sub(r'[\w]*didn\'t[\w]*','did not',text)
    text = re.sub(r'[\w]*here\'s[\w]*','here is',text)
    text = re.sub(r'[\w]*you\'d[\w]*','you would',text)
    text = re.sub(r'[\w]*he\'s[\w]*','he is',text)
    text = re.sub(r'[\w]*she\'s[\w]*','she is',text)
    text = re.sub(r'[\w]*weren\'t[\w]*','were not',text)
    
    # Remove punct except ! and ?
    text = re.sub(r"[,.:|(;@)-/^—#&%$<=>`~{}\[\]\'\"]+\ *", " ", text)
    # Separate out ! and ?
    text = re.sub("!", " ! ", text)
    text = re.sub("\?", " ? ", text)
  
    # Drop numbers
    text = re.sub("\\d+", " ", text)
        
    # Check if at least 3 consecutive substrings are in caps. Add <caps> tag at the end
    counter = 0
    for substr in text.split():
        if (substr.isupper() == True):
            counter += 1
            if counter >=3:
                text = text + " " + "XYZ" # XYZ chosen for capitals since it is a rare word present in embedding
                counter = 0
        else:
            if counter >=3:
                text = text + " " + "XYZ"
                counter = 0
            else:
                counter = 0
    
    # Convert to lower
    text = text.lower()
    
    # Lots of words are not present in the fasttext embeddings. Replace them
    text = re.sub(r'[\w]*(fuc|fck|fvc)[\w]*','fuck',text)
    text = re.sub(r'[\w]*fag[\w]*','gay',text)
    text = re.sub(r'[\w]*gay[\w]*','gay',text)
    text = re.sub(r'[\w]*peni[\w]*','dick',text)
    text = re.sub(r'[\w]*(dic|dik)[\w]*','dick',text)
    text = re.sub(r'[\w]*bi[\w]*ch[\w]*','bitch',text)
    text = re.sub(r'[\w]*s[\w]*x[\w]*','sex',text)
    text = re.sub(r'[\w]*s[\w]*k[\w]*','suck',text)
    text = re.sub(r'[\w]*nigg[\w]*','suck',text)
    text = re.sub(r'[\w]*cock[\w]*','dick',text)
    text = re.sub(r'[\w]*cunt[\w]*','cunt',text)
    text = re.sub(r'[\w]*anal[\w]*','anal',text)
    #text = re.sub(r'[\w]*ha{2,}[\w]*','haha',text)
    text = re.sub(r'[\w]*haha[\w]*','haha',text)
    text = re.sub(r'[\w]*wiki[\w]*','wikipedia',text)
    text = re.sub(r'[\w]*ency[\w]ia[\w]*','encyclopedia',text)   
           
    # Remove unwanted space
    text = " ".join(text.split())
     
    # Check for profanity, Add <profane> tag at the end
#    for substr in text.split():
#        if (substr in profane):
#            text = text + " " + "ABC"
    
#    # Replace misspelled words with correct words    
#    for substr in text.split():
#        if substr in misspell:
#            text = re.sub(substr, misspell[substr], text)

    return text


class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: {:d} - score: {:.6f}".format(epoch, score))
    

if __name__ == '__main__':
    
    
    EMBEDDING_FILE='crawl-300d-2M.vec'
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    misspellings = pd.read_csv('misspellings.csv')
    
    list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    y = train[list_classes].values
    
    
    # Clean up misspellings file and create dictionary
    misspellings = [re.sub("->"," ",spelling) for spelling in misspellings]
    misspell = {text.split()[0].lower():text.split()[1].lower() for text in misspellings}
    
    
    # Clean text
    train['comment_text'] = train['comment_text'].apply(text_preprocess)
    test['comment_text'] = test['comment_text'].apply(text_preprocess)   


    # Vectorize comments
    list_sentences_train = train["comment_text"].str.lower().fillna("_na_").values
    list_sentences_test = test["comment_text"].str.lower().fillna("_na_").values
    
    tokenizer = Tokenizer(num_words = max_features, 
                          lower = True,
                          filters='"#$%&()*+,-./:;=@[\\]^_`“<>{|}~\t\n') # not filtering out ! and ?, < >
                          
    tokenizer.fit_on_texts(list(list_sentences_train)+list(list_sentences_test))
    list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
    list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
    X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
    X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)


    # Load word vector        
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')    
    embedding_index = dict(get_coefs(*o.strip().split(" ")) for o in open(EMBEDDING_FILE, encoding='utf-8'))
    
    
    # Prepare embedding matrix
    word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.zeros((nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    
    
    # Split data into folds, compile model and run
    folds = StratifiedKFold(n_splits = n_splits, shuffle = True, random_state=25)
    oof = np.empty([len(X_t),len(list_classes)])
    sub_preds = np.zeros([len(X_te),len(list_classes)])
    foldwise_auc = []
        
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(y[:,0], y[:,0])): #StratifiedKFold expects array of shape (n,)
        
        X_train, y_train = X_t[trn_idx], y[trn_idx]
        X_val, y_val = X_t[val_idx], y[val_idx]
        
        print("Running fold %d" % fold_)     
        
        ra_val = RocAucEvaluation(validation_data=(X_val, y_val), interval=1)
        earlystop = EarlyStopping(monitor='val_loss', mode="min", patience=5, \
                          verbose=1) 
        
        def build_model(lr = 0.0, lr_d = 0.0, units = 0, dr = 0.0):
            inp = Input(shape = (maxlen,))
            x = Embedding(max_features, embed_size, weights = [embedding_matrix], trainable = False)(inp)
            x = SpatialDropout1D(dr)(x)
            x = Bidirectional(GRU(units, return_sequences = True))(x)
#            x = Dropout(0.1)(x)
            x = Conv1D(80, kernel_size = 3, padding = "valid", kernel_initializer = "he_uniform")(x)
#            avg_pool = GlobalAveragePooling1D()(x)
#            max_pool = GlobalMaxPooling1D()(x)
#            x = concatenate([avg_pool, max_pool])
            x = GlobalMaxPooling1D()(x)
            x = Dense(units,activation='relu')(x)
            x = Dense(6, activation = "sigmoid")(x)
            
            model = Model(inputs = inp, outputs = x)
            model.compile(loss = "binary_crossentropy", optimizer = Adam(lr = lr, decay = lr_d), metrics = ["accuracy"])
            model.fit(X_train, y_train, batch_size = 128, epochs = 4, validation_data = (X_val, y_val), 
                                verbose = 1, callbacks = [ra_val, earlystop])
            
            return model
                                         
        model = build_model(lr = 1e-3, lr_d = 0, units = 144, dr = 0.2)
        
        pred = model.predict(X_val, batch_size = 1024, verbose = 1)          
        
        oof[val_idx] = pred
        
        sub_preds += model.predict([X_te], batch_size=1024, verbose=1) / n_splits
            
    auc=0
    for i in range(len(list_classes)):
        auc += roc_auc_score(y[:,i], oof[:,i]) / len(list_classes)
    
    print("AUC for full run: %.6f" % auc)
    
    validation = pd.DataFrame(oof, columns = list_classes)
    validation.to_csv('validation_fasttext_bgrucnn.csv', index=False) 
    
    submission = pd.concat([test['id'], pd.DataFrame(sub_preds, columns = list_classes)], axis=1)
    submission.to_csv('submission_fasttext_bgrucnn.csv', index=False) 