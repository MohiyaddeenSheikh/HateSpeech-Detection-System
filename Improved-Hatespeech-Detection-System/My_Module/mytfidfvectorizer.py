import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

class MyTfidfVectorizer:
    def __init__(self,my_df,len,ngram=(1,1)):
        self.my_df = my_df
        self.len = len
        self.feat_ext_tfidf(self.my_df['preproc_step_7'][:len],ngram)
        
    def feat_ext_tfidf(self,df_tweet,ngram):
        self.tdidf_v = TfidfVectorizer(stop_words='english',min_df=5, max_features=3000, ngram_range=ngram)  #sublinear_tf=True, min_df=5, ngram_range=(1, 2), stop_words='english')
        self.tfidf_fit_matrix  = self.tdidf_v.fit_transform(df_tweet.astype('U')) #df_tweet.apply(TreebankWordDetokenizer().detokenize)
        self.feature_names_tfidf = self.tdidf_v.get_feature_names()
        self.data_in_array()
        
    def data_in_array(self):
        self.data = self.tfidf_fit_matrix.toarray()
        
    def detail(self):
        print("-----------------------------------------------")
        print("Features Length: ",len(sorted(self.feature_names_tfidf)))
        print("Features Length: ",self.feature_names_tfidf)
        print("Data / TFIDF fit-transform Matrix Shape: " , self.tfidf_fit_matrix.shape)
        print("-----------------------------------------------")
        
class SetLabel:
    def __init__(self,my_df,len):
        self.data  = my_df['label'][:len].to_numpy()
    def detail(self):
        print("-----------------------------------------------")
        print("Label shape : ",self.label.shape)
        print("-----------------------------------------------")
        

