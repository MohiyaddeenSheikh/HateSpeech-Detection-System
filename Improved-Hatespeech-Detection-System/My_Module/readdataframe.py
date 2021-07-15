import pandas as pd
class ReadDataFrame:
    def __init__(self,a):
        self.a=a
    def set_df(self,df_path):
        self.my_df = pd.read_csv(df_path)
    def get_df(self):
        return(self.my_df)
    
