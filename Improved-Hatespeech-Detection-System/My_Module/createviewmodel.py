from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold 
from statistics  import mean,stdev
import matplotlib.pyplot as plt
import seaborn as sns
import random
import numpy as np
from tabulate import tabulate
import pandas as pd

class ViewModel:

    def view_boxplot(self):
        plt.boxplot(self.measure_dict.get('score'), labels=[self.model_name], showmeans=True)
        plt.show()
        
    def view_displot(self):
        sns.distplot(self.measure_dict.get('score'), bins=4).set_title("Stacking")
        plt.show()
        
    def view_plot(self):
        color = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
        plt.plot(self.measure_dict.get('score') , random.choice(color) )
        plt.plot([self.measure_mean_dict.get('mean_score')]*len(self.measure_dict.get('score')), '-')
        
    def view_confusion_matrix(self):
        group_names = ['True Neg','False Pos','False Neg','True Pos']
        group_counts = ["{0:0.0f}".format(float(value)) for value in self.measure_mean_dict.get('mean_c_matrix')[0].flatten()]
        group_percentages = ["{0:.2%}".format(float(value)) for value in self.measure_mean_dict.get('mean_c_matrix') [0].flatten()/np.sum(self.measure_mean_dict.get('mean_c_matrix')[0])]
        labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
        labels = np.asarray(labels).reshape(2,2)
        plt.title(self.model_name)   
        sns.heatmap(self.measure_mean_dict.get('mean_c_matrix')[0], annot=labels, fmt='s', square=True, cmap = 'Blues') 
        
    def view_measures(self):
            precision_list, recall_list, accuracy_list, f_score_list =list(),list(),list(),list()
            TN,FP,FN,TP = [value for value in self.measure_mean_dict.get('mean_c_matrix')[0].flatten()]
            precision = TP/(TP + FP)
            precision_list.append(precision*100)
            recall = TP/(TP + FN)
            recall_list.append(recall*100)
            accuracy = (TP+TN)/(TP+TN+FP+FN)
            accuracy_list.append(accuracy*100)
            f_score = (2 * recall * precision) / (recall + precision)
            f_score_list.append(f_score*100)
            measures = pd.DataFrame({'Model_Name':self.model_name ,'Precision (%)':precision_list,'Recall (%)':recall_list,'Accuracy (%)':accuracy_list,'F_score (%)':f_score_list})
            print(tabulate(measures, headers='keys', tablefmt='fancy_grid'))
        
class CreateModel(ViewModel):
    
    def __init__(self,X,y,model_object,model_name,n_split=3):
        self.model_object = model_object
        self.model_name = model_name
        self.measure_dict = {"FoldNo":[],"score":[],"c_matrix":[]}
        self.measure_mean_dict = {"Model_Name":[],"mean_score":[],"stdev":[],"mean_c_matrix":[]}
        self.c_matrix_list = []
        self.X = X
        self.y = y
        self.k_fold(n_split)
        
    def k_fold(self,n_split):
        counter = 1
        k_fold = KFold(n_splits = n_split, shuffle = True, random_state =1)
        for train , test in k_fold.split(self.X):
            X_train = self.X[train]
            y_train = self.y[train]
            X_test = self.X[test]
            y_test = self.y[test] 
            self.ml_model(X_train, y_train, X_test, y_test,counter)
            counter =counter+1
            
        self.measure_mean_func()

    def ml_model(self,X_train, y_train, X_test, y_test,counter):
        self.model_object = self.model_object.fit(X_train, y_train)
        self.score = self.model_object.score(X_test, y_test)
        self.pred_result = self.model_object.predict(X_test)
        self.c_matrix= confusion_matrix(y_test,self.pred_result)
        self.c_matrix_list = confusion_matrix(y_test,self.pred_result)
        self.measure(counter)

    def measure(self, counter):
        self.measure_dict["FoldNo"].append(counter)
        self.measure_dict["score"].append(self.score)
        self.measure_dict["c_matrix"].append(self.c_matrix)
    
    def measure_mean_func(self):
        self.measure_mean_dict["Model_Name"].append(self.model_name)
        
        mean_score = mean(self.measure_dict.get("score"))
        self.measure_mean_dict["mean_score"].append(mean_score)
        
        stdev_score = stdev(self.measure_dict.get("score"))
        self.measure_mean_dict["stdev"].append(stdev_score)
        
        c_temp = 0
        for cmat in self.measure_dict.get('c_matrix'):  
            c_temp += cmat        
        self.measure_mean_dict["mean_c_matrix"].append(c_temp)
        

       
    