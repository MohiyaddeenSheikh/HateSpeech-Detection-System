import matplotlib.pyplot as plt
import seaborn as sns
import random
import numpy as np
from tabulate import tabulate
import pandas as pd


class CompareModels:
    #def __init__(self):
        #print(self.name_list,self.model_list,self.score_list)
        
    def selector(self, *models):
        if type(models[0]) is list:
            models = models[0]
        else:
            models = models
            
        self.model_list = list()
        self.score_list = list()
        self.name_list = list()
        
        for model in models:
            self.model_list.append(model)
            self.score_list.append(model.measure_dict.get('score'))
            self.name_list.append(model.model_name)
            
    def compare_boxplot(self,*models):
        self.selector(*models)
        fig, (ax) = plt.subplots(1 , 1,figsize=(18,7))
        ax.boxplot(self.score_list, labels=self.name_list, showmeans=True)
        plt.xticks(rotation=45)
        plt.show()
        
    def compare_displot(self,*models):
        self.selector(*models)
        import warnings
        warnings.filterwarnings('ignore')
        fig, (ax) = plt.subplots(1 , 1,figsize=(21,7))
        for name1, list1 in zip(self.name_list, self.score_list):    
            sns.distplot(list1, bins=4).set_title("Comparision Chart")
        fig.legend(labels=self.name_list)
        plt.show()
        
    def compare_plot(self,*models):
        self.selector(*models)
        fig, (ax) = plt.subplots(1 , 1,figsize=(21,5))
        color = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
        #plt.plot(self.measure_dict.get('score') , random.choice(color) )
        #plt.plot([self.measure_mean_dict.get('mean_score')]*len(self.measure_dict.get('score')), '-')
        for name1, list1 in zip(self.name_list, self.score_list):
            selected_color = random.choice(color)
            plt.plot(list1 )#, color =selected_color
            #plt.plot([mean(list1)]*len(list1), 'd' )#,color = selected_color
        fig.legend(labels=self.name_list)
        plt.show()
        
    def compare_confusion_matrix(self,*ml_models):
        if type(ml_models[0]) is list:
            ml_models = ml_models[0]
        else:
            ml_models = ml_models
            
        ml_model_list = list()
        ml_model_name = list()
        ml_c_matrix = list()
        for model in ml_models:
            ml_model_list.append(model)
            ml_c_matrix.append(model.measure_mean_dict.get('mean_c_matrix'))
            ml_model_name.append(model.model_name)
            
        fig, (ax) = plt.subplots(1 ,len(ml_model_list) ,figsize=(21,5)) #len(self.score_list)
        for name1, list1, i in zip(ml_model_name, ml_model_list, range(len(ml_model_list))):
            #print(list1.measure_mean_dict.get('mean_c_matrix'))
            group_names = ['True Neg','False Pos','False Neg','True Pos']
            group_counts = ["{0:0.0f}".format(float(value)) for value in list1.measure_mean_dict.get('mean_c_matrix')[0].flatten()]
            group_percentages = ["{0:.2%}".format(float(value)) for value in list1.measure_mean_dict.get('mean_c_matrix')[0].flatten()/np.sum(list1.measure_mean_dict.get('mean_c_matrix')[0])]
            labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
            labels = np.asarray(labels).reshape(2,2)
            plt.title(name1)
            if len(ml_model_list) > 1:
                ax[i].set(title = name1)
                sns.heatmap(list1.measure_mean_dict.get('mean_c_matrix')[0], annot=labels, fmt='s', square=True, cmap = 'Blues',ax=ax[i])
            else:
                plt.title(name1, fontsize =14)
                sns.heatmap(list1.measure_mean_dict.get('mean_c_matrix')[0], annot=labels, fmt='s', square=True, cmap = 'Blues')

    def compare_measures(self,*ml_models):
        if type(ml_models[0]) is list:
            ml_models = ml_models[0]
        else:
            ml_models = ml_models        
        
        ml_model_list = list()
        ml_model_name = list()
        ml_c_matrix = list()
        for model in ml_models:
            ml_model_list.append(model)
            ml_c_matrix.append(model.measure_mean_dict.get('mean_c_matrix'))
            ml_model_name.append(model.model_name)
        precision_list, recall_list, accuracy_list, f_score_list =list(),list(),list(),list()  
        
        for name1, list1, i in zip(ml_model_name, ml_model_list, range(len(ml_model_list))):
            TN,FP,FN,TP = [value for value in list1.measure_mean_dict.get('mean_c_matrix')[0].flatten()]
            precision = TP/(TP + FP)
            precision_list.append(precision*100)
            recall = TP/(TP + FN)
            recall_list.append(recall*100)
            accuracy = (TP+TN)/(TP+TN+FP+FN)
            accuracy_list.append(accuracy*100)
            f_score = (2 * recall * precision) / (recall + precision)
            f_score_list.append(f_score*100)
        measures = pd.DataFrame({'Model_Name':ml_model_name ,'Precision (%)':precision_list,'Recall (%)':recall_list,'Accuracy (%)':accuracy_list,'F_score (%)':f_score_list})
        print(tabulate(measures, headers='keys', tablefmt='fancy_grid'))
        
    def compare_all(self,*models):
        l=list(models)
        self.selector(l[0])
        self.compare_boxplot(l[0])
        self.compare_displot(l[0])
        self.compare_plot(l[0])
        self.compare_confusion_matrix(l[0])
        self.compare_measures(l[0])