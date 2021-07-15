from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn import model_selection
from mlxtend.classifier import StackingClassifier 
from sklearn.ensemble import StackingClassifier

import seaborn as sns
import pandas as pd

class Import_classifiers():
    def __init__(self):
        self.classifier_object_list = list()
        self.classifier_name_list = list()
        self.multiple_classifiers_list =list()
        
    def classifiers_collection(self):
        self.classifier_object_dict={"lr": LogisticRegression() ,
                                "knn": KNeighborsClassifier() ,
                                "dt":DecisionTreeClassifier(),
                                "rf":RandomForestClassifier(),
                                "svm":SVC(),
                                "nb":BernoulliNB(),
                                "multi_nb":MultinomialNB(),
                                "ab":AdaBoostClassifier(),
                                "mlp":MLPClassifier(hidden_layer_sizes=(11,11,11),max_iter=500)}
        
        self.classifier_name_dict={"lr": 'LogisticRegression' ,
                                "knn": 'KNeighborsClassifier' ,
                                "dt":'DecisionTreeClassifier',
                                "rf":'RandomForestClassifier',
                                "svm":'SupportVectorMachine',
                                "nb":'BernoulliNaiveBayes',
                                "multi_nb":'MultinomialNaiveBayes',
                                "ab":'AdaBoostClassifier',
                                "mlp":'MLPClassifier' } 
        
    def set_multiple_classifers(self, *classifiers):
        self.classifiers_collection()
        for item in classifiers:
            self.classifier_object_list.append(self.classifier_object_dict.get(item))
            self.classifier_name_list.append(self.classifier_name_dict.get(item))
            self.multiple_classifiers_list.append((self.classifier_name_dict.get(item),self.classifier_object_dict.get(item)))

        
class Create_meta_classifier(Import_classifiers):
    def __init__(self,classifier = 'lr' ):     
        self.set_classifier(classifier)
        
    def set_classifier(self, classifier):
        self.classifiers_collection()
        self.meta_classifier = self.classifier_object_dict.get(classifier)
        self.meta_classifier_name = self.classifier_name_dict.get(classifier)
         
    #def create_my_model(self):
        #print(obj.classifier_object_list)
        #self.my_model1 = StackingClassifier(estimators=self.classifier_object_list, final_estimator=self.meta_classifier, cv=5)

class CreateMyModel:
    def __init__(self,clsfrs,meta_clsfr):
        self.clsfrs_list = clsfrs.multiple_classifiers_list
        self.meta_clsfr = meta_clsfr.meta_classifier
        self.my_model()
        
    def my_model(self):
        self.model = StackingClassifier(estimators=self.clsfrs_list, final_estimator=self.meta_clsfr, cv=5)
               
