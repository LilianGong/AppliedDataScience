import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
import warnings
from sklearn.model_selection import KFold
warnings.simplefilter(action='ignore', category=FutureWarning)
import random
import matplotlib.pyplot as plt

# Logistic Regression
def get_pred_logreg(train,test):
    lr = LogisticRegression()
    lr.fit(X=train.iloc[:,:-1],y=train.iloc[:,-1])
    predict = pd.DataFrame(data= lr.predict_proba(test.iloc[:, :-1])[:,0],columns=['prob class0'])
    predict['true output'] =np.array(test.iloc[:,-1]).reshape(len(test),1)
    return predict

# Support Vector Machine
# Assumes the last column of data is the output dimension
def get_pred_svm(train,test):
    svm = SVC(probability=True)
    svm.fit(train.iloc[:,:-1],train.iloc[:,-1])
    predict = pd.DataFrame(data= svm.predict_proba(test.iloc[:, :-1])[:,0],columns=['prob class0'])
    predict['true output'] =np.array(test.iloc[:,-1]).reshape(len(test),1)
    return predict

# Naive Bayes
def get_pred_nb(train,test):
    nb = GaussianNB()
    nb.fit(train.iloc[:,:-1],train.iloc[:,-1])
    predict = pd.DataFrame(data= nb.predict_proba(test.iloc[:, :-1])[:,0],columns=['prob class0'])
    predict['true output'] =np.array(test.iloc[:,-1]).reshape(len(test),1)
    return predict

# k-Nearest Neighbor
def get_pred_knn(train,test,k):
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(train.iloc[:,:-1],train.iloc[:,-1])
    predict = pd.DataFrame(data= neigh.predict_proba(test.iloc[:, :-1])[:,0],columns=['prob class0'])
    predict['true output'] =np.array(test.iloc[:,-1]).reshape(len(test),1)
    return predict

def do_cv_class(data, num_folds, model_name):
    if num_folds > len(data):
        return "k fold out of range : k > n"
    else:
        kf = KFold(n_splits=num_folds)
        agg_pre = pd.DataFrame(columns=['prob class0','true output','k folds'])
        count = 0
        for train_id, test_id in kf.split(data.values):
            train = data.iloc[train_id,:]
            test = data.iloc[test_id,:]
            if model_name == 'logreg':
                pred = get_pred_logreg(train,test)
            if model_name == 'svm':
                pred = get_pred_svm(train,test)
            if model_name == 'nb':
                pred = get_pred_nb(train,test)
            if "nn" in model_name:
                pred = get_pred_knn(train,test,int(model_name[0]))
            #agg_pre['true output'] = act
            pred['k folds'] = count
            agg_pre = pd.concat([agg_pre,pred],axis = 0)
            count+=1
        cv_pred = agg_pre
    return cv_pred

def get_metrics(data,cutoff):
    predict_class = np.where(data.iloc[:,0]>cutoff,0,1)
    data['p_class'] = predict_class
    true_positive = data[(data.iloc[:,1]==1) & (data.iloc[:,2]==1)]
    false_positive = data[(data.iloc[:,1]==0) & (data.iloc[:,2]==1)]
    false_negative = data[(data.iloc[:,1]==1) & (data.iloc[:,2]==0)]
    accurate = data[data.iloc[:,1]==data.iloc[:,2]]
    precision = len(true_positive)/(len(true_positive)+len(false_positive))
    recall = len(true_positive)/(len(true_positive)+len(false_negative))
    report = pd.DataFrame(data=np.array([len(true_positive)/len(data),len(false_positive)\
    /len(data),len(accurate)/len(data),precision,recall]).reshape(1,5),\
    columns=['tpr','fpr','acc','precision','recall'])
    return report
#test:
'''
sample = wine.sample(10)
sample
pre1 = get_pred_nb(wine,sample)
print(pre1.head())
'''
if __name__ == "__main__":

    # prep
    wine = pd.read_csv("wine.csv")
    wine.head()
    wine['type'] = np.where(wine['type']=='high',1,0)
    random.seed(100)


# PART 1 test
    sample = wine.sample(10)
    pre1 = get_pred_nb(wine,wine)

# PART 2 test
    cv_pred = do_cv_class(wine,10,'5nn')
    #print(cv_pred.sample(10))

# PART 3 test
    predict = get_metrics(pre1,0.5)
    #print(pd.concat([predict,predict],axis = 0))

# PART 4
    # a)
    # choose k from 1 to 50
    knn_model = [str(i)+"nn" for i in range(1,51)]
    err_rate = []
    for model in knn_model:
        cv_pred = do_cv_class(wine,10,model)
        # currently I'm using majority vote to predict the classification result of knn
        # which means I use 0.5 as the threshold
        # class 0 if > 0.5; class 1 if <0.5
        cv_pred['pred_class'] = np.where(cv_pred['prob class0']>0.5,0,1)
        error = len(cv_pred[cv_pred['pred_class']!=cv_pred['true output']])/len(cv_pred)
        err_rate.append(error)


    plt.figure(figsize=(20,5))
    plt.plot(np.arange(1,51),err_rate)
    plt.ylabel("cv error rate")
    plt.xlabel("neighbor number k")
    plt.xticks(np.arange(1, 51, step=1))
    plt.title("k Nearest Neighbor 10-fold cv error rate")
    plt.savefig("knn_err_report.png")
    plt.show()


    # b)
    para_model = ['logreg','svm','nb']
    para_err = []
    for model in para_model:
        cv_pred = do_cv_class(wine,10,model)
        # currently I'm using threshold == 0.5 to predict the classfication result
        cv_pred['pred_class'] = np.where(cv_pred['prob class0']>0.5,0,1)
        error = len(cv_pred[cv_pred['pred_class']!=cv_pred['true output']])/len(cv_pred)
        para_err.append(error)

    low_rate = len(cv_pred[cv_pred['true output']==0])/len(cv_pred)
    default_err = low_rate if low_rate<0.5 else 1-low_rate

    print("Default classifier error rate: ",round(default_err,3),"\n")
    print("Logistic Regression cv error rate: ",round(para_err[0],3),"\n")
    print("SVM cv error rate: ",round(para_err[1],3),"\n")
    print("NB cv error rate: ",round(para_err[2],3),"\n")

    plt.figure(figsize=(5,5))
    x = [1, 2, 3]
    plt.scatter(x, para_err)

    for i, txt in enumerate([round(err,3) for err in para_err]):
        plt.annotate(txt, (x[i], para_err[i]))

    plt.xlabel("para models")
    plt.ylabel("cv error rate")
    plt.xticks(x, para_model)
    plt.title("3 parametric models 10-fold cv error rate")
# Pad margins so that markers don't get clipped by the axes
    plt.margins(0.2)
# Tweak spacing to prevent clipping of tick-labels
    plt.savefig("para_model_cv_err.png")
    plt.show()
