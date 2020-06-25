import argparse
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV


def log_data(file):
    df = pd.read_csv(file)
    df['date'] = pd.to_datetime(df['date'])
    df.index = df['date']
    del df['date']
    df = df.fillna(method='ffill')
    dflog = df.apply(np.log)
    daily_return = dflog.diff(periods=1).dropna()
    return df,daily_return


def fit_pca(log_data):
    pca = PCA()
    Xstd = StandardScaler().fit_transform(daily_return)
    print(Xstd.shape)
    X_new = pca.fit_transform(Xstd)
    print(X_new.shape)
    return pca, X_new


def plot_pca(pca):
    variance_ratio = pca.explained_variance_ratio_
    cumu_variance_ratio = [sum(variance_ratio[:i]) for i in range(1,len(variance_ratio))]
    fig1 = plt.plot(variance_ratio,color = '#ed6663')
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.xticks(rotation=0)
    plt.xlabel("principle component")
    plt.ylabel("variance ratio")
    plt.title("scree plot of variance")
    plt.show()
    plt.savefig("scree_plot.png")

    fig2 = plt.plot(cumu_variance_ratio,color = '#0f4c81')
    plt.xticks(rotation=0)
    plt.xlabel("principle component")
    plt.ylabel("cumulative variance ratio")
    plt.title("cumulative variance ratio")
    plt.savefig("cumu_variance.png")
    plt.show()
    return variance_ratio,cumu_variance_ratio

def find_retain(cumu_variance_ratio,threshold):
    for ratio in cumu_variance_ratio:
        if ratio >= threshold:
            w = str("{} principal components should be retained "\
            .format(cumu_variance_ratio.index(ratio)+1) +
             "to capture at least 80% of the total variance.")
            print(w)
            break


def reconstruction_err(retain,variance_ratio):
    w = str("{} is the reconstruction error "\
    .format(round(sum(variance_ratio[retain:]),3))
    +"if we only retain top 2 of the PCA components.")
    print(w)


def time_series_ana(df,pca):
    first_prin = pca.components_[0]
    time_series_dat = [df.iloc[i,:].values.dot(first_prin) for i in range(len(df))]
    frame = pd.DataFrame(time_series_dat,columns=['projection value'])
    frame['date'] = df.index
    frame.plot(x='date',y="projection value",marker='o',figsize=(20,5),markersize=1,color = '#1b262c')
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.xticks(rotation=0)
    plt.ylabel("projection value")
    plt.title("data projection value on 1st principle")
    plt.savefig("pc1_date.png")
    #plt.show()
    w = str("the lowest value is {} ".format(round(frame.iloc[np.argmin(time_series_dat),0],3))+
    "in {}".format(frame.iloc[np.argmin(first_prin),1]))
    print(w)


def sector_plot(pca,df):
    sector = pd.read_csv('SP500_ticker_mod.csv')
    frame = pd.DataFrame({'ticker':df.columns,\
    'pc1' : pca.components_[0],'pc2' : pca.components_[1]})
    merge = frame.merge(sector,how='inner')
    gp = merge.groupby(['sector'])
    mean_frame = gp[['pc1','pc2']].mean()
    mean_frame.reset_index(-1,inplace=True)

    mean_frame.plot.bar(x ='sector',y="pc1",figsize=(20,5),color='#0f4c81')
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.ylabel("1st principal compoment weight")
    plt.xlabel("sector")
    plt.xticks(rotation=0)
    plt.title("1st principal compoment mean weight by sector")
    plt.savefig("pc1_sector.png")
    plt.show()

    mean_frame.plot.bar(x ='sector',y="pc2",figsize=(20,5),color='#ed6663')
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.ylabel("2st principal compoment weight")
    plt.xlabel("sector")
    plt.xticks(rotation=0)
    plt.title("2st principal compoment mean weight by sector")
    plt.savefig("pc2_sector.png")
    plt.show()



def filter_features_by_cor(df):
    y = df.iloc[:,-1]
    corr_coef = []

    for col_index in range(len(df.columns)-1):
        x = df.iloc[:,col_index]
        cor = abs(pearsonr(x, y)[0])
        #cor = np.corrcoef(x = X, y= Y)
        corr_coef.append(cor)
    feature_corr = pd.DataFrame({"feature":df.columns[:-1],"correlation":corr_coef})\
    .sort_values('correlation',ascending=False)

    return feature_corr


def subset_selection(df):
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    selected = []
    X_new_ls = []
    for k in [1,2,3]:
        kbest = SelectKBest(f_regression, k=k)
        X_new = kbest.fit_transform(X, y)
        X_new_ls.append(X_new)
        selected.append(",".join(df.columns[np.where(kbest.get_support()==True)].tolist()))
    print(pd.DataFrame({"size of subset":[1,2,3],"feature(s) selected":selected}))
    return X_new_ls


def rfecv_selection(df):
    # use rfecv with linear regression
    lr = LinearRegression()
    X = df.iloc[:,:-1].values
    y = df.iloc[:,-1].values
    rfecv = RFECV(estimator=lr, step=1,cv=10)
    rfecv.fit(X, y)
    selected = ",".join(df.columns[np.where(rfecv.support_==True)].tolist())

    print("Optimal number of features : {}".format(rfecv.n_features_))
    print("Features get selected are : {}".format(selected))
    print("The cv score for the optimized model is : {}".format(round(max(rfecv.grid_scores_)),3))


if __name__ == "__main__":

    '''
    Problem 1.a
    '''
    print("**********Problem 1.a.1**********")
    df, daily_return = log_data('SP500_close_price.csv')
    pca, X_new = fit_pca(daily_return)


    print("**********Problem a.1 a.2**********")
    variance_ratio,cumu_variance_ratio = plot_pca(pca)
    print("plots saved as scree_plot.png and cumu_variance.png ")

    print("**********Problem 1.a.3**********")
    find_retain(cumu_variance_ratio,0.8)

    print("**********Problem 1.a.4**********")
    reconstruction_err(2,variance_ratio)

    '''
    Problem 1.b
    '''
    # 1.b.1
    print("**********Problem 1.b.1**********")
    time_series_ana(df,pca)
    print("plots saved as pc1_date.png")

    print("**********Problem 1.b.3-4**********")
    print("plots saved as pc1_sector.png and pc2_sector.png ")
    #sector_plot(pca,df)

    '''
    Problem 2.a 2.b
    '''
    print("**********Problem 2.a 2.b**********")
    bmi = pd.read_csv("BMI.csv")
    feature_corr = filter_features_by_cor(bmi)
    print(feature_corr.head(3))
    print("the top 3 features are: {}".format(','.join(feature_corr['feature'][:3])))

    '''
    Problem 2.c
    '''
    # some background knowledge for unvariate feature selection
    #Univariate feature selection works
    #by selecting the best features based on univariate statistical tests.
    #It can be seen as a preprocessing step to an estimator.
    #SelectKBest: Select features according to the k highest scores.
    print("**********Problem 2.c**********")
    X_new_ls = subset_selection(bmi)

    '''
    Problem 2.d
    '''
    print("**********Problem 2.d**********")
    # some background knowledge for RFE:
    #  recursive feature elimination (RFE)
    #is to select features by recursively
    #considering smaller and smaller sets of features.
    rfecv_selection(bmi)
