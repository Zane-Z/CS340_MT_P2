import os.path
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from time_series import TimeSeries
import pandas as pd
from datetime import datetime, timedelta

def mode(y):
    """Computes the element with the maximum count

    Parameters
    ----------
    y : an input numpy array

    Returns
    -------
    y_mode :
        Returns the element with the maximum count
    """
    if len(y)==0:
        return -1
    else:
        return stats.mode(y.flatten())[0][0]


def euclidean_dist_squared(X, Xtest):
    """Computes the Euclidean distance between rows of 'X' and rows of 'Xtest'

    Parameters
    ----------
    X : an N by D numpy array
    Xtest: an T by D numpy array

    Returns: an array of size N by T containing the pairwise squared Euclidean distances.

    Python/Numpy (and other numerical languages like Matlab and R)
    can be slow at executing operations in `for' loops, but allows extremely-fast
    hardware-dependent vector and matrix operations. By taking advantage of SIMD registers and
    multiple cores (and faster matrix-multiplication algorithms), vector and matrix operations in
    Numpy will often be several times faster than if you implemented them yourself in a fast
    language like C. The following code will form a matrix containing the squared Euclidean
    distances between all training and test points. If the output is stored in D, then
    element D[i,j] gives the squared Euclidean distance between training point
    i and testing point j. It exploits the identity (a-b)^2 = a^2 + b^2 - 2ab.
    The right-hand-side of the above is more amenable to vector/matrix operations.
    """

    return np.sum(X**2, axis=1)[:,None] + np.sum(Xtest**2, axis=1)[None] - 2 * np.dot(X,Xtest.T)

    # without broadcasting:
    # n,d = X.shape
    # t,d = Xtest.shape
    # D = X**2@np.ones((d,t)) + np.ones((n,d))@(Xtest.T)**2 - 2*X@Xtest.T

def evaluate_model(model, X, y, X_test, y_test):
    model.fit(X,y)

    y_pred = model.predict(X)
    tr_error = np.mean(y_pred != y)

    y_pred = model.predict(X_test)
    te_error = np.mean(y_pred != y_test)
    print("    Training error: %.3f" % tr_error)
    print("    Testing error: %.3f" % te_error)


def test_and_plot(model,X,y,Xtest=None,ytest=None,title=None,filename=None):

    # Compute training error
    yhat = model.predict(X)
    trainError = np.mean((yhat - y)**2)
    print("Training error = %.1f" % trainError)

    # Compute test error
    if Xtest is not None and ytest is not None:
        yhat = model.predict(Xtest)
        testError = np.mean((yhat - ytest)**2)
        print("Test error     = %.1f" % testError)

    # Plot model
    plt.figure()
    plt.plot(X,y,'b.')

    # Choose points to evaluate the function
    Xgrid = np.linspace(np.min(X),np.max(X),1000)[:,None]
    ygrid = model.predict(Xgrid)
    plt.plot(Xgrid, ygrid, 'g')

    if title is not None:
        plt.title(title)

    if filename is not None:
        filename = os.path.join("..", "figs", filename)
        print("Saving", filename)
        plt.savefig(filename)

def to_future_matrix(X, days_predict=5, days_window=5, train_model=None):
    #Note X is the dataframe that follow the format when first read from excel
    #initialize TS model
    ts_model = TimeSeries(days_window, train_model)
    all_ctry_new_df = pd.DataFrame(columns =["country_id", "date", "cases", "deaths", "cases_14_100k", "cases_100k"])
    country_id_col = X.loc[:,"country_id"].unique()
    for country in country_id_col:
        X_cur = X[X["country_id"]==country].copy(deep=True)
        ctry_df = process_ts_ctry(ts_model, country, X_cur, days_predict, days_window, train_mode)
        all_ctry_new_df = pd.concat([all_ctry_new_df, ctry_df], axis=0)
    return all_ctry_new_df

def process_ts_ctry(ts_model, country, X, days_predict=5, days_window=5, train_model=None):
    #ctry_new_mt = np.zeros((days_predict, 6)) #could not convert string to float: '09/10/2020'
    X_sorted = X.sort_values(by=['date'], inplace=True, ascending=True)
    
    #1 Get country array
    countries = get_countries(country, days_predict)
    
    #2 Get the new dates
    new_dates = get_new_dates(X_sorted.iloc[(X_sorted.shape[0]-1), 1], days_predict)
    #ctry_new_mt[:,1]=new_dates
    
    #3 Get cases
    cases_array = X_sorted[:, 2]
    cases_array = np.reshape(cases_array, (cases_array.shape[0], 1))
    cases_array = cases_array.astype(float)
    #new_cases = get_new_values(cases_array, days_predict, days_window, train_model)
    new_cases = get_new_values(ts_model, cases_array, days_predict)
    
    #4 Get deaths
    deaths_array = X_sorted[:, 3]
    deaths_array = np.reshape(deaths_array, (deaths_array.shape[0], 1))
    deaths_array = deaths_array.astype(float)
    #new_deaths = get_new_values(deaths_array, days_predict, days_window, train_model)
    new_deaths = get_new_values(ts_model, deaths_array, days_predict)
    
    #5 get cases_14_100k
    ft_100k_array = X_sorted[:, 4]
    ft_100k_array = np.reshape(ft_100k_array, (ft_100k_array.shape[0], 1))
    ft_100k_array = ft_100k_array.astype(float)
    #new_14_100k = get_new_values(ft_100k_array, days_predict, days_window, train_model)
    new_14_100k = get_new_values(ts_model, ft_100k_array, days_predict)
    
    #6 get cases_100k
    hk_array = X_sorted[:, 5]
    hk_array = np.reshape(hk_array, (hk_array.shape[0], 1))
    hk_array = hk_array.astype(float)
    #new_100k = get_new_values(hk_array, days_predict, days_window, train_model)
    new_100k = get_new_values(ts_model, hk_array, days_predict)
    
    #merge the new arrays into dataframe
    ctry_new_df = pd.DataFrame({'country_id': countries, 'date': new_dates, 'cases': new_cases, 'deaths': new_deaths, 'cases_14_100k': new_14_100k, 'cases_100k': new_100k})
    
    return ctry_new_df

def get_countries(country, days_predict=5):
    a=[]
    for i in range(1, (days_predict+1)):
        a=np.append(a, country)
    
    return a

def get_new_dates(date, days_predict=5):
    date_1 = datetime.strptime(date, "%m/%d/%Y")
    a=[]
    for i in range(1, (days_predict+1)):
        new_date = date_1+ timedelta(days=i)
        new_date_str = datetime.strftime(new_date, "%m/%d/%Y")
        a=np.append(a, new_date_str)
        
    return a

#def get_new_values(the_array, days_predict=5, days_window=5, train_model=None):
def get_new_values(ts_model, the_array, days_predict=5):
    X =ts_model.get_tseries_X(the_array, True)
    y =ts_model.get_tseries_Y(the_array)
    ts_model.fit(X,y)
    days_window = ts_model.get_window()-1
    a=[]
    l=len(the_array)
    
    for i in range(1, (days_predict+1)):
        if(l>days_window):
            cur_X = the_array[l-days_window:,]
            new_val = ts_model.predict(cur_X)
            a=np.append(a, new_val)
            the_array=np.append(the_array, new_val)
            l=l+1 #didn't use l=len(the_array) to save resourse. 
        else:
            return "not enough length for days_window"
        
    return a
    
    #X_to_pred = ts_model.get_newX(the_array, days_predict)
    #y_hat = ts_model.predict(X_to_pred)
    
    #for i in range(1, (days_predict+1)):    
    #return y_hat