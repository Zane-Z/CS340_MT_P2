from numpy.linalg import solve
import numpy as np  # this comes with Anaconda
import pandas as pd  # this comes with Anaconda
# --------------------------Kay Part----------------------------
class TimeSeries:
    def __init__(self, days_window=5, train_model=None):
        self.days_window = days_window
        self.train_model = train_model

    #def get_tseries_X(self, X, window_length=5, preapp_one=True):
    def get_tseries_X(self, X, preapp_one=True):
        window_length=self.days_window, 
        x_shape = X.shape
        num_row = x_shape[0]
        num_col = x_shape[1]
        new_dim = num_row - window_length

        if (num_col == 1):
            new_max = np.zeros((new_dim, window_length-1))

            for i in range(0, new_dim):
                new_max[i] = X[i:(i + window_length-1), 0]
            if (preapp_one == True):
                new_max = np.hstack((np.ones((new_dim, 1)), new_max))

            return new_max

        elif (num_col > 1):
            return "invalid input"
        else:
            return "invalid input"

    #def get_tseries_Y(self, Y, window_length=5):
    def get_tseries_Y(self, Y):
        window_length = self.days_window
        new_y = Y[window_length:,]
        return new_y
    
    def get_window(self):
        return self.days_window
    #def get_newX(self, the_array, days_predict):
        

    def fit (self, X,y):
        if(self.train_model==None):
            self.w = solve(X.T @ X, X.T @ y)
        else:
            self.w = self.train_model.fit(X, y)


    def predict(self, X):
        if(self.train_model==None):
            return X @ self.w
        else:
            return self.train_model.predict(X)

