import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

dates = []
prices = []

def fetch_data(filename):
    with open(filename, 'r') as csvfile: # opens the csv file
        reader = csv.reader(csvfile) # returns a iterable reader object
        next(reader) # skips the first row

        for row in reader: # iterating through every row to add data to lists
            dates.append(int(row[0].split(' ')[0].split('/')[1]))
            prices.append(float(row[1]))
        return

def predict_prices(dates, prices, x):
    dates = np.reshape(dates, (len(dates), 1)) # reshaping dates list into a n by 1 matrix

    svr_lin = SVR(kernel='linear', C=1e3)
    svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)

    svr_lin.fit(dates, prices)
    svr_poly.fit(dates, prices)
    svr_rbf.fit(dates, prices)

    plt.scatter(dates, prices, color='black', label='Data')
    plt.plot(dates, svr_lin.predict(dates), color='green', label='Linear Model')
    plt.plot(dates, svr_poly.predict(dates), color='blue', label='Polynomial Model')
    plt.plot(dates, svr_rbf.predict(dates), color='red', label='RBF Model')

    plt.xlabel('Date')
    plt.xlabel('Price')
    plt.title('Support Vector Regression')

    plt.legend()
    plt.show()

    return svr_lin.predict(x)[0], svr_poly.predict(x)[0], svr_rbf.predict(x)[0]

fetch_data('AAPL_Data.csv')
predicted_prices = predict_prices(dates, prices, 30)