import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

customers=pd.read_csv('Ecommerce Customers')
print(customers.head())
print(customers.describe())
print(customers.info())

x=customers[['Length of Membership','Time on Website', 'Time on App','Avg. Session Length']]
y=customers['Yearly Amount Spent']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)

pre=lm.predict(X_test)
print(pre)
plt.scatter(y_test,pre)
plt.xlabel('y_test values')
plt.ylabel('predicted values')
from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, pre))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, pre))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, pre)))

coeffecients = pd.DataFrame(lm.coef_,x.columns)
coeffecients.columns = ['Coeffecient']
print(coeffecients)

