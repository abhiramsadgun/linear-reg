import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df=pd.read_csv('height-weight.csv')
print(df.head())
plt.scatter(df['Weight'],df['Height'])
plt.title("Height vs Weight")
plt.xlabel("Weight")
plt.ylabel("Height")
plt.show()
#print(df.corr())
print(df.corr(method='spearman'))

import seaborn as sb
sb.pairplot(df)
plt.show()

#Weight is an independent feature it should be a dataframe
X=df[['Weight']] 
#type(X)
Y=df[['Height']]
type(Y)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=42)

from sklearn.preprocessing import StandardScaler
scaler =StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

from sklearn.linear_model import LinearRegression
regression=LinearRegression(n_jobs=-1)
regression.fit(X_train,Y_train)
print("Coefficient ",regression.coef_)
print("Intercept ",regression.intercept_)
plt.scatter(X_train,Y_train)
plt.plot(X_train,regression.predict(X_train))
plt.show()

Y_pre=regression.predict(X_test)

from sklearn.metrics import mean_absolute_error,mean_squared_error
mse=mean_squared_error(Y_test,Y_pre)
mae=mean_absolute_error(Y_test,Y_pre)
rmse=np.sqrt(mse)
from sklearn.metrics import r2_score
r2=r2_score(Y_test,Y_pre)
print("MSE : ",mse)
print("MAE : ",mae)
print("RMSE : ",rmse)
print("R2 Score : ",r2)

import statsmodels.api as s
model=s.OLS(Y_train,X_train).fit()
prediction=model.predict(X_test)
print(prediction)
print(model.summary())
j=float(input("Enter value for prediction"))
print(regression.predict(scaler.transform([[j]])))

