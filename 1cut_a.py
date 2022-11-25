import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv("예제1\myproject_a.csv")
df.head()

x = df[['max_wrong_rate','above70','nth']]
y = df[['1cut']]
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.6, test_size=0.4)


mlr = LinearRegression()
mlr.fit(x_train, y_train) 
my_apartment = [[76.8,2,4]]
my_predict = mlr.predict(my_apartment)
y_predict = mlr.predict(x_test)
print(my_predict)
print(mlr.coef_)
print(mlr.score(x_train, y_train))

# plt.scatter(df[['max_wrong_rate']], df[['1cut']], alpha=0.4)
plt.scatter(df[['nth']], df[['1cut']], alpha=0.4)

plt.xlabel('above70')
plt.ylabel("1cut")
plt.title("MULTIPLE LINEAR REGRESSION")
plt.show()