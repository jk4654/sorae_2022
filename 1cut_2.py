import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
#재수생이 포함되지 않은 3월 제외
#6월, 9월, 수능 데이터로만 분석

df = pd.read_csv("compiler1\mydata_b.csv")

x = df[['max_wrong_rate','above70','nth']]
y = df[['1cut']]


scores=[]
predicts=[]

for i in range(100):
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, test_size=0.1)
    mlr = LinearRegression()
    mlr.fit(x_train, y_train) 
    my_apartment = [[76.8,2,4]]
    my_predict = mlr.predict(my_apartment)
    y_predict = mlr.predict(x_test)
    # print("예측값:" + str(my_predict))
    # print(mlr.coef_)
    # print(mlr.score(x_train, y_train))

    scores.append(mlr.score(x_train, y_train)) 
    predicts.append(my_predict[0][0])

from statistics import mean 
print("평균 결정계수: %s"%(mean(scores)))
print("평균 예측값: %s"%(round(mean(predicts))))


# # plt.scatter(df[['max_wrong_rate']], df[['1cut']], alpha=0.4)
# plt.scatter(df[['nth']], df[['1cut']], alpha=0.4)

# plt.xlabel('above70')
# plt.ylabel("1cut")
# plt.title("MULTIPLE LINEAR REGRESSION")
# plt.show()