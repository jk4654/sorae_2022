import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# import matplotlib.pyplot as plt
#독립변수에 ‘오답률이 60% 이상 70% 미만인 문제의 수’를 추가한 모형(2등급컷)

df = pd.read_csv("compiler1\mydata.csv")

x = df[['max_wrong_rate','above70','nth','60to70']]
y = df[['2cut']]

scores=[]
predicts=[]

for i in range(100):
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2)


    mlr = LinearRegression()
    mlr.fit(x_train, y_train) 
    input = [[76.8,2,4,4]]  #예측을 할 데이터 입력
    my_predict = mlr.predict(input) #예측값
    y_predict = mlr.predict(x_test)

    # print(mlr.score(x_train, y_train))
    scores.append(mlr.score(x_train, y_train)) 
    predicts.append(my_predict[0][0])

    # print(mlr.coef_)
from statistics import mean 
print("평균 결정계수: %s"%(mean(scores)))
print("평균 예측값: %s"%(round(mean(predicts))))

# plt.scatter(df[['nth']], df[['1cut']], alpha=0.4)
# plt.xlabel('above70')
# plt.ylabel("1cut")
# plt.title("MULTIPLE LINEAR REGRESSION")
# plt.show()



# fig = plt.figure()
# ax = fig.gca(projection='3d')


# ax.set_xlabel('max_wrong_rate')
# ax.set_ylabel('above70')
# ax.set_zlabel('1cut')
# plt.suptitle('Takeoff distance prediction', fontsize=16)
# plt.show()