from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import mglearn
from scipy import sparse
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

#iris dataset load
iris_dataset = load_iris()
#print("Iris_dataset의 키 : \n{}".format(iris_dataset.keys()))
#print(iris_dataset['DESCR']+ "\n...")
#print("target_names : \n{}".format(iris_dataset['target_names']))
#print("feature_names : \n{}".format(iris_dataset['feature_names']))
#print("target의 타입 : \n{}".format(type(iris_dataset['target'])))
#print("data의 크기  : \n{}".format(iris_dataset['data'].shape))
#print("data의 6 행  : \n{}".format(iris_dataset['data'][:6]))
#print("target의 타입 : \n{}".format(iris_dataset['target'].shape))
#print("target의 타입 : \n{}".format(iris_dataset['target']))

#1.7.2 훈련데이터와 테스트데이터
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)
#print("X_train 크기 : {}".format(X_train.shape))
#print("y_train 크기 : {}".format(y_train.shape))
#print("X_test 크기 : {}".format(X_test.shape))
#print("y_test 크기 : {}".format(y_test.shape))

# 열의 이름은 iris_dataset.feature_names에 있는 문자열(4개)을 사용
#iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
# 데이터 프레임을 사용해 y_train에 다라 색으로 구분된 산점도 행렬을 만듭니다.
#pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15,15), marker='o', hist_kwds={'bins':20}, s=60, alpha=.8)
#plt.show()

#1.7.4 K-Nearest Neighbors 알고리즘
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

KNeighborsClassifier(algorithm = 'auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=1, n_neighbors=1, p=2,
                     weights='uniform')
#1.7.5 test date로 예측하기
#X_new = np.array([[5, 2.9, 1, 0.2]])
#print("X_new.shape : {}".format(X_new.shape))

#prediction = knn.predict(X_new)
#print("예측: {}".format(prediction))
#print("예측한 타겟의 이름 : {}".format(iris_dataset['target_names'][prediction]))

#1.7.6 모델 평가하기
y_pred = knn.predict(X_test)
print("테스트 세트에 대한 예측값 :\n {}".format(y_pred))
print("훈련   세트에 대한 값 :\n {}".format(y_test))
print("테스트 세트의 정확도 : {:.2f}".format(np.mean(y_pred == y_test)))
print("테스트 세트의 정확도 : {:.2f}".format(knn.score(X_test, y_test)))
