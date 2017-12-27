# --------------------------------------------------------------#
# from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
# --------------------------------------------------------------#
# 3.3.2 데이터 변환 적용하기 p170
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_blobs
# 단축 메서드와 효율적인 방법 p176
from sklearn.preprocessing import StandardScaler
# 3.3.4 지도 학슴에서 데이터 전처리 효과
from sklearn.svm import SVC
# 평균 0, 분산 1을 갖도록 스케일 조정
from sklearn.preprocessing import StandardScaler

# 3.3 데이터 전처리와 스케일 조정 p169
# 변환 방법
# mglearn.plots.plot_scaling()
# plt.show()

# 3.3.2 데이터 변환 적용하기 p170
cancer = load_breast_cancer()
# X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=1)
# print(X_train.shape)
# print(X_test.shape)
# scaler = MinMaxScaler()
# scaler = MinMaxScaler(copy=True, feature_range=(0,1))
# scaler.fit(X_train)
# 데이터 변환
# x_train_scaled = scaler.transform(X_train)
# 스케일이 조정된 후 데이터셋의 속성을 출력합니다.
# print("변환된 후 크기 : {}".format(x_train_scaled.shape))
# print("스케일 조정 전 특성별 최솟값 : \n {}".format(X_train.min(axis=0)))
# print("스케일 조정 전 특성별 최댓값 : \n {}".format(X_train.max(axis=0)))
# print("스케일 조정 후 특성별 최솟값 : \n {}".format(x_train_scaled.min(axis=0)))
# print("스케일 조정 후 특성별 최댓값 : \n {}".format(x_train_scaled.max(axis=0)))
# 테스트 데이터 변환
# x_test_scaled = scaler.transform(X_test)
# 스케일이 조정된 후 테스트 데이터의 속성을 출력합니다.
# print("스케일 조정 후 특성별 최솟값 : \n {}".format(x_test_scaled.min(axis=0)))
# print("스케일 조정 후 특성별 최댓값 : \n {}".format(x_test_scaled.max(axis=0)))


# 인위적인 데이터셋 생성
# X, _ = make_blobs(n_samples=50, centers=5, random_state=4, cluster_std=2)
# 훈련 세트와 테스트 세트로 나눕니다.
# X_train, X_test = train_test_split(X, random_state=5, test_size=.1)

# 훈련 세트와 테스트 세트로 나눕니다.
# fig, axes = plt.subplots(1, 3, figsize=(13,4))
# axes[0].scatter(X_train[:, 0], X_train[:, 1], c=mglearn.cm2(0),label="train set", s=60)
# axes[0].scatter(X_test[:, 0], X_test[:, 1], marker='^', c=mglearn.cm2(1),label="test set", s=60)
# axes[0].legend(loc='upper left')
# axes[0].set_title("raw data")

# MinMaxScaler를 사용해 스케일을 조정합니다.
# scaler = MinMaxScaler()
# scaler.fit(X_train)
# X_train_scaled = scaler.transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# 스케일이 조정된 데이터의 산점도를 그립니다.
# axes[1].scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=mglearn.cm2(0), label="train set", s=60)
# axes[1].scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], marker='^', c=mglearn.cm2(1), label="test set", s=60)
# axes[1].set_title("scaled data")

# 테스트 세트의 스케일을 따로 조정합니다.
# 테스트 세트의 최솟값은 0, 최댓값은 1이 됩니다.
# 이는 예제를 위한 것으로, 절대로 이렇게 사용해서는 안됩니다.
# test_scaler = MinMaxScaler()
# test_scaler.fit(X_test)
# X_test_scaled_badly = test_scaler.transform(X_test)
# 잘못 조정된 데이터의 산점도를 그립니다.
# axes[2].scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=mglearn.cm2(0), label="train set", s=60)
# axes[2].scatter(X_test_scaled_badly[:, 0], X_test_scaled_badly[:, 1], marker='^', c=mglearn.cm2(1), label="test set", s=60)
# axes[2].set_title("bad scaled date")
# for ax in axes :
#     ax.set_xlabel("att 0")
#     ax.set_ylabel("att 1")
# plt.show()

# 단축 메서드와 효율적인 방법 p176
# scaler = StandardScaler()
# 메서드 체이닝(chainig)을 사용하여 fit과 transform을 연달아 호출합니다.
# X_scaled = scaler.fit(X_train).transform(X_train)
# 위와 동일하지만 더 효율적입니다.
# X_scaled_d = scaler.fit_transform(X_train)

# 3.3.4 지도 학슴에서 데이터 전처리 효과
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)
svm = SVC(C=100)
svm.fit(X_train, y_train)
print("test set {:.2f}".format(svm.score(X_test, y_test)))

# 0~1 사이로 스케일 조정
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
# 조정된 데이터로 SVM 학습
svm.fit(X_train_scaled, y_train)
# 스케일 조정된 테스트 세트의 정확도
print("scaled test set {:.2f}".format(svm.score(X_test_scaled, y_test)))

# 평균 0, 분산 1을 갖도록 스케일 조정
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled  = scaler.transform(X_test)
# 조정된 데이터로 SVM 학습
svm.fit(X_train_scaled, y_train)
# 스케일 조정된 테스트 세트의 정확도
print("SVM test accurancy : {:.2f}".format(svm.score(X_test_scaled, y_test)))
