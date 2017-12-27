# -------------------------------------------------------------- #
# from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
# -------------------------------------------------------------- #
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# 6.1 데이터 전처리와 매개변수 선택
from sklearn.model_selection import GridSearchCV
# 6.2 파이프라인 구축하기 p373
from sklearn.pipeline import Pipeline
# 6.4.1 make_pipeline을 사용한 파이프라인 생성
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# 6.4.3 그리드 서치 안의 파이프라인 속성에 접근하기
from sklearn.linear_model import LogisticRegression, Ridge
# 6.5 전처리와 모델의 매개변수를 위한 그리드서치
from sklearn.datasets import load_boston
from sklearn.preprocessing import PolynomialFeatures
# 6.6 모델 선택을 위한 그리드 서치
from sklearn.ensemble import RandomForestClassifier
# -------------------------------------------------------------- #

# 데이터 적재와 분할
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)
# 훈련 데이터의 최솟값, 최댓값을 계산합니다.
scaler = MinMaxScaler().fit(X_train)
# 훈련 데이터의 스케일을 조정합니다.
X_train_scaled = scaler.transform(X_train)
svm = SVC()
# 스케일 조정된 훈련 데이터에 SVM을 학습시킵니다.
svm.fit(X_train_scaled, y_train)
# 테스트 데이터의 스케일을 조정하고 점수를 계산합니다.
X_test_scaled = scaler.transform(X_test)
# print("테스트 점수 : {:.2f}".format(svm.score(X_test_scaled, y_test)))

# 6.1 데이터 전처리와 매개변수 선택
# 이 코드는 예를 위한 것입니다. 실제로 사용하지 마세요.
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
              'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=5)
grid.fit(X_train_scaled, y_train)
# print("최상의 교차 검증 정확도 : {}".format(grid.best_score_))
# print("테스트 세트 점수 : {}".format(grid.score(X_test_scaled, y_test)))
# print("최적의 매개변수 : ",grid.best_params_)
# mglearn.plots.plot_improper_processing()
# plt.show()

# 6.2 파이프라인 구축하기 p373
# pipe = Pipeline([("scaler", MinMaxScaler()), ("svm", SVC())])
# pipe.fit(X_train, y_train)
# print("테스트 점수 {}".format(pipe.score(X_test, y_test)))

# 6.3 그리드 서치에 파이프라인 적용하기
# param_grid = {'svm__C' : [0.001, 0.01, 0.1, 1, 10, 100],   'svm__gamma' : [0.001, 0.01, 0.1, 1, 10, 100]}
# grid = GridSearchCV(pipe, param_grid=param_grid, cv=5)
# grid.fit(X_train, y_train)
# print("최상의 교차 검증 정확도 : {:.2f}".format(grid.best_score_))
# print("테스트 세트 점수 : {:.2f}".format(grid.score(X_test, y_test)))
# print("최적의 매개변수 : {}".format(grid.best_params_))

# 6.4 파이프라인 인터페이스
def fit(self, X, y):
    X_transformed = X
    for name, estimator in self.steps[:-1]:
        # 마지막 단계를 빼고 fit과 transform을 반복합니다.
        X_transformed = estimator.fit_transform(X_transformed, y)
    # 마지막 단계 fit을 호출합니다.
    self.steps[-1][1].fit(X_transformed, y)
    return  self

def predict(self, X):
    X_transformed = X
    for step in self.steps[:-1]:
        # 마지막 단계를 빼고 transform을 반복합니다.
        X_transformed = step[1].transform(X_transformed)
    # 마지막 단계에서 predict를 호출합니다.
    return self.steps[-1][1].predict(X_transformed)

# 6.4.1 make_pipeline을 사용한 파이프라인 생성
# 표준적인 방법
# pipe_long = Pipeline([("scaler", MinMaxScaler()), ("svm", SVC(C=100))])
# 간소화된 방법
# pipe_short = make_pipeline(MinMaxScaler(), SVC(C=100))
# print("파이프라인 단계 :\n{}".format(pipe_short.steps))
# pipe = make_pipeline(StandardScaler(), PCA(n_components=2), StandardScaler())
# print("파이프라인 단계 {}".format(pipe.steps))

# 6.4.2 단계 속성 접근하기
# cancer 데이터셋에 앞서 만든 파이프라인을 적용합니다.
# pipe.fit(cancer.data)
# "pca" 단계의 두 개 주성분을 추출합니다.
# components = pipe.named_steps["pca"].components_
# print("components : {}".format(components))
# print("components.steps : {}".format(components.shape))

# 6.4.3 그리드 서치 안의 파이프라인 속성에 접근하기
# pipe = make_pipeline(StandardScaler(), LogisticRegression())
# param_grid = {'logisticregression__C': [0.01, 0.1, 1, 10, 100]}
# X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=4)
# grid = GridSearchCV(pipe, param_grid, cv=5)
# grid.fit(X_train, y_train)
# print("최상의 모델 : \n{}".format(grid.best_estimator_))
# print("로지스틱 회귀 단계 :\n{}".format(grid.best_estimator_.named_steps["logisticregression"]))
# print("로지스틱 회귀 계수 :\n{}".format(grid.best_estimator_.named_steps["logisticregression"].coef_))

# 6.5 전처리와 모델의 매개변수를 위한 그리드서치
boston = load_boston()
# X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=0)
# pipe = make_pipeline(StandardScaler(), PolynomialFeatures(), Ridge())
# param_grid = {'polynomialfeatures__degree' : [1,2,3], 'ridge__alpha'  : [0.001, 0.01, 0.1, 1, 10, 100]}
# grid = GridSearchCV(pipe, param_grid=param_grid, cv=5)
# grid.fit(X_train, y_train)
# mglearn.tools.heatmap(grid.cv_results_['mean_test_score'].reshape(3, -1), xlabel="ridge__alpha", ylabel="polynomialfeatures__degree", xticklabels=param_grid['ridge__alpha'], yticklabels=['polynomialfeatures__degree'], vmin=0)
# plt.show()

# 6.6 모델 선택을 위한 그리드 서치
pipe = Pipeline([('preprocessing', StandardScaler()), ('classifier', SVC())])
param_grid = [
    {'classifier' : [SVC()],
     'preprocessing' : [StandardScaler()],
     'classifier__gamma' : [0.001, 0.01, 0.1, 1, 10, 100],
     'classifier__C' : [0.001, 0.01, 0.1, 1, 10, 100]},
    {'classifier' : [RandomForestClassifier(n_estimators=100)],
     'preprocessing' : [None], 'classifier__max_features':[1, 2, 3]}]
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)
grid = GridSearchCV(pipe, param_grid=param_grid, cv=5)
grid.fit(X_train, y_train)
print("최적의 매개변수 : \n{}\n".format(grid.best_params_))
print("최상의 교차 점수 : {:.2f}".format(grid.best_score_))
print("테스트 점수 : {:.2f}".format(grid.score(X_test, y_test)))
