from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from scipy import sparse
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_boston
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
# 릿지 회귀
from sklearn.linear_model import Ridge
# 라쏘
from sklearn.linear_model import Lasso
# 분류용 선형 모델
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
# 다중 클래스 분류용 선형 모델
from sklearn.datasets import make_blobs
# 2.3.5 결정트리
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import graphviz
# 랜덤 포레스트 115p
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons
# 그래디언트 부스팅 회귀 트리 p122
from sklearn.ensemble import GradientBoostingClassifier
# 2.3.7 커널 서포트 벡터 머신 p126
from mpl_toolkits.mplot3d import Axes3D, axes3d
# 커널 기법 p130
from sklearn.svm import SVC
# 신경망 튜닝 p142
from sklearn.neural_network import MLPClassifier
# from sklearn.datasets import make_moons
# 분류 예측의 불확실성 추정 p153
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_circles

#2.3.1
# 데이터셋을 만듭니다.
#X, y = mglearn.datasets.make_forge()
# 산점도를 그립니다.
#mglearn.discrete_scatter(X[:,0], X[:,1],y)
#plt.legend(["A class 0", "B class 1"], loc=4)
#plt.xlabel("첫 번째 특성")
#plt.ylabel("두 번째 특성")
#print("X.Shape : {}".format(X.shape))
#plt.show()

#회귀알고리즘 make_wave 사용
#X, y = mglearn.datasets.make_wave(n_samples=40)
#plt.plot(X, y, 'o')
#plt.ylim(-3, 3)
#plt.xlabel("att")
#plt.ylabel("target")
#plt.show()

#cancer dataset
#cancer = load_breast_cancer()
#print("cancer.key() :\n{}".format(cancer.keys()))
#print("유방암 데이터의 형태 : {}".format(cancer.data.shape))
#print("클래스별 샘플 개수:\n{}".format({n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}))
#print("특성 이름 : \n{}".format(cancer.feature_names))

#회귀 분석용 보스턴 주택가격
#boston = load_boston()
#print("데이터의 형태 : {}".format(boston.data.shape))
#X, y = mglearn.datasets.load_extended_boston()
#print("X.shape : {}".format(X.shape))
#n_neighbors 값을 변화시켜보며 테스트 데이터 분류가 달라지는 것을 확인
#mglearn.plots.plot_knn_classification(n_neighbors=3)
#plt.show()

#X, y = mglearn.datasets.make_forge()
#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
#clf = KNeighborsClassifier(n_neighbors=3)
#clf.fit(X_train, y_train)
#print("테스트 세트 예측 : {}".format(clf.predict(X_test)))
#print("훈련   세트 예측 : {}".format(clf.predict(X_train)))
#print("테스트 세트 정확도 : {:.2f}".format(clf.score(X_test, y_test)))

#KNeighborsClassifier 분석
#fig, axes = plt.subplots(1, 3, figsize=(10,3))
#for n_neighbors, ax in zip([1,3,9], axes):
    # fit 메서드는 self 객체를 반환합니다.
    # 그래서 객체 생성과 fit 메서드를 한 줄에 쓸 수 있습니다.
#    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X,y)
#    mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
#    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
#    ax.set_title("{} neighbor".format(n_neighbors))
#    ax.set_xlabel("att 0")
#    ax.set_ylabel("att 1")
#axes[0].legend(loc=3)
#plt.show()

#유방암 데이터셋
#cancer = load_breast_cancer()
#X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify = cancer.target, random_state=66)
#training_accuracy = []
#test_accuracy = []
#1에서 10까지 n_neighbors를 적용
#neighbors_settings = range(1,11)

#for n_neighbors in neighbors_settings:
    #모델생성
#    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
#    clf.fit(X_train, y_train)
    #훈련 세트 정확도 저장
#    training_accuracy.append(clf.score(X_train, y_train))
    #일반화 정확도 저장
#    test_accuracy.append(clf.score(X_test, y_test))

#plt.plot(neighbors_settings, training_accuracy, label="train accuracy")
#plt.plot(neighbors_settings, test_accuracy,     label="test accuracy")
#plt.ylabel("accuracy")
#plt.xlabel("n_neighbors")
#plt.legend()
#plt.show()

#K-최근접 이웃 회귀
#mglearn.plots.plot_knn_regression(n_neighbors=3)
#plt.show()

#회귀 dataset sample 40개 make_wave
# X, y = mglearn.datasets.make_wave(n_samples=40)
# wave 데이터셋을 훈련세트와 테스트 세트로 나눕니다.
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# 이웃의 수를 3으로 하여 모델의 객체를 만듭니다.
# reg = KNeighborsRegressor(n_neighbors=3)
# 훈련 데이터와 타깃을 사용하여 모델을 학습시킵니다.
# reg.fit(X_train, y_train)
# print("테스트 세트 예측 :\n{}".format(reg.predict(X_test)))
# print("테스트 세트 R^2: {:.2f}".format(reg.score(X_test, y_test)))

#KNeighborsRegressor 분석 72p
# fig, axes = plt.subplots(1, 3, figsize=(15,4))
# # -3과 3 사이에 1,000개의 데이터 포인트를 만듭니다.
# line = np.linspace(-3, 3, 1000).reshape(-1, 1)
# for n_neighbors, ax in zip([1,3,9], axes):
#     # 1, 3, 9 이웃을 사용한 예측을 합니다.
#     reg = KNeighborsRegressor(n_neighbors=n_neighbors)
#     reg.fit(X_train, y_train)
#     ax.plot(line, reg.predict(line))
#     ax.plot(X_train, y_train, '^', c=mglearn.cm2(0), markersize=8)
#     ax.plot(X_test,  y_test,  'v', c=mglearn.cm2(1), markersize=8)
#
#     ax.set_title("{} train score : {:.2f} test score : {:.2f}".format(n_neighbors,  reg.score(X_train, y_train), reg.score(X_test, y_test)))
#     ax.set_xlabel("att")
#     ax.set_ylabel("target")
# axes[0].legend(["model", "train", "test"], loc="best")
# plt.show()

# 2.3.3 선형모델 73p
#회귀의 선형 모델
# mglearn.plots.plot_linear_regression_wave()
# plt.show()

# 선형 회귀(최소제곱법) 76p, 과소적합
# X, y = mglearn.datasets.make_wave(n_samples=60)
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
# # 선형 회귀(최소제곱법) 적용
# lr = LinearRegression().fit(X_train, y_train)
# print("lr.coef_ : {}",format(lr.coef_))
# print("lr.intercept_: {}".format(lr.intercept_))
# print("train score {:.2f}".format(lr.score(X_train, y_train)))
# print("test  score {:.2f}".format(lr.score(X_test, y_test )))

# 선형 회귀(최소제곱법) 77p, 과대적합
# X, y = mglearn.datasets.load_extended_boston()
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# lr = LinearRegression().fit(X_train, y_train)
# print("train score : {:.2f}".format(lr.score(X_train, y_train)))
# print("test  score : {:.2f}".format(lr.score(X_test,  y_test )))

# 릿지 회귀
# ridge = Ridge().fit(X_train, y_train)
# print("train score : {:.2f}".format(ridge.score(X_train, y_train)))
# print("test  score : {:.2f}".format(ridge.score(X_test,  y_test )))
# ridge10 = Ridge(alpha=10).fit(X_train, y_train)
# print("train score : {:.2f}".format(ridge10.score(X_train, y_train)))
# print("test  score : {:.2f}".format(ridge10.score(X_test,  y_test )))
# ridge01 = Ridge(alpha=0.1).fit(X_train, y_train)
# print("train score : {:.2f}".format(ridge01.score(X_train, y_train)))
# print("test  score : {:.2f}".format(ridge01.score(X_test,  y_test )))

# plt.plot(ridge10.coef_, '^', label="Ridge alpha=10")
# plt.plot(ridge.coef_,   's', label="Ridge alpha=1")
# plt.plot(ridge.coef_,   'v', label="Ridge alpha=0.1")
# plt.plot(lr.coef_,      'o', label="LinearRegression")
# plt.xlabel("menu")
# plt.ylabel("size")
# plt.hlines(0, 0, len(lr.coef_))
# plt.ylim(-25, 25)
# plt.legend()
# plt.show()

# 학습 곡선 p81
# mglearn.plots.plot_ridge_n_samples()
# plt.show()

# 라쏘 p83
# lasso = Lasso().fit(X_train, y_train)
# print("train  score : {:.2f}".format(lasso.score(X_train, y_train)))
# print("test   score : {:.2f}".format(lasso.score(X_test,  y_test )))
# print("att    num   : {}"     .format(np.sum(lasso.coef_ != 0)))
# print("att    num   : {}"     .format(np.sum(lasso.coef_ == 0)))
# "max_iter" 기본값을 증가시키지 않으면 max_iter 값을 늘리라는 경고가 발생합니다.
# lasso001 = Lasso(alpha=0.01, max_iter=100000).fit(X_train, y_train)
# print("train  score : {:.2f}".format(lasso001.score(X_train, y_train)))
# print("test   score : {:.2f}".format(lasso001.score(X_test,  y_test )))
# print("att    num   : {}"     .format(np.sum(lasso001.coef_ != 0)))
# lasso00001 = Lasso(alpha=0.0001, max_iter=100000).fit(X_train, y_train)
# print("train  score : {:.2f}".format(lasso00001.score(X_train, y_train)))
# print("test   score : {:.2f}".format(lasso00001.score(X_test,  y_test )))
# print("att    num   : {}"     .format(np.sum(lasso00001.coef_ != 0)))
# plt.plot(lasso.coef_,       's',    label="Lasso alpha=1")
# plt.plot(lasso001.coef_,    '^',    label="Lasso alpha=0.01")
# plt.plot(lasso00001.coef_,  'v',    label="Lasso alpha=0.0001")
# plt.plot(ridge01.coef_,     'o',    label="Ridge alpha=0.1")
# plt.legend(ncol=2, loc=(0,1.05))
# plt.ylim(-25,25)
# plt.xlabel("menu")
# plt.ylabel("size")
# plt.show()

# 분류용 선형 모델 p87
# X, y = mglearn.datasets.make_forge()
# fig, axes = plt.subplots(1, 2, figsize=(10, 3))
# for model, ax in zip([LinearSVC(), LogisticRegression()], axes):
#     clf = model.fit(X, y)
#     mglearn.plots.plot_2d_separator(clf, X, fill=False, eps=0.5, ax=ax, alpha=.7)
#     mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
#     ax.set_title("{}".format(clf.__class__.__name__))
#     ax.set_xlabel("att 0")
#     ax.set_ylabel("att 1")
# axes[0].legend()
# plt.show()

# mglearn.plots.plot_linear_svc_regularization()
# plt.show()

# 유방암 p89
# cancer = load_breast_cancer()
# X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)
# logreg = LogisticRegression().fit(X_train, y_train)
# print("train score  :   {:.3f}".format(logreg.score(X_train, y_train)))
# print("test  score  :   {:.3f}".format(logreg.score(X_test,  y_test )))
# logreg100 = LogisticRegression(C=100).fit(X_train, y_train)
# print("train score  :   {:.3f}".format(logreg100.score(X_train, y_train)))
# print("test  score  :   {:.3f}".format(logreg100.score(X_test,  y_test )))
# logreg001 = LogisticRegression(C=0.01).fit(X_train, y_train)
# print("train score  :   {:.3f}".format(logreg001.score(X_train, y_train)))
# print("test  score  :   {:.3f}".format(logreg001.score(X_test,  y_test )))

# plt.plot(logreg.coef_.T,    'o',    label="C=1")
# plt.plot(logreg100.coef_.T, '^',    label="C=100")
# plt.plot(logreg001.coef_.T, 'v',    label="C=0.001")
# plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
# plt.hlines(0, 0, cancer.data.shape[1])
# plt.ylim(-5, 5)
# plt.xlabel("att")
# plt.ylabel("size")
# plt.legend()
# plt.show()

# 로지스틱 회귀
# for C, marker in zip([0.001, 1, 100], ['o', '^', 'v']):
#     lr_l1 = LogisticRegression(C=C, penalty="l1").fit(X_train, y_train)
#     print("C={:.3f}인 l1 로지스틱 회귀의 훈련 정확도 : {:.2f}".format(C, lr_l1.score(X_train, y_train)))
#     print("C={:.3f}인 l1 로지스틱 회귀의 테스트 정확도 : {:.2f}".format(C, lr_l1.score(X_test, y_test)))
#     plt.plot(lr_l1.coef_.T, marker, label="C={:.3f}".format(C))
# plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
# plt.hlines(0, 0, cancer.data.shape[1])
# plt.xlabel("att")
# plt.ylabel("size")
# plt.ylim(-5, 5)
# plt.legend(loc=3)
# plt.show()

# 다중 클래스 분류용 선형 모델 p94
# X, y = make_blobs(random_state=42)
# mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
# plt.xlabel("att 0")
# plt.ylabel("att 1")
# plt.legend(["class 0", "class 1", "class 2"])
# plt.show()
# linear_svm = LinearSVC().fit(X, y)
# print("계수 배열의 크기 : ", linear_svm.coef_.shape)
# print("절편 배열의 크기 : ", linear_svm.intercept_.shape)
# mglearn.discrete_scatter(X[:, 0], X[:, 1],y)
# line = np.linspace(-15, 15)
# for coef, intercept, color in zip(linear_svm.coef_, linear_svm.intercept_, mglearn.cm3.colors):
#     plt.plot(line, -(line * coef[0] + intercept) / coef[1], c=color)
# plt.ylim(-10, 15)
# plt.xlim(-10, 8)
# plt.xlabel("att 0")
# plt.ylabel("att 1")
# plt.legend(['class 0', 'class 1', 'class 2', 'class 0 line', 'class 1 line', 'class 2 line'])
# plt.show()

# mglearn.plots.plot_2d_classification(linear_svm, X, fill=True, alpha=.7)
# mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
# line = np.linspace(-15, 15)
# for coef, intercept, color in zip(linear_svm.coef_, linear_svm.intercept_, mglearn.cm3.colors):
#      plt.plot(line, -(line * coef[0] + intercept) / coef[1], c=color)
#      print("coef[0] : ", coef[0])
#      print("coef[1] : ", coef[1])
#      print("intercept ", intercept)
# plt.legend(['class 0', 'class 1', 'class 2', 'class 0 line', 'class 1 line', 'class 2 line'])
# plt.xlabel("att 0")
# plt.ylabel("att 1")
# plt.show()

# 2.3.4 나이브 베이즈 분류기 p99
# X = np.array([[0,1,0,1],
#               [1,0,1,0],
#               [0,0,0,1],
#               [1,0,1,0]])
# y = np.array([0,1,0,1])
#
# counts = {}
# for label in np.unique(y):
#     # 클래스 마다 반복
#     # 특성마다 1이 나타난 횟수를 센다.
#     counts[label] = X[y == label].sum(axis=0)
#     print("label : " , label)
#     print("y=label " , y == label)
#     print("X[y == label] ", X[y == label])
# print("특성 카운트 :n{}".format(counts))

# 2.3.5 결정트리 101p
# cancer = load_breast_cancer()
# X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)
# tree = DecisionTreeClassifier(random_state=0)
# tree.fit(X_train, y_train)
# print("train  score : {:.3f}".format(tree.score(X_train, y_train)))
# print("test   score : {:.3f}".format(tree.score(X_test,  y_test )))
# tree = DecisionTreeClassifier(max_depth=4, random_state=0)
# tree.fit(X_train, y_train)
# print("train  score : {:.3f}".format(tree.score(X_train, y_train)))
# print("test   score : {:.3f}".format(tree.score(X_test,  y_test )))
# export_graphviz(tree, out_file="tree.dot", class_names=["bad", "good"], feature_names=cancer.feature_names, impurity=False, filled=True)
# with open("tree.dot") as f:
#     dot_graph = f.read()
# display(graphviz.Source(dot_graph))
# dot = graphviz.Source(dot_graph)
# dot.format = "png"
# dot.render(filename="test")

#################################################################################
## def
#################################################################################
# print("특성 중요도 :\n{}".format(tree.feature_importances_))
# def plot_feature_importances_cancer(model):
#     n_features = cancer.data.shape[1]
#     plt.barh(range(n_features), model.feature_importances_, align='center')
#     plt.yticks(np.arange(n_features), cancer.feature_names)
#     plt.xlabel("att importance")
#     plt.ylabel("att")
#     plt.ylim(-1, n_features)

# plot_feature_importances_cancer(tree)
# plt.show()

# tree = mglearn.plots.plot_tree_not_monotone()
# plt.show()

# RAM 가격 동향
# os가 안됨
# ram_prices = pd.read_csv(os.path.join(mglearn.datasets.DATA_PATH, "ram_price.csv"))
# plt.semilogy(ram_prices.date, ram_prices.price)
# plt.xlabel("year")
# plt.ylabel("price")
# plt.show()

# 랜덤 포레스트
# X, y = make_moons(n_samples=100, noise=0.25, random_state=3)
# X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
# random forest 적용
# forest = RandomForestClassifier(n_estimators=5, random_state=2)
# forest.fit(X_train, y_train)
# fig, axes = plt.subplots(2, 3, figsize=(15, 10))
# for i, (ax, tree) in enumerate(zip(axes.ravel(), forest.estimators_)):
#     ax.set_title("tree {}".format(i))
#     mglearn.plots.plot_tree_partition(X, y, tree, ax=ax)
# random forest 출력
# mglearn.plots.plot_2d_separator(forest, X, fill=True, ax=axes[-1, -1], alpha=.4)
# axes[-1, -1].set_title("random forest")
# mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
# plt.show()

# cancer = load_breast_cancer()
# X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)
# random forest 적용
# forest = RandomForestClassifier(n_estimators=100, random_state=0)
# forest.fit(X_train, y_train)
# print("train  score : {:.3f}".format(forest.score(X_train, y_train)))
# print("test   score : {:.3f}".format(forest.score(X_test,  y_test )))
# plot_feature_importances_cancer(forest)
# plt.show()

# 그래디언트 부스팅 회귀 트리 p122
# cancer = load_breast_cancer()
# X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)
# gbrt = GradientBoostingClassifier(random_state=0)
# gbrt = GradientBoostingClassifier(random_state=0, max_depth=1)
# gbrt = GradientBoostingClassifier(random_state=0, learning_rate=0.01)
# gbrt.fit(X_train, y_train)
# print("train score : {:.3f}".format(gbrt.score(X_train, y_train)))
# print("test  score : {:.3f}".format(gbrt.score(X_test,  y_test )))
# plot_feature_importances_cancer(gbrt)
# plt.show()

# 2.3.7 커널 서포트 벡터 머신 p126
# X, y = make_blobs(centers=4, random_state=8)
# y = y % 2
# mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
# plt.xlabel("att 0")
# plt.ylabel("att 1")
# plt.legend()
# plt.show()

# linear_svm = LinearSVC().fit(X, y)
# mglearn.plots.plot_2d_separator(linear_svm, X)
# mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
# plt.xlabel("att 0")
# plt.ylabel("att 1")
# plt.legend()
# plt.show()

# X_new = np.hstack([X, X[:, 1:] ** 2])
# figure = plt.figure()
# 3차원 그래프
# ax = Axes3D(figure, elev=-152, azim=-26)
# y == 0인 포인트를 먼저 그리고 그다음 y==1 인 포인트를 그립니다.
# mask = y == 0
# ax.scatter(X_new[mask, 0], X_new[mask, 1], X_new[mask, 2], c='b', cmap=mglearn.cm2, s=60, edgecolor='k')
# ax.scatter(X_new[~mask, 0], X_new[~mask, 1], X_new[~mask, 2], c='r', marker='^', cmap=mglearn.cm2, s=60, edgecolor='k')
# ax.set_xlabel("att 0")
# ax.set_ylabel("att 1")
# ax.set_zlabel("att 1 ** 2")
# plt.show()

# linear_svm_3d = LinearSVC().fit(X_new, y)
# coef, intercept = linear_svm_3d.coef_.ravel(), linear_svm_3d.intercept_
# 선형 결정 경계 그리기
# figure = plt.figure()
# ax = Axes3D(figure, elev=-152, azim=-26)
# xx = np.linspace(X_new[:, 0].min() - 2, X_new[:, 0].max() + 2, 50)
# yy = np.linspace(X_new[:, 1].min() - 2, X_new[:, 1].max() + 2, 50)
# XX, YY = np.meshgrid(xx, yy)
# ZZ = (coef[0] * XX + coef[1] * YY + intercept) / -coef[2]
# ax.plot_surface(XX, YY, ZZ, rstride=8, cstride=8, alpha=0.3)
# ax.scatter(X_new[mask, 0], X_new[mask, 1], X_new[mask, 2], c='b', cmap=mglearn.cm2, s=60, edgecolor='k')
# ax.scatter(X_new[~mask, 0], X_new[~mask, 1], X_new[~mask, 2], c='r', marker='^', cmap=mglearn.cm2, s=60, edgecolor='k')
# ax.set_xlabel("att 0")
# ax.set_ylabel("att 1")
# ax.set_zlabel("att 1 ** 2")
# plt.show()

# ZZ = YY ** 2
# dec = linear_svm_3d.decision_function(np.c_[XX.ravel(), YY.ravel(), ZZ.ravel()])
# plt.contour(XX, YY, dec.reshape(XX.shape), levels=[dec.min(), 0, dec.max()],
#             cmap=mglearn.cm2, alpha=0.5)
# mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
# plt.xlabel("att 0")
# plt.ylabel("att 1")
# plt.show()

# 커널 기법 p130
# X, y = mglearn.tools.make_handcrafted_dataset()
# rbf 커널
# svm = SVC(kernel='rbf', C=10, gamma=0.1).fit(X, y)
# mglearn.plots.plot_2d_separator(svm, X, eps=.5)
# mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
# 서포트 벡터
# sv = svm.support_vectors_
# dual_coef_의 부호에 의해 서포트 벡터의 클래스 레이블이 결정됩니다.
# sv_labels = svm.dual_coef_.ravel() > 0
# mglearn.discrete_scatter(sv[:, 0], sv[:, 1], sv_labels, s=15, markeredgewidth=3)
# plt.xlabel("att 0")
# plt.ylabel("att 1")
# plt.show()

# fig, axes = plt.subplots(3, 3, figsize=(15,10))
# for ax, C in zip(axes, [-1, 0, 3]):
#     for a, gamma in zip(ax, range(-1, 2)):
#         mglearn.plots.plot_svm(log_C=C, log_gamma=gamma, ax=a)
# axes[0, 0].legend(["class 0", "class 1", "class 0 support vector", "class 1 support vector"], ncol=4, loc=(.9, 1.2))
# plt.show()

# rbf 커널 유방암 데이터 셋 p134
# cancer = load_breast_cancer()
# X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)
# svc = SVC()
# svc.fit(X_train, y_train)
# print("train score : {:.2f}".format(svc.score(X_train, y_train)))
# print("test  score : {:.2f}".format(svc.score(X_test,  y_test )))
# plt.boxplot(X_train, manage_xticks=False)
# plt.yscale("symlog")
# plt.xlabel("menu")
# plt.ylabel("size")
# plt.show()

# 훈련 세트에서 특성별 최솟값 계산
# min_on_training = X_train.min(axis=0)
# 훈련 세트에서 특성별 (최댓값 - 최솟값) 범위 계산
# range_on_training = (X_train - min_on_training).max(axis=0)
# 훈련 데이터에 최솟값을 빼고 범위로 나누면
# 각 특성에 대해 최솟값은 0, 최댓값은 1입니다.
# X_train_scaled = (X_train - min_on_training) / range_on_training
# print("특성별 최소 값 \n{}".format(X_train_scaled.min(axis=0)))
# print("특성별 최대 값 \n{}".format(X_train_scaled.max(axis=0)))

# 테스트 세트에도 같은 작업을 적용하지만
# 훈련 세트에서 계산한 최솟값과 범위를 사용합니다
# X_test_scaled = (X_test - min_on_training) / range_on_training
# svc = SVC()
# svc.fit(X_train_scaled, y_train)
# print("train score : {:.3f}".format(svc.score(X_train_scaled, y_train)))
# print("test  score : {:.3f}".format(svc.score(X_test_scaled,  y_test)))
# # 매개변수 C=1000
# svc = SVC(C=1000)
# svc.fit(X_train_scaled, y_train)
# print("train score : {:.3f}".format(svc.score(X_train_scaled, y_train)))
# print("test  score : {:.3f}".format(svc.score(X_test_scaled,  y_test)))

# 2.3.8 신경망(딥러닝) p138
# display(mglearn.plots.plot_logistic_regression_graph())
# line = np.linspace(-3, 3, 100)
# plt.plot(line, np.tanh(line), label="tanh")
# plt.plot(line, np.maximum(line, 0), label="relu")
# plt.legend(loc="best")
# plt.xlabel("x")
# plt.ylabel("relu(x), tanh(x)")
# plt.show()

# 신경망 튜닝 p142
# X, y = make_moons(n_samples=100, noise=0.25, random_state=3)
# X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
# mlp = MLPClassifier(solver='lbfgs', random_state=0).fit(X_train, y_train)
# mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3)
# mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
# plt.xlabel("att 0")
# plt.ylabel("att 1")
# plt.show()

# 신경망 튜닝 유방암 data p147
# cancer = load_breast_cancer()
# print("유방암 데이터의 특성별 최댓값:\n{}".format(cancer.data.max(axis=0)))
# X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)
# mlp = MLPClassifier(random_state=42)
# mlp.fit(X_train, y_train)
# print("train score : {:.2f}".format(mlp.score(X_train, y_train)))
# print("test  score : {:.2f}".format(mlp.score(X_test,  y_test )))
# 훈련 세트 각 특성의 평균을 계산합니다.
# mean_on_train = X_train.mean(axis=0)
# 훈련 세트 각 특성의 표준 편차를 계산합니다.
# std_on_train = X_train.std(axis=0)

# 데이터에서 평균을 빼고 표준 편차로 나누면
# 평균 0, 표준 편차 1인 데이터로 변환됩니다.
# X_train_scaled = (X_train - mean_on_train) / std_on_train
# (훈련 데이터의 평균과 표준 편차를 이용해) 같은 변환을 테스트 세트에도 합니다.
# X_test_scaled = (X_test - mean_on_train) / std_on_train
# mlp = MLPClassifier(max_iter=1000, alpha=1, random_state=0)
# mlp.fit(X_train_scaled, y_train)
# print("train score : {:.3f}".format(mlp.score(X_train_scaled, y_train)))
# print("test  score : {:.3f}".format(mlp.score(X_test_scaled,  y_test )))
# plt.figure(figsize=(20,5))
# plt.imshow(mlp.coefs_[0], interpolation='none', cmap='viridis')
# plt.yticks(range(30), cancer.feature_names)
# plt.xlabel("unit")
# plt.ylabel("att")
# plt.colorbar()
# plt.show()

# 분류 예측의 불확실성 추정 p153
X, y = make_circles(noise=0.25, factor=0.5, random_state=1)
# 예제를 위해 클래스의 이름은 "blue"와 "red"로 바꿉니다.
y_named = np.array(["blue", "red"])[y]
# 여러 배열을 한꺼번에 train_test_split에 넣을 수 있습니다.
# 훈련 세트와 테스트 세트로 나뉘는 방식은 모두 같습니다.
X_train, X_test, y_train_named, y_test_named, y_train, y_test = train_test_split(X, y_named, y, random_state=0)
# 그래디언트 부스팅 모델을 만듭니다.
gbrt = GradientBoostingClassifier(random_state=0)
gbrt.fit(X_train, y_train_named)
# 2.4.1 결정 함수
# print("X_test.shape : {}".format(X_test.shape))
# print("결정 함수 결과 형태 : {}".format(gbrt.decision_function(X_test).shape))
# 결정 함수 결과 중 앞부분 일부를 확인합니다.
# print("결정 함수:\n{}".format(gbrt.decision_function(X_test)[:6]))
# decision_function > 0 이면 양성 클래스 (일치)
# print("임계치와 결정 함수 결과 비교:\n{}".format(gbrt.decision_function(X_test) > 0 ))
# print("예측 : \n{}".format(gbrt.predict(X_test)))
# 불리언 값을 0과 1로 반환합니다.
greater_zero = (gbrt.decision_function(X_test) > 0).astype(int)
# print("greater_zero : ", greater_zero)
# classes_에 인덱스로 사용합니다.
pred = gbrt.classes_[greater_zero]
# print("pred : ", pred)
# print("gbrt.predict(X_test) : ", gbrt.predict(X_test))
# pred와 gbrt.predict의 결과를 비교합니다.
# print("pred는 예측 결과와 같다: {}".format(np.all(pred == gbrt.predict(X_test))))
decision_function = gbrt.decision_function(X_test)
# print("결정 함수 최솟값 : {:.2f} 최댓값 : {:.2f}".format(np.min(decision_function), np.max(decision_function)))
fig, axes = plt.subplots(1, 2, figsize=(13,5))
mglearn.tools.plot_2d_separator(gbrt, X, ax=axes[0], alpha=.4, fill=True, cm=mglearn.cm2)
scores_image = mglearn.tools.plot_2d_separator(gbrt, X, ax=axes[1], alpha=.4, cm=mglearn.ReBl)

for ax in axes:
    # 훈련 포인트와 테스트 포인트를 그리기
    mglearn.discrete_scatter(X_test[:, 0], X_test[:, 1], y_test, markers='^', ax=ax)
    mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train, markers='o', ax=ax)
    ax.set_xlabel("att 0")
    ax.set_ylabel("att 1")
# cbar = plt.colorbar(scores_image, ax=axes.tolist())
axes[0].legend(["test class 0", "test class 1", "train class 0", "train class 1"], ncol=4, loc=(.1, 1.1))
# plt.show()

# 2.4.2 예측 확률 p157
print("확률 값의 형태 : {}".format(gbrt.predict_proba(X_test).shape))
