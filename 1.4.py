from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import mglearn
from scipy import sparse

#1.4.2 NumPy
x = np.array([[1,2,3], [4,5,6]])
#print("x:\n{}".format(x))

#1.4.3 SciPy 행렬
eye = np.eye(4)
#print("Numpy 배열 : \n{}".format(eye))
sparse_matrix = sparse.csr_matrix(eye)
#print("SciPy의 CSR행렬 :\n{}".format(sparse_matrix))

#1.4.4 matplotlib 그래프 그리기
x = np.linspace(-10,10,100)
y = np.sin(x)
#plt.plot(x,y,marker="x")
#plt.show()

#1.4.5 Pandas 데이터 처리와 분석
import pandas as pd
# 회원 정보가 들어간 간단한 데이터셋을 생성합니다.
data = {'Name' : ["John", "Anna", "Peter", "Linda"],
        'Location' : ["New York", "Paris", "Berlin", "London"],
        'Age' : [24,13,53,33]}
data_pandas = pd.DataFrame(data)
#display(data_pandas)
#Age 열의 값이 30이상인 모든 행을 선택합니다
display(data_pandas[data_pandas.Age > 30])
