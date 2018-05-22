import numpy as np
# 4.1 numpy ndarray: 다차원 배열 객체
# numpy의 핵심기능중 하나는  N차원의 배열 객체 또는 ndarray로, 파이썬에서 사용할 수 있는
# 대규모 데이터 집합을 담을 수 있는 빠르고 유연한 자료구조

data = np.random.randn(2,3)
print(data)
print(data * 10)
print(data + data)

# ndarray는 같은 종류의 데이터를 담을 수 있는 포괄적인 다차원 배열이며, ndarray의 모든 원소는 같은 자료형이여야한다.
# 모든 배열은 각 차원의 크기를 알려주는 shape라는 튜플과 배열에 저장된 자료형을 알려주는 dtype이라는 객체를 가지고 있다.

print(data.shape)
print(data.dtype)

# 4.1.1 ndarray 생성

data1 = [6, 7.5, 8, 0, 1]
arr1 = np.array(data1)
print(arr1)

# 같은 길이의 리스트가 담겨있는 순차 데이터는 다치원 배열로 변환이 가능하다

data2 = [[1, 2, 3, 4], [5, 6, 7, 8]]

arr2 = np.array(data2)

print(arr2)
print(arr2.ndim)  # ndim : 몇차원 배열인가?
print(arr2.shape)

print(arr1.dtype)
print(arr2.dtype)

print(np.zeros(10))  # zeros: 주어진 길이나 모양에 0으로 구성된 배열을 생성
print(np.ones((3, 6)))  # ones: 주어진 길이나 모양에 1으로 구성된 배열을 생성

print(np.empty((2, 3, 2)))  # empty: 초기화 되지 않은 배열을 생성한다.

print(np.arange(15))  # arange는 파이썬의 range함수의 배열 버전이다.


# 4.1.2 ndarray의 자료형

arr1 = np.array([1, 2, 3], dtype=np.float64)
arr2 = np.array([1, 2, 3], dtype=np.int32)

print(arr1.dtype)
print(arr2.dtype)

# ndarray의 astype 메서드를 사용해서 배열의 dtype을 다른 형으로 명시적으로 변경이 가능하다.

arr = np.array([1, 2, 3, 4, 5])

print(arr.dtype)

float_arr = arr.astype(np.float64)

print(float_arr.dtype)

arr = np.array([3,7, -1.2, -2.6, 0.5, 12.9, 10.1])

print(arr)

print(arr.astype(np.int32))

numeric_strings = np.array(['1.25', '-9.6', '42'], dtype=np.string_)

print(numeric_strings.astype(float))

print(numeric_strings.astype(float).dtype)  # float 몇이라고 지정안해도 알아서 바꿔준다.

int_array = np.array(10)

calibers = np.array([.22, .270, .357, .380, .44, .50], dtype=np.float64)

int_array.astype(calibers.dtype)

empty_uint32 = np.empty(8, dtype='u4')

print(empty_uint32)

# 4.1.3 배열과 스칼라 간의 연산

arr = np.array([[1., 2., 3.], [4., 5., 6.]])

# 배열은 for 반복문을 작성하지 않고 데이터를 일괄처리 할 수 있기 때문에 중요하다.
# 이를 벡터화라고 하는데, 같은 크기의 배열 간 산술연산은 배열의 각 요소 단위로 적용된다.

print(arr)
print(arr * arr)
print(arr - arr)

# 스칼라 값에 대한 산술 연산은 각 요소로 전달된다.

print(1 / arr)
print(arr ** 0.5)

# 4.1.4 색인과 슬라이싱 기초

arr = np.arange(10)

print(arr)
print(arr[5])
print(arr[5:8])

# 브로드캐스팅

arr[5:8] = 12

print(arr)

arr_slice = arr[5:8]

arr_slice[1] = 12345

print(arr)

# 다량의 데이터 처리를 염두에 두고 만든 numpy package이기에 배열의 slice는 원본의 참조를 리턴합니다.
# 따라서, slice의 원소를 변경하면 원본에도 영향을 끼칩니다.

arr_slice[:] = 64

print(arr)

arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print(arr2d[2])
print(arr2d[0][2])
print(arr2d[0, 2])

arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
print(arr3d)
print(arr3d[0])

old_values = arr3d[0].copy()

arr3d[0] = 42
print(arr3d)

arr3d[0] = old_values
print(arr3d)

print(arr3d[1, 0])

# 슬라이스 색인

print(arr[1:6])
print(arr2d)
print(arr2d[:2])
print(arr2d[:2, 1:])

print(arr2d[1, :2])
print(arr2d[2, :1])
print(arr2d[:, :1])

arr2d[:2, 1:] = 0

# 4.1.5 불리언 색인

names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])

data = np.random.randn(7, 4)
print(names)
print(names.dtype)
print(data)

# broadcasting 이 이루어 진다.
# boolean indexing 이 이거, sql에서 where과 같다.
print(names == 'Bob')  # 해당 위치에 대한 True, false값으로.

# , 가 없기에, bob이 축이 된다.
print(data[names == 'Bob'])
print(data[names == 'Bob', 2:])
print(data[names == 'Bob', 3])  # Bob이 해당되는 col의 4번째 col값

print(names != 'Bob')

# ~ : not
print(~(names == 'Bob'))

print([~(names == 'Bob')])
print(names == 'Bob')
print(names == 'Will')

mask = (names == 'Bob')|(names == 'Will')
print(mask)

print(data[mask])

print(data<0)
print(data[data<0])

data[data<0] = 0  # 0보다 작은 값은 0으로 만든다!

print(data)

data[names != 'Joe'] = 7
print(data)

# 4.1.6 팬시 색인

arr = np.empty((8, 4))

for i in range(8):
    arr[i] = i

print(arr)

# 특정한 순서로 로우를 선택하고 싶으면 그냥 원하는 순서가 명시된 정수가 담긴
# ndarray나 리스트를 넘기면 된다

print(arr[[4, 3, 0, 6]]) # True 값만 변환

# 색인으로 음수를 사용하면 끝에서부터 로우를 시작한다.

print(arr[[-3, -5, -7]]) # 3은 최대값에서 뺀 값

arr = np.arange(32).reshape(8, 4)

print(arr)  # 앞의 .은 소속을 나타내고, 뒤의 .은 메소드

print(arr[[1, 5, 7, 2], [0, 3, 1, 2]]) # (1, 0), (5, 3), (7, 1), (2, 2)에 대응되는 값들이 선택된다.

print(arr[[1, 5, 7, 2]][:, [0, 3, 1, 2]])  # 행렬의 행과 열에 대응하는 사각형 모양의 값이 선택되는 것.

# np.ix_ 함수를 사용하면 위와 같은 결과를 얻을 수 있다.

print(arr[np.ix_([1, 5, 7, 2], [0, 3, 1, 2])])

ar = np.arange(8)
print(ar)
print(ar[3:7])
print(ar[[3, 5, 2, 1]])

ar2 = ar.reshape(2, 4)
print(ar2)
print(ar2.shape)
print(ar2[1])
print(ar2[:, 1])

ar3 = np.arange(60).reshape(3, 5, 4)  # 3x5x4 = 60
print(ar3)
print(ar3[[1, 2], [4, 2], [0, 1]])
print(ar3[np.ix_([1, 2], [4, 2], [0, 1])]) # np.ix_를 쓰기에, 순서대로 0, 1, 2번 축이 된다.

arr = np.arange(60)
print(arr) # 행 벡터, 1차원 배열
print(arr.shape)
arr = arr.reshape((60, 1))
print(arr)

# 4.1.7 배열 전치와 축 바꾸기
# 배열 전치는 데이터를 복사하지 않고, 데이터 모양이 바뀐 뷰를 반환하는 특별한 기능!
# 전치 배열 , 보통 2차원 배열 또는 행렬일때 사용한다.

arr = np.arange(15).reshape((3, 5))

print(arr)
print(arr.T) # 행렬명.T로 전치행렬을 표현한다. 3x5 -> 5x3

arr = np.random.randn(6, 3)
print(np.dot(arr.T, arr)) # 행렬 곱셈 = 행렬명.dot

# 다차원 배열의 경우 transpose 메서드는 튜플로 축 번호를 받아서 치환한다.
# 실제로 계산하는거 무리임.

arr = np.arange(16).reshape((2, 2, 4))
print(arr)
print(arr.shape)

# transpose : 축의 순서를 바꿔주는 것.
print(arr.transpose((1, 2, 0)))  # 1번 축을 0번 축에, 2번을 1번축에, 0번 축을 2번축에
print(arr.transpose((1, 2, 0)).shape)

#행렬명.swapaxes = 2개의 축 번호를 받아서 배열을 뒤바꾼다.
print(arr.swapaxes(1, 2))

# 유니버셜 함수
# ufunc라고 불리는 유니버셜함수는 ndarray 안에 있는 데이터 원소별로 연산을 수행하는
# 함수이다. 유니버셜 함수는 하나 이상의 스칼라 값을 받아서 하나 이상의 스칼라 결과
# 값을 반환하는 간단한 함수를 고속으로 수행할 수 있는 벡터화된 wrapper함수라고 생각하면 된다!

arr = np.arange(10)
print(np.sqrt(arr))  # sqrt(x) =  각 원소의 제곱근을 계산한다. x ** 0.5와 동일
print(np.exp(arr))
print(len(arr))  # 유니버셜 함수가 아니므로, 결과는 1건

x = np.random.randn(8)
y = np.random.randn(8)

# 위의 예제를 유니버셜 함수가 아니면 이렇게 써야한다
np.array([np.sqrt(x) for x in arr])

# 이항 유니버셜 함수 : add 나 maximum처럼 2개의 인자를 취해서 단일 배열을 반환하는 함수
print(x)
print(y)

print(np.maximum(x, y))  # x, y 중 큰것만 배열에 넣는다.

# 배열 여러개를 반환하는 유니버설 함수
# modf는 파이썬 내장 함수인  divmod의 벡터화 버전
# modf는 각각의 요소에 대해 소수점 이하의 값과 정수값을 각각의 배열로 반환

arr = np.random.randn(7) * 5
print(np.modf(arr))


