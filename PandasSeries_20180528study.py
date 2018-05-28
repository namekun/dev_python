from pandas import Series, DataFrame
import pandas as pd
import numpy as np

# 5.1.1 Series는 일련의 객체를 담을 수 있는 1차원 배열같은 자료구조.
# 그리고 색인(idx)라는 배열의 데이터에 연관된 이름을 갖고 있다.
# 가장 간단한 Series 객체는 배열 데이터로부터 생성가능하다.

obj = Series([4, 7, -5, 3])
print(obj)

# Series 객체의 문자열 표현은 왼쪽은 색인, 오른쪽은 해당 색인의 값을 보여준다.
# 앞의 예제에서는 데이터의 색인을 지정하지 않았으니, 기본 색인은 0부터 시작한다.

print(obj.values)  # Series의 배열 객체 얻기
print(obj.index)  # Series의 idx 객체 얻기

# 색인을 지정해 줄 때.

obj2 = Series([4, 7, -5, 3], index=['d', 'b', 'a', 'c'])
print(obj2)
print(obj2.index)

# 배열에서 값을 선택하거나, 대입할 때는 색인을 이용해서 접근한다.

print(obj2['a'])
print(obj2['d'])
print(obj2[['c', 'a', 'b']])

# boolean 배열을 사용해서 값을 걸러내거나, 산술 곱셈을 유지하거나
# 또는 수학 함수를 적용하는 등 Numpy 배열 연산을 수행해도 색인-값의 연결은 유지된다.

print(obj2[obj2 > 0])
print(obj2 * 2)
print(np.exp(obj2))

# Series는 Python의 사전형과 비슷하다.

print('b' in obj2)
print('e' in obj2)

# 파이썬 사전형에 데이터를 저장해야 한다면 파이썬 사전 객체로부터 Series 객체를 생성할 수도 있다.

sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
obj3 = Series(sdata)
print(obj3)

# 사전 객체만 가지고 Series 객체를 생성된 Series 객체의 색인은 사전의 키 값이 순서대로 들어간다.

states = ['California', 'Ohio', 'Oregon', 'Texas']
obj4 = Series(sdata, index=states)
print(obj4)

# 함수의 누락된 데이터를 찾을 때

print(pd.isnull(obj4))
print(pd.notnull(obj4))

# 이 메서드는 Series의 인스턴스 메서드이기도 하다.

print(obj4.isnull())

# Series 객체와 Series의 색인은 모두 name 속성이 있는데, 이 속성은 Pandas의 기능에서 중요한 부분을 차지한다.

obj4.name = 'population'
obj4.index.name = 'state'

print(obj4)

# index는 대입을 통해서 바꿀 수 있다.

obj.index = ['Bob', 'Steve', 'Jeff', 'Ryan']
print(obj)

# 5.1.2 DataFrame

data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],
        'year': [2000, 2001, 2002, 2001, 2002],
        'pop': [1.5, 1.7, 3.6, 2.4, 2.9]}

# DataFrame은 표 같은 스프레드 시트 형식의 자료구조로 여러 개의 칼럼이 있는데,
# 각 칼럼은 서로 다른 종류의 값을 담을 수 있다.

frame = DataFrame(data)
print(frame)

# 원하는 순서대로 columns를 지정하면 원하는 순서를 갖는 DataFrame 객체가 생성된다.

print(DataFrame(data, columns=['year', 'state', 'pop']))

# Series와 마찬가지로 data에 없는 값을 넘기면 NA 값이 저장된다.

frame2 = DataFrame(data, columns=['year', 'state', 'pop', 'debt'],
                   index=['one', 'two', 'three', 'four', 'five'])
print(frame2)

# DataFrame의 칼럼은 Series처럼 사전 형식의 표기법으로 접근하거나, 속성 형식으로 접근가능하다.

print(frame2['state'])
print(frame2.year)

# 반환된 Series 객체가 DataFrame 같은 색인을 가지면 알맞은 값으로 name 속성이 채워진다.

print(frame2.ix['three'])  # row 는 ix와 같은 몇 가지 메소드로 접근 할 수 있다.
print(frame2.loc['three'])
print(frame2.iloc[2])


# 칼럼의 대입

frame2['debt'] = 16.5

print(frame2)

frame2['debt'] = np.arange(5.)

print(frame2)

# 칼럼에 리스트 혹은 배열을 대입하려면 대입하려는 값의 길이가 DataFrame의 크기와 같아야함.

val = Series([-1.2, -1.5, -1.7], index=['two', 'four', 'five'])

frame2['debt'] = val

print(frame2)

# 없는 칼럼을 대입하면 새로운 칼럼이 생성된다.

frame2['eastern'] = frame2.state == 'Ohio'
print(frame2)

# 칼럼의 삭제는 del

del frame2['eastern']

print(frame2)

# 중첩된 사전

pop = {'Nevada': {2001: 2.4, 2002: 2.9},
       'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}} # 사전이 중첩된 형태.

# 바깥에 있는 사전의 키 값이 칼럼이 되고, 안에 있는 키는 로우가 된다.

frame3 = DataFrame(pop)

print(frame3)

# Numpy와 같이 값을 뒤집을 수 있다.

print(frame3.T)

# index를 직접 지정한다면 지정한 색인으로 DataFrame을 생성한다.

print(DataFrame(pop, index=[2001, 2002, 2003]))

# Serieres 객체를 담고 잇는 사전 데이터도 같은 방식으로 취급 된다.

pdata = {'Ohio': frame3['Ohio'][:-1],
         'Nevada': frame3['Nevada'][:2]}
print(DataFrame(pdata))

# 데이터프레임 생성자에 넘길 수 있는 자료형의 목록은 따로 참고

frame3.index.name = 'year'
frame3.columns.name = 'state'

print(frame3)

# Series와 유사하게 values 속성은 DataFrame 에 저장된 데이터를 2차원 배열로 반환한다.

print(frame3.values)

# DataFrame의 칼럼에 서로 다른 dtype이 있다면 모든 칼럼을 수용하기 위해 그 칼럼 배열의 dtype이 선택된다.

print(frame2.values)