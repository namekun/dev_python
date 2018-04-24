#sort

a= [7, 2, 5, 1, 3]

a.sort()

print(a)

#정렬 기준으로 사용할 값을 반환 할 수 있다.

b = ['saw', 'small', 'He', 'foxes', 'six']

b.sort(key = len)

print(b)

#이진탐색과 정렬된 리스트 유지하기
#bisect.bisect메소드는 값이 추가될 때 리스트가 정렬된 상태를 유지할 수 있는 위치를 반환하며
# bisect.insort메소드는 실제로 정렬된 상태를 유지한 채 값을 추가한다.

import bisect

c = [1,2,2,2,3,4,7]

print(bisect.bisect(c, 2))

print(bisect.bisect(c, 5))

print(bisect.insort(c, 6))

print(c)

#슬라이싱

seq = [7,2,3,7,5,6,0,1]

print(seq[1 : 5])

print(seq[3 : 4])

print(seq)

#색인의 시작 위치에 있는 값은 포함되지만, 끝 위치에 있는 값은 포함되지 않는다. 따라서 슬라이싱의 결과 개수는 stop-start이다.
#색인의 시작 값이나, 끝값이 생략될 수 있다. 이 경우 생략된 값은 각각 순차 자료형의 맨 처음, 혹은 제일 마지막 값이 된다.

print(seq[:5])

print(seq[3:])

print(seq[-4:])

print(seq[-6:-2])

#두번째 콜론 다음에 간격도 정해줄 수 있다.

print(seq[::2])

#간격 값으로 -1을 사용하면 리스트나 튜플을 역순으로 반환한다.

print(seq[::-1])

#내장 순차 자료형 함수

#enumerate : 순차 자료형에서 현재 아이템의 색인을 함께 처리하고자 할 떄, 흔히 사용된다.
#색인을 통해 데이터에 접근할 때, enumerate를 사용하는 유용한 패턴은 순차 자료형에서의 값과 그 위치를 dict에 넘겨주는 것.

some_list = ['foo', 'bar', 'baz']

mapping = dict((v, i) for i, v in enumerate(some_list))

print(mapping)

#sorted함수는 정렬된 새로운 순차 자료형을 반환한다.

print(sorted([7,1,2,6,0,3,2]))
print(sorted('horse race'))

#순차 자료형에서 유일한 값을 가지는 정렬된 리스트를 가져오는 패턴은 sorted와 set을 통해 구현할 수 있다.

print(sorted('this is just some string'))

#zip - 여러개의 리스트나 튜플 또는 다른 순차 자료형을 서로 짝지어서 튜플의 리스트를 생성한다.

seq1 = ['foo', 'bar', 'baz']
seq2 = ['one','two','three']
seq3 = [False, True]

print(list(zip(seq1, seq2)))
print(list(zip(seq1, seq2, seq3)))

for i,(a,b) in enumerate(zip(seq1, seq2)):
    print('%d: %s, %s' % (i, a, b))

#reversed는 순차자료형을 역순으로

print(list(reversed(range(10))))

#dictionary

d1 = {'a':'some value', 'b': [1,2,3,4]}

print(d1)

d1[7] = 'an integer'

print(d1)

print(d1['b'])

'b' in d1 #사전에 어떤 키가 있는지확인

#update 메서드를 통해 다른 사전과의 병합이 가능

d1.update({'b': 'foo'})

print(d1)

mapping = dict (zip(range(5), reversed(range(5))))

print(mapping)

#세트는 유일한 원소만 담는 정렬되지 않은 자료형이다. 사전과 유사하지만, 값은 없고 키만 갖고 있는 형태이다.

set([2,2,2,1,3,3])

print({2, 2, 2, 1, 3, 3})

#set의 산술연산집합

a = {1, 2, 3, 4, 5}
b = {3, 4, 5, 6, 7, 8}

print(a | b) #합집합
print(a & b) #교집합
print(a - b) #차집합
print(a ^ b) #대칭차집합(xor)

print({1, 2, 3}.issubset(a)) #어떤 세트가 다른 세트의 부분집합인지 확인하는 함수

print(a.issubset({1, 2, 3})) #반대

#내포
#기본형 : expr for val in collection if condition
#이를 반복문으로 구현하면 다음과 같다
#result = []
#for val in collection : if condition: result.append(expr)

strings = ['a', 'as', 'bat', 'car', 'dove', 'python']

print([x.upper() for x in strings if len(x) > 2])
 
#dict_comp = {key-expr : valaue-expr for value in collection if condition}
#set_comp = {expr for value in collection if condition

unique_lengths = {len(x) for x in strings} # - 리스트 내의 문자열들의 길이가 저장된 세트 생성

print(unique_lengths)

loc_mapping = {val : index for index, val in enumerate(strings)} #  - 리스트에서 문자열의 위치를 저장하고 있는 사전을 생성

print(loc_mapping)

