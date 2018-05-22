import random
import matplotlib.pyplot as plt
import numpy as np

# 순수 파이썬으로 계단오르내리기를 1000번 수행하는 코드를 작성.

position = 0
walk = [position]
steps = 1000
for i in range(steps):
    step = 1 if random.randint(0, 1) else -1
    position += step
    walk.append(position)

plt.plot(walk[:100])

print(walk[:20])

# numpy.random으로 구현하기

nsteps = 1000
draws = np.random.randint(0, 2, size=nsteps)
steps = np.where(draws > 0, 1, -1)
walk = steps.cumsum()

# 계단을 오르내린 위치의 최소/ 최대 값을 구할 수 있다.

print(walk.min())
print(walk.max())

# 계단의 처음 위치에서 최초로 10칸 떨어지기까지 얼마나 걸렸는가?

print((np.abs(walk) >= 10).argmax())  # argmax() = 불리언 배열에서 최대값의 처음 색인을 반환

# 4.7.1 한번에 계단 오르내리기 시뮬레이션하기

nwalks = 5000
nsteps = 1000
draws = np.random.randint(0, 2, size=(nwalks, nsteps)) # 0 or 1
steps = np.where(draws > 0, 1, -1)
walks = steps.cumsum(1)
walks = np.hstack((np.zeros((nwalks, 1), dtype=np.int32), walks))  #처음 시작 위치 0을 모든 랜덤워크  첫값으로 설정

#draw값이 0이면 한계단 내려가기 , 1이면 올라가기
steps = np.where(draws > 0, 1, -1)
print(steps[:5, :10])
print(walks.max())
print(walks.min())

# 누적합이 30 or -30 이 되는 최소 시점을 구해보자

hits30 = (np.abs(walks) >= 30).any(1)  # np.abs = 절대값을 구해주는 메서드
print(hits30)

print(hits30.sum())  # 30 혹은 -30에 도달한 시뮬레이션의 개수

crossing_items = (np.abs(walks[hits30]) >= 30).argmax(1)

print(crossing_items.mean())

# 다른 분포를 사용해서도 여러 가지 시도해보기.
# normal 함수에 표준편차와 평균 값을 넣어 정규분포에서 표본을 추출하는 것처럼
# 그냥 다른 난수 발생 함수를 사용하기만 하면 된다.

steps = np.random.normal(loc=0, scale=0.25,
                         size=(nwalks, nsteps))

print(walks[hits30][:,10])
print((np.abs(walks) >= 10))