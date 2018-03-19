import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras import losses

#데이터 수집 , 정제
feature_data = np.array([[0,0], [0,1],  [1,0],  [1,1]])
target_data = np.array([[0],    [1],    [1],    [0]])

#레이어 제작
model = Sequential()
model.add(Dense(16, input_dim=2, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#학습 프로세스 설정
model.compile(loss=losses.mean_squared_error,
              optimizer='adam',
              metrics=['binary_accuracy'])

# 학습
model.fit(feature_data, target_data, epochs=500, verbose=2)
print(model.evaluate(feature_data, target_data))

#값 예측 반올림
print (model.predict(feature_data).round())
#값 예측 소수점
print (model.predict(feature_data))
