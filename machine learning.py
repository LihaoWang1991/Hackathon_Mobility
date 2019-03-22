import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor

file=pd.read_csv('C:/Users/lwang/MEH/Hackthon/Mobility Hackathon Data/Code/Global_Car_Info_final.csv')

'''''''''''''''''''''''''''''''''Read data'''''''''''''''''''''''''''''''''

X=[]
Y=[]
size=len(file['speed_score'])

speed_score = file['speed_score']
aggressive_score =  file['aggressive_score']
slow_score = file['slow_score']
anticipative_score =  file['anticipative_score']
calm_score =  file['calm_score']
erratic_score = file['erratic_score']
smooth_score =  file['smooth_score']

maintenance_km = file['payload_ngp_payload_vehicle_maintenance_km']


for i in range(size):
    if speed_score[i] == speed_score[i] and aggressive_score[i] == aggressive_score[i] and slow_score[i] == slow_score[i] and anticipative_score[i] == anticipative_score[i] and calm_score [i] == calm_score[i] and erratic_score[i] == erratic_score[i] and smooth_score[i] == smooth_score[i] and maintenance_km[i] == maintenance_km[i]:
        X.append([speed_score[i],aggressive_score[i],slow_score[i],anticipative_score[i],calm_score[i],erratic_score[i],smooth_score[i]])
        Y.append(maintenance_km[i])

'''''''''''''''''''''''''''''''''Data normalization'''''''''''''''''''''''''''''''''

Y_range = max(Y) - min(Y)
Y_mean = np.mean(Y, axis=0)
Y = (Y - Y_mean) / Y_range


'''''''''''''''''''''''''''''''''Divide data into training and test sets'''''''''''''''''''''''''''''''''

train_size = int(len(X)*0.8)
test_size = len(X) - train_size

train_X = X[0:train_size]
train_Y = Y[0:train_size]

test_X = X[train_size:len(X)+1]
test_Y = Y[train_size:len(X)+1]

'''''''''''''''''''''''''''''''''Define and train machine learning model'''''''''''''''''''''''''''''''''
model = MLPRegressor(solver='adam', alpha=1e-5,hidden_layer_sizes=(100,100), random_state=1, max_iter=10000, tol=1e-10)
model.fit(train_X, train_Y)

'''''''''''''''''''''''''''''''''Print prediction results'''''''''''''''''''''''''''''''''

predict_Y = model.predict(test_X)
predict_Y = predict_Y * Y_range + Y_mean
test_Y = test_Y * Y_range + Y_mean
print(predict_Y[0:20])
print(test_Y[0:20])


