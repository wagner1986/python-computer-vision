import numpy as np
def prob_to_binary(predict, labels, threshold):
    return [labels[id] for id,x in enumerate(predict) if x>=threshold]

pred=np.array([0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
  0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 8.1126725e-1])

labels = ['adptador','bandeja','bateria','cabo','carregador','cartucho','coldre','pendrive','spark']

print(pred.shape)
print(prob_to_binary(pred,labels,0.3))