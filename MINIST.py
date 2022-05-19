
import numpy as np
import matplotlib.pyplot as plt
#1. 导入TensorFlow
import tensorflow as tf

#（c）找到 784 × 10 = 7840权重使得网络输出 [1 0 0 ... 0] T如果输入图像对应于 0，[0 1 0 ... 0] T
def init(x,y):
    layer=np.random.uniform(-1.,1.,size=(x,y))/np.sqrt(x*y)
    return layer.astype(np.float32)
#have a neural network 28×28 = 784 nodes
# _init_()保存成员变量的设置
layer = init(28 * 28, 10)

def get_error(eta, r, n, layer,break_point):
    tr_error = []
    errors = n

    while ((errors / n) > r and break_point > 0):
        errors = 0
        break_point-=1
        for i in range(n):
            y_true=np.zeros((1, 10))
            y_pred=np.zeros((1, 10))
            y = np.dot(np.array(x_train[i]).reshape(-1, 784), layer)
            y_pred[0][np.argmax(y)]=1
            y_true[0][y_train[i]]=1
            if(y_pred[0][y_train[i]]!=1):
                errors+=1
        tr_error.append(errors)
        for i in range(n):
            y_true=np.zeros((1, 10))
            y_pred=np.zeros((1, 10))
            y = np.dot(np.array(x_train[i]).reshape(-1, 784), layer)
            y_pred[0][np.argmax(y)]=1
            y_true[0][y_train[i]]=1
            layer = layer+eta*np.dot(np.transpose(np.array(x_train[i]).reshape(-1,784)),(y_true-y_pred))
#e
    te_error=0
    for i in range(10000):#test images
        y_true=np.zeros((1, 10))
        y_pred=np.zeros((1, 10))
        y = np.dot(np.array(x_test[i]).reshape(-1, 784), layer)
        y_pred[0][np.argmax(y)]=1
        y_true[0][y_test[i]]=1
        if(y_pred[0][y_test[i]]!=1):
            te_error+=1
    return tr_error, te_error

#Mnist数据集的载入
mnist = tf.keras.datasets.mnist #2. 通过keras使用数据集
(x_train, y_train), (x_test, y_test) = mnist.load_data() # 载入数据

#d(1)
np.random.seed(88)
'''
eta = 1
r = 0
n = 50
break_point=50
ta_error, te_error =  get_error(eta, r, n, layer, break_point)

plt.xlabel('epoch times')
plt.ylabel('mislabeled point')
plt.plot(ta_error, color='black', label='n = 50')
plt.legend()
plt.show()
print("test error among 10000 test images: ", te_error)
'''
'''
eta = 1
r = 0
n = 1000
break_point=50
ta_error, te_error =  get_error(eta, r, n, layer, break_point)

plt.xlabel('epoch times')
plt.ylabel('mislabeled point')
plt.plot(ta_error, color='black', label='n = 1000')
plt.legend()
plt.show()
print("test error among 10000 test images: ", te_error)
'''



eta = 1
r = 0.05
n = 60000
break_point=20
ta_error, te_error =  get_error(eta, r, n, layer, break_point)

plt.xlabel('epoch times')
plt.ylabel('mislabeled point')
plt.plot(ta_error, color='black', label='eta = 1,r = 0.05')
plt.legend()
plt.show()
print("test error among 10000 test images: ", te_error)


e