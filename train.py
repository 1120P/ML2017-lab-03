import feature
import ensemble
from PIL import Image
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import pickle
import codecs

path1='/Users/apple/Documents/ml/3/datasets/original/face/'
path2='/Users/apple/Documents/ml/3/datasets/original/nonface/'
size_tran=24,24
face=[]
nface=[]
face_l=[]
nface_l=[]
N=100
for i in range(N):
    name = path1+'face_' + str(i).zfill(3) + '.jpg'
    obj = Image.open(name)#打开图片
    
    obj.convert('L')#彩图转换为灰度图
    obj.thumbnail(size_tran, Image.ANTIALIAS)
    npd=feature.NPDFeature(np.array(obj))
    face.append(pickle.dumps(npd.extract().tolist(),True))#讲道理我不知道老师要求里是不是这个意思但是我觉得怪怪的你们可以再改一下…ˊ_>ˋ
    #face.append(n.tolist())
    
    name = path2+'nonface_' + str(i).zfill(3) + '.jpg'
    obj = Image.open(name)
    
    obj.convert('L')
    obj.thumbnail(size_tran, Image.ANTIALIAS)
    
    npd = feature.NPDFeature(np.array(obj))
    nface.append(pickle.dumps(npd.extract().tolist(),True))
    #nface.append(npd.extract().tolist())
p=np.ones(N)
n=-p
label=np.concatenate((p.reshape(-1,1),n.reshape(-1,1)),axis=0)
for i in range(N):
    face_l.append(pickle.loads(face[i]))
    nface_l.append(pickle.loads(nface[i]))
data=np.concatenate((face_l,nface_l),axis=0)#把pickle

weak_classifier=DecisionTreeClassifier()
clf=ensemble.AdaBoostClassifier(weak_classifier=weak_classifier,n_weakers_limit=10)


from sklearn.model_selection import train_test_split
training_X, validation_X,training_y,validation_y = train_test_split(data,label,test_size=0.2)


clf.fit(training_X,training_y)
y_pred=clf.predict(validation_X)

y_pred=y_pred.reshape(-1).tolist()
y_true=validation_y.reshape(-1).tolist()


target_names=['face','nonface']
fout = codecs.open('report.txt','w','utf-8')
result=classification_report(y_true, y_pred, target_names=target_names,digits=3)
fout.write(result)
print(result)
fout.close()
'''
精度(precision) = 正确预测的个数(TP)/被预测正确的个数(TP+FP)
召回率(recall)=正确预测的个数(TP)/预测个数(TP+FN)
F1 = 2*精度*召回率/(精度+召回率)
    '''

if __name__ == "__main__":
    print("aaaaa")
    pass

