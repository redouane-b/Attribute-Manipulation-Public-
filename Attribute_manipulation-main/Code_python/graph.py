import matplotlib.pyplot as plt
import pickle

with open('Code_python/graph/accuracy.npy',"rb") as f:
  acc=pickle.load(f)

with open('Code_python/graph/loss.npy',"rb") as f:
  loss=pickle.load(f)

with open('Code_python/graph/validaton_score.npy',"rb") as f:
  validation_score=pickle.load(f)

loss=[elem.detach().numpy() for elem in loss]


fig,ax= plt.subplots()

'''
#Graph score MLP

ax.plot(acc,color='r',label='accuracy')
ax.plot(loss,color='c',label='loss')
ax.plot(validation_score,color='b',label='cross validation score')

plt.ylim([0,1])
plt.legend()
plt.show()
'''

'''
#Graph score SVM wrt age

vect1=[0.5,0.51435407,0.57939509,0.63930582,0.68900425]
vect2=[0.55909091,0.54269082,0.58865248,0.68350168,0.72829689]
vect3=[0.56814427,0.56715795,0.68392238,0.79051956,0.83311478]
vect4=[0.41882435,0.45234899,0.68584223,0.79342561,0.82969129]

ax.plot(vect1,color='r',label='Group 1: 0-6')
ax.plot(vect2,color='b',label='Group 2: 7-14')
ax.plot(vect3,color='c',label='Group 3: 15-49')
ax.plot(vect4,color='k',label='Group 4: 50-120')

plt.ylim([0.5,1])
plt.legend()
plt.show()
'''


'''
score for 1 = 0.5388095238095238
score for 2 = 0.5486190476190477
score for 3 = 0.6688571428571428
score for 4 = 0.7698571428571429
score for 5 = 0.816047619047619
'''

'''
Score for people above 15
score with latents 1 = 0.5494429570774678
score with latents 2 = 0.5479747819328094
score with latents 3 = 0.6893514120390362
score with latents 4 = 0.7960100181362812
score with latents 5 = 0.8426461697901373
'''

vect_lats=[0,0.5335570469798657,0.5503355704697986,0.6241610738255033, 0.7315436241610739,0.7818791946308725,0.7583892617449665,
0.7516778523489933,0.7550335570469798,0.7751677852348994,0.8204697986577181,0.8003355704697986]

fig,ax= plt.subplots()

ax.plot(vect_lats,color='b',label='score')
ax.axvline(x=5,ymin=0,ymax=0.7818,linestyle="dotted",color='r')
ax.axvline(x=10,ymin=0,ymax=0.82,linestyle="dotted",color='r')

plt.ylim([0,1])
plt.legend()
plt.show()