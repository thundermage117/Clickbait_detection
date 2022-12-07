import nltk
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
import pandas as pd
import numpy as np
# nltk.download()
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from model import SimilarityAware,LSD,LSDA,cos_sim,loss_function,train,train2
from input_functions import getVecDataFrame,extractData,getMaxSentLength

regexp = nltk.tokenize.RegexpTokenizer('\w+')
word_lem= nltk.stem.WordNetLemmatizer()
stopwords = nltk.corpus.stopwords.words("english")

data = pd.read_json("../clickbait-17/clickbait17-validation-170630/instances.jsonl", lines=True)
labels = pd.read_json("../clickbait-17/clickbait17-validation-170630/truth.jsonl", lines=True)

data = data.sort_values(by=['id'])
labels = labels.sort_values(by=['id'])

data = data.reset_index()
labels = labels.reset_index()

# train_webis=pd.read_json("./dataset/clickbait17-train-170331/instances.jsonl",lines=True)
columns=np.array(['id','postText','targetDescription','targetParagraphs'])
train_hd_bd=data[columns]
df2 = train_hd_bd.apply(lambda x: x.astype(str).str.lower())
df2["id"]=pd.to_numeric(df2["id"])
keys=df2.keys()

for i in range(1,4):
    df2[f"{keys[i]}_proc"]=df2[keys[i]].apply(word_lem.lemmatize)
    df2[f"{keys[i]}_proc"]=df2[f"{keys[i]}_proc"].apply(regexp.tokenize)
    df2[f"{keys[i]}_proc"]=df2[f"{keys[i]}_proc"].apply(lambda x: [word for word in x if word not in stopwords])

# df2_w2vTrain=df2[keys[:]].apply(lambda x: ",".join(x.astype(str)),axis=1)
# df2_w2vTrain=[ df2.row.sum() for row in df2.row]
keys1=df2.keys()

df2_w2vTrain=df2[keys1[len(keys)]]
# print(keys[len(keys)-1])
print(range(len(keys)+1,len(keys1)))
for i in range(len(keys)+1,len(keys1)):
    df2_w2vTrain=df2_w2vTrain+df2[keys1[i]]

word2vecLength=100
modelW2v = Word2Vec(window=10, min_count=1, workers=4,vector_size=word2vecLength)
modelW2v.build_vocab(df2_w2vTrain, progress_per=1000)
modelW2v.train(df2_w2vTrain, total_examples=modelW2v.corpus_count, epochs=modelW2v.epochs)

heading,body,labels_val=extractData(df2,labels)
head_len=getMaxSentLength(heading)
body_len=getMaxSentLength(body)
norm_len=max(head_len,body_len)
df_new,head_vec,body_vec=getVecDataFrame(heading,body,labels_val,norm_len,word2vecLength,modelW2v.wv)

print(head_vec.shape,body_vec.shape,labels_val.shape)
# print(modelW2v.wv)

head=df_new['Heading']
body=df_new['Body']
label=torch.FloatTensor(labels_val)
print(head[0].shape,body[0].shape,label.shape)

print(label.shape)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
Hin=100
Hout=100

head_vec=head_vec.to(device)
body_vec=body_vec.to(device)
label=label.to(device)

model2=LSDA(in_size=Hin,hidden_size=120,K_size=60,g_and_l=True,n_layers=2)
model2=model2.to(device)

losses,accuracies,_,_,_=train2(model2,head_vec,body_vec,label,batch_size=50,n_epochs=200,lr=0.01,gamma=0.5)

torch.save(model2.state_dict(), 'checkpoint_n.pth')

fig,axes=plt.subplots(2,1)
loss_sum=np.cumsum(losses)
x_axis=np.arange(len(losses))+1
avg_loss=loss_sum/x_axis
axes[0].plot(avg_loss)
axes[0].xlabel('num_epochs')
axes[0].ylabel('loss')
axes[0].title('Epooch vs loss')
axes[0].grid()

acc_sum=np.cumsum(accuracies)
x_axis=np.arange(len(acc_sum))+1
avg_acc=acc_sum/x_axis
axes[1].plot(avg_acc)
axes[1].xlabel('num_epochs')
axes[1].ylabel('Accuracy')
axes[1].title('Epooch vs Accuracy')
axes[1].grid()