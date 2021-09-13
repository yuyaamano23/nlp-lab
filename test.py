from bert_nli import BertNLIModel
import pandas as pd
from bertscore import calc_bert_score

def load_dataset(filepath, encoding='utf-8'):
    df = pd.read_csv(filepath, encoding=encoding)
    df['t1']=df['t1'].replace("\'s", '')
    df['t2']=df['t2'].replace("\'s", '')
    a=[]
    b=[]
    c=[]
    for d1 in df['t1']:
        a.append(d1)
    for d2 in df['t2']:
        b.append(d2)
    for l in df['label']:
        c.append(l)
    #for l in df['label']:
        #if l=='not_entailment':
        #    ll='contradiction'
        #elif l=='entailment':
        #    ll='entail'
        #c.append(ll)

    return a,b,c
sent1,sent2,labels=load_dataset('./ge_test.csv')

acc=len(sent1)
pre=0
pre1=0
#bert_type = 'bert-base'

model = BertNLIModel('./nli_model_acc0.8831943861332694.state_dict')
for s1,s2,l in zip(sent1,sent2,labels):
    fla=''
    if l in 'entail':
        fla='contradiction'
    else:
        fla='entail'
    sent_pairs = [(s1,s2)]
    #print(sent_pairs)
    label,prob= model(sent_pairs)
    print('正解：',l,'予測',label[0])
    print('前提：',s1)
    print('仮定：',s2)
    print(' ')
    if l==label[0] or label[0] in 'neutral':
        pre+=1
    P, R, F1 = calc_bert_score([s1], [s2])
    F1=F1[0]
    if F1>=0.85 and label[0] in 'entail':
        pre1+=1
    elif F1<0.50 and label[0] in 'contradiction':
        pre1+=1
    elif F1<0.85 and label[0] in 'neutral':
        pre1+=1

print('正解率1',pre/acc)
print('正解率2',pre1/acc)