from bert_nli import BertNLIModel
import pandas as pd
from bertscore import calc_bert_score

# csvデータセットを整形
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
    return a,b,c
sent1,sent2,labels=load_dataset('./ge_test.csv')

# 問題数
acc=len(sent1)
# nliの正解数
pre=0
# bertscoreの正解数
pre1=0
#bert_type = 'bert-base'

# 以下nli
# ファインチューニング済のモデルを読み込む
model = BertNLIModel('./nli_model_acc0.8831943861332694.state_dict')
for s1,s2,l in zip(sent1,sent2,labels):
    sent_pairs = [(s1,s2)]
    nli_label,prob= model(sent_pairs)
    print('【nli】','正解：',l,'予測：',nli_label[0])
    if nli_label[0] in l:
        pre+=1
    # 以下bertscore
    P, R, F1 = calc_bert_score([s1], [s2])
    F1=F1[0]
    print('文1',s1)
    print('文2',s2)
    print('【bertscore】','正解：',l,'予測：',F1)
    if F1>=0.85 and 'entail' in l:
        pre1+=1
    elif F1<0.50 and 'contradiction' in l:
        pre1+=1
    elif F1<0.85 and 'neutral' in l:
        pre1+=1
    print('======================================================')

print('正解率1',pre/acc)
print('正解率2',pre1/acc)