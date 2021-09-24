from bert_nli import BertNLIModel
import pandas as pd
from bertscore import calc_bert_score

# csvデータセットを整形
def load_dataset(filepath, encoding='utf-8'):
    # read_csv()は区切り文字がカンマ,でread_table()は区切り文字がタブ\t。
    df = pd.read_csv(filepath, encoding=encoding)
    # 意図せず\sが出てきてしまっていたので消す
    df['S']=df['S'].replace("\'s", '')
    # df['答え']=df['答え'].replace("\'s", '')
    a=[]
    b=[]
    # c=[]
    for d1 in df['S']:
        a.append(d1)
    for d2 in df['T']:
        b.append(d2)
        # c.append('contradiction')
    return a,b
    # return c
sent1,sent2=load_dataset('./論文不正解文.csv')
# sent1,sent2,labels=load_dataset('./論文不正解文.csv')

# 問題数
acc=len(sent1)
print('len(acc)',acc)
# nliの正解数
pre=0
# bertscoreの正解数
pre1=0
#bert_type = 'bert-base'

# 以下nli
# ファインチューニング済のモデルを読み込む
model = BertNLIModel('./nli_model_acc0.8831943861332694.state_dict')
# for s1,s2,l in zip(sent1,sent2,labels):
for s1,s2 in zip(sent1,sent2):
    # 両方向で含意関係認識
    sent_pairs_right = [(s1,s2)]
    sent_pairs_left = [(s2,s1)]
    nli_label_right,prob_right = model(sent_pairs_right)
    print('prob_right:',prob_right[0])
    nli_label_left,prob_left = model(sent_pairs_left)
    print('prob_left:', prob_left[0])
    nli_label = 'contradiction'
    print('type1111111111111:',type(prob_right[0][1].item()))
    if prob_right[0][1].item() > 0.95 and prob_left[0][1].item() > 0.95:
        nli_label = 'entail'
    print('【nli】','正解：contradiction','予測：',nli_label)
    # contradiction entail neutral
    if nli_label == 'contradiction':
        pre+=1
    print('nli_pre：',pre)
    # 以下bertscore
    # P, R, F1 = calc_bert_score([s1], [s2])
    # # 予測
    # pred = 'contradiction or neutral'
    # # 閾値
    # th = 0.98
    # F1=F1[0]
    # print(F1)
    # if F1>=th and 'entail' in l:
    #     pre1+=1
    #     pred = 'entail'
    # elif F1<th and ('contradiction' in l or 'neutral' in l):
    #     print('よばれました')
    #     pre1+=1
    # print('文1:',s1)
    # print('文2:',s2)
    # print('【bertscore】','正解：',l,'予測：',pred,'数値：',F1)
    # print('bertscore_pre1',pre1)
    # print('======================================================')


# 適合率の算出
# def calc_precicsion():



print('正解率1',pre/acc)
# print('正解率2',pre1/acc)