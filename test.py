from bert_nli import BertNLIModel
import pandas as pd
from bertscore import calc_bert_score

# csvデータセットを整形
def load_dataset(filepath, encoding='utf-8'):
    # read_csv()は区切り文字がカンマ,でread_table()は区切り文字がタブ\t。
    df = pd.read_csv(filepath, encoding=encoding)
    # 意図せず\sが出てきてしまっていたので消す
    df['S']=df['S'].replace("\'s", '')
    df['L']=df['L'].replace("\'s", '')
    a=[]
    b=[]
    c=[]
    for d1 in df['S']:
        a.append(d1)
    for d2 in df['T']:
        b.append(d2)
    for d3 in df['L']:
        c.append(d3)
    return a,b,c
sent1,sent2,labels=load_dataset('./論文データ文.csv')

# 問題数
acc=len(sent1)
print('len(acc)',acc)
# nliの正解数
ans=0
# bertscoreの正解数
ans_bert_score=0
#bert_type = 'bert-base'


# ファインチューニング済のモデルを読み込む
model = BertNLIModel('./nli_model_acc0.8831943861332694.state_dict')
for s1,s2,l in zip(sent1,sent2,labels):
# ========================  以下nli  ================================
    # 両方向で含意関係認識
    sent_pairs_right = [(s1,s2)]
    sent_pairs_left = [(s2,s1)]
    nli_label_right,prob_right = model(sent_pairs_right)
    nli_label_left,prob_left = model(sent_pairs_left)
    nli_label = ''
    # contradiction entail neutralの順番で2次元配列に格納されている
    if prob_right[0][1].item() > 0.95 and prob_left[0][1].item() > 0.95:
        nli_label = 'entail'
    else:
        nli_label = 'contradiction'
    print('【nli】','正解：',l,'予測：',nli_label)
    if nli_label == l:
        ans+=1
        print('ansプラスされました!!')
    else:
        print('ansプラスされませんでした!')
    print('nli：',ans)
# ========================  以下bert_score  ===========================
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



# =========================== 以下データ算出 =================================
# 1文字目：T(True)は予測正解、F(False)は予測不正解。
# 2文字目：Pは予測が正(Positive)、Nは予測が負(Negative)
# https://qiita.com/FukuharaYohei/items/be89a99c53586fa4e2e4#%E6%B7%B7%E5%90%88%E8%A1%8C%E5%88%97confusion-matrix%E3%81%A8tp-fp-fn-tn
# 適合率...正と予測したデータのうち、実際に正であった確率 「精度」と呼ぶこともあり。
# 再現率...実際に正であるもののうち、正であると予測された確率 私は心の中では「回収率」と呼んでいます(車などのリコール(回収)と同じ)。実際に正であるものをどれだけ回収(予測)できたかの割合
# F値... 2∗適合率∗再現率 適合率+再現率

# 分母は114 37
# 適合率の算出
def calc_precicsion(tp,fp):
    pre = tp / tp + fp
    return pre

# 再現率の算出
def calc_recall(tp,fn):
    rec = tp / tp + fn
    return rec



print('nli正解率:',ans/acc)
# print('正解率2',pre1/acc)