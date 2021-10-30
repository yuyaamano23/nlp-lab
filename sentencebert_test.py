import pandas as pd
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

# csvデータセットを整形
def load_dataset(filepath, encoding='utf-8'):
    # read_csv()は区切り文字がカンマ,でread_table()は区切り文字がタブ\t。
    df = pd.read_csv(filepath, encoding=encoding)
    # 意図せず\sが出てきてしまっていたので消す
    df['S'] = df['S'].replace("\'s", '')
    df['L'] = df['L'].replace("\'s", '')
    a = []
    b = []
    c = []
    for d1 in df['S']:
        a.append(d1)
    for d2 in df['T']:
        b.append(d2)
    for d3 in df['L']:
        c.append(d3)
    return a, b, c


sent1, sent2, labels = load_dataset('./論文データ文.csv')

# 問題番号
index = 0
# 問題数
acc = len(sent1)
print('len(acc)', acc)
# bert_scoreでの不正解、正解それぞれについてのtpを初期化
tp_contradiction_sentence_bert = 0
tp_entail_bert_sentence_bert = 0
#bert_type = 'bert-base'


for s1, s2, l in zip(sent1, sent2, labels):
    sentence_bert_label = ''
    index += 1

    # Two lists of sentences
    sentences1 = [s1]
    sentences2 = [s2]

    #Compute embedding for both lists
    embeddings1 = model.encode(sentences1, convert_to_tensor=True)
    embeddings2 = model.encode(sentences2, convert_to_tensor=True)

    #Compute cosine-similarits
    cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
    # 閾値
    th = 0.87
    if cosine_scores.item() > th:
        sentence_bert_label = 'entail'
        # sentence_bertでentailと判断するかつ、実際に正解文である時
        if sentence_bert_label == l:
            tp_entail_bert_sentence_bert += 1
    else:
        sentence_bert_label = 'contradiction'
        # sentence_bertでcontradictionと判断するかつ、実際に不正解文である時
        if sentence_bert_label == l:
            tp_contradiction_sentence_bert += 1

    print('【bertscore】>>>','問題番号：', index,'正解ラベル：',l,'予測：', sentence_bert_label, '数値：', cosine_scores.item())


# =========================== 以下データ算出 =================================
# 1文字目：T(True)は予測正解、F(False)は予測不正解。
# 2文字目：Pは予測が正(Positive)、Nは予測が負(Negative)
# https://qiita.com/FukuharaYohei/items/be89a99c53586fa4e2e4#%E6%B7%B7%E5%90%88%E8%A1%8C%E5%88%97confusion-matrix%E3%81%A8tp-fp-fn-tn
# 適合率...正と予測したデータのうち、実際に正であった確率 「精度」と呼ぶこともあり。
# 再現率...実際に正であるもののうち、正であると予測された確率 私は心の中では「回収率」と呼んでいます(車などのリコール(回収)と同じ)。実際に正であるものをどれだけ回収(予測)できたかの割合
# F値... 2∗適合率∗再現率 適合率+再現率

# 適合率の算出
def calc_precicsion(tp=0, fp=0):
    pre = tp / (tp + fp)
    return pre

# 再現率の算出
def calc_recall(tp=0, fn=0):
    rec = tp / (tp + fn)
    return rec

# F値の算出
def calc_f(pre=0, rec=0):
    f = (2 * pre * rec) / (pre + rec)
    return f


# 結果の出力
# 不正解問題数:164,正解問題数:41

# 不正解文について
print('tp_cont_bert_score', tp_contradiction_sentence_bert)
print('tp_ent_bert_score', tp_entail_bert_sentence_bert)
huseikai_pre = calc_precicsion(tp_contradiction_sentence_bert, 41 - tp_entail_bert_sentence_bert)
huseikai_rec = calc_recall(tp_contradiction_sentence_bert, 164 - tp_contradiction_sentence_bert)
huseikai_f = calc_f(huseikai_pre, huseikai_rec)
print('=============不正解文================')
print('誤り検出あり：', tp_contradiction_sentence_bert, '誤り検出無し；', 164 - tp_contradiction_sentence_bert, '適合率；', huseikai_pre , '再現率：', huseikai_rec, 'F値', huseikai_f)

# 正解文について
seikai_pre = calc_precicsion(tp_entail_bert_sentence_bert, 164 - tp_contradiction_sentence_bert)
seikai_rec = calc_recall(tp_entail_bert_sentence_bert, 41 - tp_entail_bert_sentence_bert)
seikai_f = calc_f(seikai_pre, seikai_rec)
print('=============正解文================')
print('誤り検出あり：', 41 - tp_entail_bert_sentence_bert, '誤り検出無し；', tp_entail_bert_sentence_bert, '適合率；', seikai_pre , '再現率：', seikai_rec, 'F値', seikai_f)
