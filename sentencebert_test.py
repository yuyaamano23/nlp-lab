import pandas as pd
from sentence_transformers import SentenceTransformer, util

# model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
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


sent1, sent2, labels = load_dataset('./中間発表論文データ.csv')

# 問題番号
index = 0
# 問題数
acc = len(sent1)
print('len(acc)', acc)
# bert_scoreでの不正解、正解それぞれについてのtpを初期化
tp_contradiction_sentence_bert = 0
tp_entail_bert_sentence_bert = 0
#bert_type = 'bert-base'

# 0.90
# tp_cont_bert_score 149
# tp_ent_bert_score 113
# =============不正解文================
# 誤り検出あり： 149 誤り検出無し； 15 適合率； 0.745 再現率： 0.9085365853658537 F値 0.8186813186813187
# =============正解文================
# 誤り検出あり： 51 誤り検出無し； 113 適合率； 0.8828125 再現率： 0.6890243902439024 F値 0.773972602739726

# 閾値： 0.87
# tp_cont_bert_score 141
# tp_ent_bert_score 126
# =============不正解文================
# 誤り検出あり： 141 誤り検出無し； 23 適合率； 0.7877094972067039 再現率： 0.8597560975609756 F値 0.8221574344023324
# =============正解文================
# 誤り検出あり： 38 誤り検出無し； 126 適合率； 0.8456375838926175 再現率： 0.7682926829268293 F値 0.805111821086262

# 閾値： 0.865
# tp_cont_bert_score 138
# tp_ent_bert_score 131
# =============不正解文================
# 誤り検出あり： 138 誤り検出無し； 26 適合率； 0.8070175438596491 再現率： 0.8414634146341463 F値 0.8238805970149253
# =============正解文================
# 誤り検出あり： 33 誤り検出無し； 131 適合率； 0.8343949044585988 再現率： 0.7987804878048781 F値 0.8161993769470406

# 0.86
# tp_cont_bert_score 136
# tp_ent_bert_score 133
# =============不正解文================
# 誤り検出あり： 136 誤り検出無し； 28 適合率； 0.8143712574850299 再現率： 0.8292682926829268 F値 0.8217522658610271
# =============正解文================
# 誤り検出あり： 31 誤り検出無し； 133 適合率； 0.8260869565217391 再現率： 0.8109756097560976 F値 0.8184615384615384

# 閾値： 0.855
# tp_cont_bert_score 134
# tp_ent_bert_score 133
# =============不正解文================
# 誤り検出あり： 134 誤り検出無し； 30 適合率； 0.8121212121212121 再現率： 0.8170731707317073 F値 0.8145896656534954
# =============正解文================
# 誤り検出あり： 31 誤り検出無し； 133 適合率； 0.8159509202453987 再現率： 0.8109756097560976 F値 0.8134556574923547

# 閾値： 0.85
# tp_cont_bert_score 133
# tp_ent_bert_score 135
# =============不正解文================
# 誤り検出あり： 133 誤り検出無し； 31 適合率； 0.8209876543209876 再現率： 0.8109756097560976 F値 0.8159509202453987
# =============正解文================
# 誤り検出あり： 29 誤り検出無し； 135 適合率； 0.8132530120481928 再現率： 0.823170731707317 F値 0.8181818181818181

# 閾値： 0.84
# tp_cont_bert_score 130
# tp_ent_bert_score 139
# =============不正解文================
# 誤り検出あり： 130 誤り検出無し； 34 適合率； 0.8387096774193549 再現率： 0.7926829268292683 F値 0.8150470219435738
# =============正解文================
# 誤り検出あり： 25 誤り検出無し； 139 適合率； 0.8034682080924855 再現率： 0.8475609756097561 F値 0.8249258160237388

# 0.83
# tp_cont_bert_score 126
# tp_ent_bert_score 139
# =============不正解文================
# 誤り検出あり： 126 誤り検出無し； 38 適合率； 0.8344370860927153 再現率： 0.7682926829268293 F値 0.8
# =============正解文================
# 誤り検出あり： 25 誤り検出無し； 139 適合率； 0.7853107344632768 再現率： 0.8475609756097561 F値 0.81524926686217

# 0.80
# tp_cont_bert_score 114
# tp_ent_bert_score 148
# =============不正解文================
# 誤り検出あり： 114 誤り検出無し； 50 適合率； 0.8769230769230769 再現率： 0.6951219512195121 F値 0.7755102040816327
# =============正解文================
# 誤り検出あり： 16 誤り検出無し； 148 適合率； 0.7474747474747475 再現率： 0.9024390243902439 F値 0.8176795580110497

# 閾値
th = 0.875
print('閾値：',th)

# rocファイル書き込み
roc_output_option = input("ROC用のデータを出力しますか?(Y/N)")
y_true = []
y_pred = []

for s1, s2, l in zip(sent1, sent2, labels):
    sentence_bert_label = ''
    index += 1

    # rocファイルへの書き込み
    if l == 'entail':
        y_true.append(1)
        y_true.append(',')
    else:
        y_true.append(0)
        y_true.append(',')

    # Two lists of sentences
    sentences1 = [s1]
    sentences2 = [s2]

    #Compute embedding for both lists
    embeddings1 = model.encode(sentences1, convert_to_tensor=True)
    embeddings2 = model.encode(sentences2, convert_to_tensor=True)

    #Compute cosine-similarits
    cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)

    # rocファイルへの書き込み
    y_pred.append(cosine_scores.item())
    y_pred.append(',')

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


# roc抽出したいなら (Y/N) Yを入力
if roc_output_option == 'Y':
    with open('sbert_roc_data.txt','a') as f:
        for d in y_true:
            f.write("%s" % d)
        f.write("\n")
        for t in y_pred:
            f.write("%s" % t)
    f.close()


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
# 不正解問題数:y_true.count(0),正解問題数:y_true.count(1)

# 閾値
print('閾値：',th)
# 不正解文について
print('tp_cont_bert_score', tp_contradiction_sentence_bert)
print('tp_ent_bert_score', tp_entail_bert_sentence_bert)
huseikai_pre = calc_precicsion(tp_contradiction_sentence_bert, y_true.count(1) - tp_entail_bert_sentence_bert)
huseikai_rec = calc_recall(tp_contradiction_sentence_bert, y_true.count(0) - tp_contradiction_sentence_bert)
huseikai_f = calc_f(huseikai_pre, huseikai_rec)
print('=============不正解文================')
print('誤り検出あり：', tp_contradiction_sentence_bert, '誤り検出無し；', y_true.count(0) - tp_contradiction_sentence_bert, '適合率；', huseikai_pre , '再現率：', huseikai_rec, 'F値', huseikai_f)

# 正解文について
seikai_pre = calc_precicsion(tp_entail_bert_sentence_bert, y_true.count(0) - tp_contradiction_sentence_bert)
seikai_rec = calc_recall(tp_entail_bert_sentence_bert, y_true.count(1) - tp_entail_bert_sentence_bert)
seikai_f = calc_f(seikai_pre, seikai_rec)
print('=============正解文================')
print('誤り検出あり：', y_true.count(1) - tp_entail_bert_sentence_bert, '誤り検出無し；', tp_entail_bert_sentence_bert, '適合率；', seikai_pre , '再現率：', seikai_rec, 'F値', seikai_f)
