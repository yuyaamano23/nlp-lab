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


sent1, sent2, labels = load_dataset('./自己採点.csv')

# 問題番号
index = 0
# 問題数
acc = len(sent1)
print('len(acc)', acc)

#bert_type = 'bert-base'


# データセットの計算結果格納2次元配列
dataset_output = []
# [正解ラベル,SBERTcos類似度],
# イメージ例
# ["entail",0.890904032940],
# ["contradiction",0.86519589198]]

# 閾値を格納する配列
th_input_list = []
th_input = 0
while th_input != '':
    th_input = input("閾値を入力してください：")
    if th_input == '':
        break
    th_input_list.append(float(th_input))

# rocファイル書き込み用の変数定義と標準出力
roc_output_option = input("ROC用のデータを出力しますか?(Y/N)")
y_true = []
y_pred = []

for s1, s2, l in zip(sent1, sent2, labels):
    index += 1

    # rocファイルへの書き込み用変数に要素追加
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

    # rocファイルへの書き込み用変数に要素追加
    y_pred.append(cosine_scores.item())
    y_pred.append(',')

    # データセットの計算結果を2次元配列に格納
    dataset_output.append([l,cosine_scores.item()])

    print('【',index,'】','正解ラベル：',l,'  cos類似度：',cosine_scores.item())


# =========================== 以下分析データ算出 =================================
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

# =================================================================================

# 標準入力されたそれぞれの閾値で判定
for th in th_input_list:
    # bert_scoreでの不正解、正解それぞれについてのtpを初期化
    tp_contradiction_sentence_bert = 0
    tp_entail_sentence_bert = 0
    # データセットの計算結果を2次元配列を展開
    for arr in dataset_output:
        # sentence_bertでentailと予測するかつ、実際に正解文である時
        if arr[1] > th  and arr[0] == 'entail':
            tp_entail_sentence_bert += 1

        # sentence_bertでcontradictionと判断するかつ、実際に不正解文である時
        elif arr[1] <= th and arr[0] == 'contradiction':
            tp_contradiction_sentence_bert += 1

    # 結果の出力
    print('閾値：',th)
    # 不正解問題数:y_true.count(0),正解問題数:y_true.count(1)
    # 不正解文について
    print('tp_entail_sbert', tp_entail_sentence_bert)
    print('tp_contradiction_sbert', tp_contradiction_sentence_bert)
    huseikai_pre = calc_precicsion(tp_contradiction_sentence_bert, y_true.count(1) - tp_entail_sentence_bert)
    huseikai_rec = calc_recall(tp_contradiction_sentence_bert, y_true.count(0) - tp_contradiction_sentence_bert)
    huseikai_f = calc_f(huseikai_pre, huseikai_rec)
    print('=============不正解文================')
    print('誤り検出あり：', tp_contradiction_sentence_bert, '誤り検出無し；', y_true.count(0) - tp_contradiction_sentence_bert, '適合率；', huseikai_pre , '再現率：', huseikai_rec, 'F値', huseikai_f)

    # 正解文について
    seikai_pre = calc_precicsion(tp_entail_sentence_bert, y_true.count(0) - tp_contradiction_sentence_bert)
    seikai_rec = calc_recall(tp_entail_sentence_bert, y_true.count(1) - tp_entail_sentence_bert)
    seikai_f = calc_f(seikai_pre, seikai_rec)
    print('=============正解文================')
    print('誤り検出あり：', y_true.count(1) - tp_entail_sentence_bert, '誤り検出無し；', tp_entail_sentence_bert, '適合率；', seikai_pre , '再現率：', seikai_rec, 'F値', seikai_f)

    print("===================================================================================")



# roc抽出したいなら (Y/N) Yを入力
if roc_output_option == 'Y':
    with open('sbert_roc_data.txt','a') as f:
        for d in y_true:
            f.write("%s" % d)
        f.write("\n")
        for t in y_pred:
            f.write("%s" % t)
    f.close()
