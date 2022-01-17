from bert_nli import BertNLIModel
import pandas as pd

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


sent1, sent2, labels = load_dataset('中間発表論文データ.csv')

# 問題番号
index = 0
# 問題数
acc = len(sent1)
print('len(acc)', acc)
# nliでの不正解、正解それぞれについてのtpを初期化
tp_contradiction_nli = 0
tp_entail_nli = 0
#bert_type = 'bert-base'
# 閾値
th = 0.95
print('閾値：',th)

# rocファイル書き込み
y_pred = []


# ファインチューニング済のモデルを読み込む
model = BertNLIModel('./nli_model_acc0.8831943861332694.state_dict')
for s1, s2, l in zip(sent1, sent2, labels):
    # ========================  以下nli  ================================
    # 両方向で含意関係認識
    sent_pairs_right = [(s1, s2)]
    sent_pairs_left = [(s2, s1)]
    nli_label_right, prob_right = model(sent_pairs_right)
    nli_label_left, prob_left = model(sent_pairs_left)
    nli_label = ''

    # rocファイルへの書き込み
    y_pred.append((prob_right[0][1].item() + prob_left[0][1]) / 2)
    y_pred.append(',')

    # contradiction entail neutralの順番で2次元配列に格納されている
    if prob_right[0][1].item() > th and prob_left[0][1].item() > th:
        nli_label = 'entail'
        # nliでentailと判断するかつ、実際に正解文である時
        if nli_label == l:
            tp_entail_nli += 1
    else:
        nli_label = 'contradiction'
        # nliでcontradictionと判断するかつ、実際に不正解文章である時
        if nli_label == l:
            tp_contradiction_nli += 1
    index += 1
    print('【nli】>>>', '問題番号：', index, '正解ラベル：', l, '予測：', nli_label)


with open('bertnli_roc_data.txt','a') as f:
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
# 不正解問題数:164,正解問題数:164

# 閾値
print('閾値：',th)

# 不正解文について
print('tp_cont_nli',tp_contradiction_nli)
print('tp_ent_nli',tp_entail_nli)
huseikai_pre = calc_precicsion(tp_contradiction_nli, 164 - tp_entail_nli)
huseikai_rec = calc_recall(tp_contradiction_nli, 164 - tp_contradiction_nli)
huseikai_f = calc_f(huseikai_pre, huseikai_rec)
print('=============不正解文================')
print('誤り検出あり：', tp_contradiction_nli, '誤り検出無し；', 164 - tp_contradiction_nli, '適合率；', huseikai_pre , '再現率：', huseikai_rec, 'F値', huseikai_f)

# 正解文について
seikai_pre = calc_precicsion(tp_entail_nli, 164 - tp_contradiction_nli)
seikai_rec = calc_recall(tp_entail_nli, 164 - tp_entail_nli)
seikai_f = calc_f(seikai_pre, seikai_rec)
print('=============正解文================')
print('誤り検出あり：', 164 - tp_entail_nli, '誤り検出無し；', tp_entail_nli, '適合率；', seikai_pre , '再現率：', seikai_rec, 'F値', seikai_f)
