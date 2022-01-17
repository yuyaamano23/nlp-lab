import pandas as pd

# csvデータセットを整形
def load_dataset(filepath, encoding='utf-8'):
    # read_csv()は区切り文字がカンマ,でread_table()は区切り文字がタブ\t。
    df = pd.read_csv(filepath, encoding=encoding)
    # 意図せず\sが出てきてしまっていたので消す
    df['採点フラグ'] = df['採点フラグ'].replace("\'s", '')
    df['110のみ正解とする'] = df['110のみ正解とする'].replace("\'s", '')
    a = []
    b = []
    c = []
    d = []
    e = []
    f = []
    g = []
    h =[]
    for d1 in df['採点フラグ']:
        a.append(d1)
    for d2 in df['J-Escore']:
        b.append(d2)
    for d3 in df['E-Escore']:
        c.append(d3)
    for d4 in df['111のみ正解とする']:
        d.append(d4)
    for d5 in df['010のみ正解とする']:
        e.append(d5)
    for d6 in df['0のみ不正解とする']:
        f.append(d6)
    for d7 in df['011のみ正解とする']:
        g.append(d7)
    for d8 in df['110のみ正解とする']:
        h.append(d8)
    return a, b, c, d, e, f, g, h


flag, JEscore, EEscore, L111, L010, L000, L011, L110 = load_dataset('./アウトソーシング検証用データ.csv')


# 問題番号
index = 0
# 問題数
acc = len(flag)
print('len(acc)', acc)



# 閾値を格納する配列
th_input_list = []
th_input = 0
while th_input != '':
    th_input = input("閾値を入力してください：")
    if th_input == '':
        break
    th_input_list.append(float(th_input))


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

# ここでどのフラグを採用するか選択
L = L010

# 標準入力されたそれぞれの閾値で判定
for th in th_input_list:
    # sbertでの不正解、正解それぞれについてのtpを初期化
    tp_contradiction_sentence_bert = 0
    tp_entail_sentence_bert = 0
    # データセットの計算結果を2次元配列を展開
    index = 0
    for i in JEscore:
        # sentence_bertでentailと予測するかつ、実際に正解文である時
        if JEscore[index] > th  and L[index] == 1:
            tp_entail_sentence_bert += 1

        # sentence_bertでcontradictionと判断するかつ、実際に不正解文である時
        elif JEscore[index] <= th and L[index] == 0:
            tp_contradiction_sentence_bert += 1
        index += 1

    # 結果の出力
    print('閾値：',th)
    # 不正解問題数:y_true.count(0),正解問題数:y_true.count(1)
    # 不正解文について
    print('tp_entail_sbert', tp_entail_sentence_bert)
    print('tp_contradiction_sbert', tp_contradiction_sentence_bert)
    huseikai_pre = calc_precicsion(tp_contradiction_sentence_bert, L.count(1) - tp_entail_sentence_bert)
    huseikai_rec = calc_recall(tp_contradiction_sentence_bert, L.count(0) - tp_contradiction_sentence_bert)
    huseikai_f = calc_f(huseikai_pre, huseikai_rec)
    print('=============不正解文================')
    print('誤り検出あり：', tp_contradiction_sentence_bert, '誤り検出無し；', L.count(0) - tp_contradiction_sentence_bert, '適合率；', huseikai_pre , '再現率：', huseikai_rec, 'F値', huseikai_f)

    # 正解文について
    seikai_pre = calc_precicsion(tp_entail_sentence_bert, L.count(0) - tp_contradiction_sentence_bert)
    seikai_rec = calc_recall(tp_entail_sentence_bert, L.count(1) - tp_entail_sentence_bert)
    seikai_f = calc_f(seikai_pre, seikai_rec)
    print('=============正解文================')
    print('誤り検出あり：', L.count(1) - tp_entail_sentence_bert, '誤り検出無し；', tp_entail_sentence_bert, '適合率；', seikai_pre , '再現率：', seikai_rec, 'F値', seikai_f)

    print("===================================================================================")


# ==========================正解不正解両方のF値の合計が最大となるデータ
# 閾値： 0.93 L111
# tp_entail_sbert 75
# tp_contradiction_sbert 274
# =============不正解文================
# 誤り検出あり： 274 誤り検出無し； 59 適合率； 0.7486338797814208 再現率： 0.8228228228228228 F値 0.7839771101573677
# =============正解文================
# 誤り検出あり： 92 誤り検出無し； 75 適合率； 0.5597014925373134 再現率： 0.4491017964071856 F値 0.49833887043189373
# 合計1.281

# 閾値： 0.92 L11%
# tp_entail_sbert 94
# tp_contradiction_sbert 234
# =============不正解文================
# 誤り検出あり： 234 誤り検出無し； 69 適合率； 0.6943620178041543 再現率： 0.7722772277227723 F値 0.73125
# =============正解文================
# 誤り検出あり： 103 誤り検出無し； 94 適合率； 0.5766871165644172 再現率： 0.47715736040609136 F値 0.5222222222222221
# 合計1.253
# スペルミスで低くスコアがでるため許容しない?

# 閾値： 0.86 L%11
# tp_entail_sbert 194
# tp_contradiction_sbert 138
# =============不正解文================
# 誤り検出あり： 138 誤り検出無し； 113 適合率； 0.7150259067357513 再現率： 0.549800796812749 F値 0.6216216216216217
# =============正解文================
# 誤り検出あり： 55 誤り検出無し； 194 適合率； 0.6319218241042345 再現率： 0.7791164658634538 F値 0.697841726618705
# 合計1.318

# 閾値： 0.83 L%1%
# tp_entail_sbert 253
# tp_contradiction_sbert 102
# =============不正解文================
# 誤り検出あり： 102 誤り検出無し； 95 適合率； 0.6710526315789473 再現率： 0.5177664974619289 F値 0.5845272206303724
# =============正解文================
# 誤り検出あり： 50 誤り検出無し； 253 適合率； 0.7270114942528736 再現率： 0.834983498349835 F値 0.7772657450076805
# 合計1.361


# 使わない
# 閾値： 0.83 L000
# tp_entail_sbert 310
# tp_contradiction_sbert 66
# =============不正解文================
# 誤り検出あり： 66 誤り検出無し； 38 適合率； 0.4342105263157895 再現率： 0.6346153846153846 F値 0.515625
# =============正解文================
# 誤り検出あり： 86 誤り検出無し； 310 適合率； 0.8908045977011494 再現率： 0.7828282828282829 F値 0.8333333333333334
# 合計1.348


# ==========================不正解のみの値の合計が最大となるデータ

# 閾値： 0.95 L111
# tp_entail_sbert 49
# tp_contradiction_sbert 308
# =============不正解文================
# 誤り検出あり： 308 誤り検出無し； 25 適合率； 0.7230046948356808 再現率： 0.924924924924925 F値 0.8115942028985508
# =============正解文================
# 誤り検出あり： 118 誤り検出無し； 49 適合率； 0.6621621621621622 再現率： 0.2934131736526946 F値 0.4066390041493776

# 閾値： 0.95 L110
# tp_entail_sbert 53
# tp_contradiction_sbert 282
# =============不正解文================
# 誤り検出あり： 282 誤り検出無し； 21 適合率； 0.6619718309859155 再現率： 0.9306930693069307 F値 0.7736625514403292
# =============正解文================
# 誤り検出あり： 144 誤り検出無し； 53 適合率； 0.7162162162162162 再現率： 0.26903553299492383 F値 0.39114391143911437

# 閾値： 0.935 L011
# tp_entail_sbert 94
# tp_contradiction_sbert 227
# =============不正解文================
# 誤り検出あり： 227 誤り検出無し； 24 適合率； 0.5942408376963351 再現率： 0.9043824701195219 F値 0.7172195892575038
# =============正解文================
# 誤り検出あり： 155 誤り検出無し； 94 適合率； 0.7966101694915254 再現率： 0.37751004016064255 F値 0.5122615803814714

# 閾値： 0.905 L010
# tp_entail_sbert 161
# tp_contradiction_sbert 160
# =============不正解文================
# 誤り検出あり： 160 誤り検出無し； 37 適合率； 0.5298013245033113 再現率： 0.8121827411167513 F値 0.6412825651302606
# =============正解文================
# 誤り検出あり： 142 誤り検出無し； 161 適合率； 0.8131313131313131 再現率： 0.5313531353135313 F値 0.6427145708582835
