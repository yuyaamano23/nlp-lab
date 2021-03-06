import pandas as pd

# csvデータセットを整形
def load_dataset(filepath, encoding='utf-8'):
    # read_csv()は区切り文字がカンマ,でread_table()は区切り文字がタブ\t。
    df = pd.read_csv(filepath, encoding=encoding)
    # 意図せず\sが出てきてしまっていたので消す
    df['ラベルの値_解答to正答'] = df['ラベルの値_解答to正答'].replace("\'s", '')
    df['110のみ正解とする'] = df['110のみ正解とする'].replace("\'s", '')
    a = []
    b = []
    c = []
    d = []
    e = []
    f = []
    g = []
    for d1 in df['ラベルの値_解答to正答']:
        a.append(d1)
    for d2 in df['ラベルの値_正答to解答']:
        b.append(d2)
    for d3 in df['111のみ正解とする']:
        c.append(d3)
    for d4 in df['010のみ正解とする']:
        d.append(d4)
    for d5 in df['0のみ不正解とする']:
        e.append(d5)
    for d6 in df['011のみ正解とする']:
        f.append(d6)
    for d7 in df['110のみ正解とする']:
        g.append(d7)
    return a, b, c, d, e, f, g


labelKaitoSeito, labelSeitoKaito, L111, L010, L000, L011, L110 = load_dataset('./双方向データ250後半.csv')


# ここでどのフラグを採用するか選択
L_input = input("ラベルを入力してください(L111,L110,L011,L010)：")

if L_input == "L111":
    L = L111
elif L_input == "L110":
    L = L110
elif L_input == "L011":
    L = L011
elif L_input == "L010":
    L = L010
else:
    print("正しい値を入力してください")

# 問題数
print("問題数", len(L))
print("正解数：",L.count(1),"不正解数：",L.count(0))

# 250問(後半)ずつで閾値求める

# 閾値： 0.998 L111
# tp_entail_sbert 12
# tp_contradiction_sbert 118
# =============不正解文================
# 誤り検出あり： 118 誤り検出無し； 7 適合率； 0.5108225108225108 再現率： 0.944 F値 0.6629213483146067
# =============正解文================
# 誤り検出あり： 113 誤り検出無し； 12 適合率； 0.631578947368421 再現率： 0.096 F値 0.16666666666666669

# 閾値： 0.997 L110
# tp_entail_sbert 30
# tp_contradiction_sbert 101
# =============不正解文================
# 誤り検出あり： 101 誤り検出無し； 24 適合率； 0.5153061224489796 再現率： 0.808 F値 0.6292834890965733
# =============正解文================
# 誤り検出あり： 95 誤り検出無し； 30 適合率； 0.5555555555555556 再現率： 0.24 F値 0.33519553072625696

# 閾値： 0.998 L011
# tp_entail_sbert 12
# tp_contradiction_sbert 118
# =============不正解文================
# 誤り検出あり： 118 誤り検出無し； 7 適合率； 0.5108225108225108 再現率： 0.944 F値 0.6629213483146067
# =============正解文================
# 誤り検出あり： 113 誤り検出無し； 12 適合率； 0.631578947368421 再現率： 0.096 F値 0.16666666666666669

# 閾値： 0.998 L010
# tp_entail_sbert 12
# tp_contradiction_sbert 118
# =============不正解文================
# 誤り検出あり： 118 誤り検出無し； 7 適合率； 0.5108225108225108 再現率： 0.944 F値 0.6629213483146067
# =============正解文================
# 誤り検出あり： 113 誤り検出無し； 12 適合率； 0.631578947368421 再現率： 0.096 F値 0.16666666666666669


# 250問(前半)ずつで検証

# 閾値： 0.998 L111
# tp_entail_sbert 11
# tp_contradiction_sbert 122
# =============不正解文================
# 誤り検出あり： 122 誤り検出無し； 4 適合率； 0.5191489361702127 再現率： 0.9682539682539683 F値 0.6759002770083102
# =============正解文================
# 誤り検出あり： 113 誤り検出無し； 11 適合率； 0.7333333333333333 再現率： 0.08870967741935484 F値 0.15827338129496402

# 閾値： 0.997 L110
# tp_entail_sbert 27
# tp_contradiction_sbert 111
# =============不正解文================
# 誤り検出あり： 111 誤り検出無し； 15 適合率； 0.5336538461538461 再現率： 0.8809523809523809 F値 0.6646706586826346
# =============正解文================
# 誤り検出あり： 97 誤り検出無し； 27 適合率； 0.6428571428571429 再現率： 0.21774193548387097 F値 0.3253012048192771

# 閾値： 0.998 L011
# tp_entail_sbert 11
# tp_contradiction_sbert 122
# =============不正解文================
# 誤り検出あり： 122 誤り検出無し； 4 適合率； 0.5191489361702127 再現率： 0.9682539682539683 F値 0.6759002770083102
# =============正解文================
# 誤り検出あり： 113 誤り検出無し； 11 適合率； 0.7333333333333333 再現率： 0.08870967741935484 F値 0.15827338129496402

# 閾値： 0.998 L010
# tp_entail_sbert 11
# tp_contradiction_sbert 122
# =============不正解文================
# 誤り検出あり： 122 誤り検出無し； 4 適合率； 0.5191489361702127 再現率： 0.9682539682539683 F値 0.6759002770083102
# =============正解文================
# 誤り検出あり： 113 誤り検出無し； 11 適合率； 0.7333333333333333 再現率： 0.08870967741935484 F値 0.15827338129496402




# 問題番号
index = 0



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

# 標準入力されたそれぞれの閾値で判定
for th in th_input_list:
    # sbertでの不正解、正解それぞれについてのtpを初期化
    tp_contradiction_sentence_bert = 0
    tp_entail_sentence_bert = 0
    # データセットの計算結果を2次元配列を展開
    index = 0
    for i in labelKaitoSeito:
        # sentence_bertでentailと予測するかつ、実際に正解文である時
        if ((labelKaitoSeito[index] + labelSeitoKaito[index]) / 2) > th  and L[index] == 1:
            tp_entail_sentence_bert += 1

        # sentence_bertでcontradictionと判断するかつ、実際に不正解文である時
        elif ((labelKaitoSeito[index] + labelSeitoKaito[index]) / 2) <= th and L[index] == 0:
            tp_contradiction_sentence_bert += 1
        index += 1

    # 結果の出力
    print('閾値：',th,L_input)
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


# ==========================不正解のみの値が最大となるデータ

# 閾値： 0.999 L111
# tp_entail_sbert 1
# tp_contradiction_sbert 333
# =============不正解文================
# 誤り検出あり： 333 誤り検出無し； 0 適合率； 0.6673346693386774 再現率： 1.0 F値 0.8004807692307693
# =============正解文================
# 誤り検出あり： 166 誤り検出無し； 1 適合率； 1.0 再現率： 0.005988023952095809 F値 0.011904761904761906

# 閾値： 0.999 L110
# tp_entail_sbert 1
# tp_contradiction_sbert 303
# =============不正解文================
# 誤り検出あり： 303 誤り検出無し； 0 適合率； 0.6072144288577155 再現率： 1.0 F値 0.7556109725685786
# =============正解文================
# 誤り検出あり： 196 誤り検出無し； 1 適合率； 1.0 再現率： 0.005076142131979695 F値 0.010101010101010102

# 閾値： 0.999 L011
# tp_entail_sbert 1
# tp_contradiction_sbert 251
# =============不正解文================
# 誤り検出あり： 251 誤り検出無し； 0 適合率； 0.503006012024048 再現率： 1.0 F値 0.6693333333333333
# =============正解文================
# 誤り検出あり： 248 誤り検出無し； 1 適合率； 1.0 再現率： 0.004016064257028112 F値 0.007999999999999998

# 閾値： 0.999 L010
# tp_entail_sbert 1
# tp_contradiction_sbert 197
# =============不正解文================
# 誤り検出あり： 197 誤り検出無し； 0 適合率； 0.39478957915831664 再現率： 1.0 F値 0.5660919540229885
# =============正解文================
# 誤り検出あり： 302 誤り検出無し； 1 適合率； 1.0 再現率： 0.0033003300330033004 F値 0.006578947368421053