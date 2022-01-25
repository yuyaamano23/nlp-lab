import pandas as pd

# csvデータセットを整形
def load_dataset(filepath, encoding='utf-8'):
    # read_csv()は区切り文字がカンマ,でread_table()は区切り文字がタブ\t。
    df = pd.read_csv(filepath, encoding=encoding)
    # 意図せず\sが出てきてしまっていたので消す
    df['解答文'] = df['解答文'].replace("\'s", '')
    df['解答文'] = df['解答文'].str.split()
    df['正答文'] = df['正答文'].replace("\'s", '')
    df['正答文'] = df['正答文'].str.split()
    df['110のみ正解とする'] = df['110のみ正解とする'].replace("\'s", '')
    a = []
    b = []
    c = []
    d = []
    e = []
    f = []
    g = []
    for d1 in df['解答文']:
        a.append(d1)
    for d2 in df['正答文']:
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


s_kaito, s_monndai, L111, L010, L000, L011, L110 = load_dataset('./mix_label前半.csv')

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

# 閾値： 0.65 L111
# tp_entail_sbert 21
# tp_contradiction_sbert 152
# =============不正解文================
# 誤り検出あり： 152 誤り検出無し； 9 適合率； 0.6909090909090909 再現率： 0.9440993788819876 F値 0.7979002624671916
# =============正解文================
# 誤り検出あり： 68 誤り検出無し； 21 適合率； 0.7 再現率： 0.23595505617977527 F値 0.3529411764705882

# 閾値： 0.65 L110
# tp_entail_sbert 22
# tp_contradiction_sbert 140
# =============不正解文================
# 誤り検出あり： 140 誤り検出無し； 8 適合率； 0.6363636363636364 再現率： 0.9459459459459459 F値 0.7608695652173914
# =============正解文================
# 誤り検出あり： 80 誤り検出無し； 22 適合率； 0.7333333333333333 再現率： 0.21568627450980393 F値 0.3333333333333333

# 閾値： 0.6 L011
# tp_entail_sbert 37
# tp_contradiction_sbert 119
# =============不正解文================
# 誤り検出あり： 119 誤り検出無し； 6 適合率； 0.5748792270531401 再現率： 0.952 F値 0.7168674698795181
# =============正解文================
# 誤り検出あり： 88 誤り検出無し； 37 適合率； 0.8604651162790697 再現率： 0.296 F値 0.44047619047619047

# 閾値： 0.5 L010
# tp_entail_sbert 62
# tp_contradiction_sbert 85
# =============不正解文================
# 誤り検出あり： 85 誤り検出無し； 14 適合率； 0.4885057471264368 再現率： 0.8585858585858586 F値 0.6227106227106227
# =============正解文================
# 誤り検出あり： 89 誤り検出無し； 62 適合率； 0.8157894736842105 再現率： 0.4105960264900662 F値 0.5462555066079295


# 250問(前半)ずつで検証

# 閾値： 0.65 L111
# tp_entail_sbert 37
# tp_contradiction_sbert 319
# =============不正解文================
# 誤り検出あり： 319 誤り検出無し； 14 適合率； 0.7104677060133631 再現率： 0.9579579579579579 F値 0.8158567774936061
# =============正解文================
# 誤り検出あり： 130 誤り検出無し； 37 適合率； 0.7254901960784313 再現率： 0.2215568862275449 F値 0.3394495412844037

# 閾値： 0.65 L110
# tp_entail_sbert 38
# tp_contradiction_sbert 290
# =============不正解文================
# 誤り検出あり： 290 誤り検出無し； 13 適合率； 0.6458797327394209 再現率： 0.9570957095709571 F値 0.7712765957446809
# =============正解文================
# 誤り検出あり： 159 誤り検出無し； 38 適合率； 0.7450980392156863 再現率： 0.19289340101522842 F値 0.3064516129032258

# 閾値： 0.6 L011
# tp_entail_sbert 70
# tp_contradiction_sbert 240
# =============不正解文================
# 誤り検出あり： 240 誤り検出無し； 11 適合率； 0.5727923627684964 再現率： 0.9561752988047809 F値 0.7164179104477613
# =============正解文================
# 誤り検出あり： 179 誤り検出無し； 70 適合率； 0.8641975308641975 再現率： 0.28112449799196787 F値 0.4242424242424242

# 閾値： 0.5 L010
# tp_entail_sbert 113
# tp_contradiction_sbert 175
# =============不正解文================
# 誤り検出あり： 175 誤り検出無し； 22 適合率； 0.4794520547945205 再現率： 0.8883248730964467 F値 0.6227758007117437
# =============正解文================
# 誤り検出あり： 190 誤り検出無し； 113 適合率； 0.837037037037037 再現率： 0.37293729372937295 F値 0.5159817351598174



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
    for i in s_kaito:
        # sentence_bertでentailと予測するかつ、実際に正解文である時
        if (len(set(s_kaito[index]) & set(s_monndai[index])) / len(s_kaito[index])) > th  and L[index] == 1:
            tp_entail_sentence_bert += 1

        # sentence_bertでcontradictionと判断するかつ、実際に不正解文である時
        elif (len(set(s_kaito[index]) & set(s_monndai[index])) / len(s_kaito[index])) <= th and L[index] == 0:
            tp_contradiction_sentence_bert += 1
        index += 1
        print(index)
        print(len(set(s_kaito[index]) & set(s_monndai[index])) / len(s_kaito[index]))

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