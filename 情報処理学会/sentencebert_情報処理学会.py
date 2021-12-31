import pandas as pd

# csvデータセットを整形
def load_dataset(filepath, encoding='utf-8'):
    # read_csv()は区切り文字がカンマ,でread_table()は区切り文字がタブ\t。
    df = pd.read_csv(filepath, encoding=encoding)
    # 意図せず\sが出てきてしまっていたので消す
    df['採点フラグ'] = df['採点フラグ'].replace("\'s", '')
    df['0のみ不正解とする'] = df['0のみ不正解とする'].replace("\'s", '')
    a = []
    b = []
    c = []
    d = []
    e = []
    f = []
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
    return a, b, c, d, e, f


flag, JEscore, EEscore, L111, L010, L000 = load_dataset('./アウトソーシング検証用データ.csv')


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
L = L111

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

