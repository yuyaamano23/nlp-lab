pre_input = float(input("適合率を入力してください："))
rec_input = float(input("再現率を入力してください："))

# F値の算出
def calc_f(pre=0, rec=0):
    f = (2 * pre * rec) / (pre + rec)
    return f

print("f値",calc_f(pre_input,rec_input))
