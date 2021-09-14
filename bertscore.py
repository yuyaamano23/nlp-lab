from bert_score import score

def calc_bert_score(cands, refs):
    """ BERTスコアの算出
​
    Args:
        cands ([List[str]]): [比較元の文]
        refs ([List[str]]): [比較対象の文]
​
    Returns:
        [(List[float], List[float], List[float])]: [(Precision, Recall, F1スコア)]
    """
    Precision, Recall, F1 = score(cands, refs, lang="en", verbose=True)
    return Precision.numpy().tolist(), Recall.numpy().tolist(), F1.numpy().tolist()


if __name__ == "__main__":
    refs = ["Personal computers have become much cheaper and therefore more widespread"]
    cands = ["The fact that the price has become considerably cheaper doesn't have spurred the spread of personal computers"]

    P, R, F1 = calc_bert_score(cands, refs)
    print(F1)
    for p,r, f1 in zip(P, R, F1):
        print("P:%f, R:%f, F1:%f" %(p, r, f1))
