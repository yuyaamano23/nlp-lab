from bert_nli import BertNLIModel

if __name__ == '__main__':
    bert_type = 'bert-large'
    model = BertNLIModel('/Users/ariyoshihiroyuki/myworkspace/研究室/prog/bert_nli/nli_model_acc0.8831943861332694.state_dict')
    sent_pairs = [('I like baseball','I like sports')]#s1,s2
    sent_pairs1=[(sent_pairs[0][1],sent_pairs[0][0])]
    labels, probs = model(sent_pairs)
    labels1,probs1=model(sent_pairs1)
    print(labels,probs)#s1karas2#label[contradiction,entail,neutral]
    print(labels1,probs1)#s2karas1
