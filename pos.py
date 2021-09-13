from bert_nli import BertNLIModel
import nltk
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import numpy

def create_mask_set(sent,swa):
    sentences = []
    sent = sent.strip().split()
    pos=nltk.pos_tag(sent)
    for i in range(len(sent)):
        if i==0:
            new_sent = sent[:]
            new_sent.insert(i, '[MASK]')
            text = " ".join(new_sent)
            sentences.append(text)

        elif i==len(sent)-1:
            new_sent = sent[:]
            new_sent.insert(len(sent), '[MASK]')
            text = " ".join(new_sent)
            sentences.append(text)
        if pos[i][1] in swa:
            if i==0:
                new_sent = sent[:]
                new_sent[i] = '[MASK]'
                text = " ".join(new_sent)
                sentences.append(text)
            elif i==len(sent)-1:
                new_sent = sent[:]
                new_sent[i] = '[MASK]'
                text = " ".join(new_sent)
                sentences.append(text)
            else:
                new_sent = sent[:]
                new_sent[i] = '[MASK]'
                text = " ".join(new_sent)
                sentences.append(text)
                new_sent=sent[:]
                new_sent.insert(i, '[MASK]')
                text1=" ".join(new_sent)
                sentences.append(text1)
                new_sent=sent[:]
                new_sent.insert(i+1, '[MASK]')
                text2=" ".join(new_sent)
                sentences.append(text2)

    return sentences

string =""
mis_string="Sota lives in Hamamatsu City, Mie Prefecture."
kflag=0
if string[-1]=='.':
    string=string[:-1]
#if ',' in string:
#    n=string.find(',')
#    string=string[:n]+' '+string[n:]

if mis_string[-1]=='.':
    mis_string=mis_string[:-1]
if ',' in mis_string:
    kflag=1
    kst=''
    n=mis_string.find(',')
    for i in range(n+2,len(mis_string)):
        if mis_string[i]!=' ':
            kst+=mis_string[i]
        else:
            mis_string=mis_string[:n]+mis_string[n+1:]
            break
#    mis_string=mis_string[:n]+' '+mis_string[n:]
model = BertNLIModel('/home/matsui/bert_nli/output/nli_bert-large-2020-11-07_17-23-12/nli_model_acc0.8831943861332694.state_dict')
sent_check1=[(string,mis_string)]
sent_check2=[(mis_string,string)]
labels,a = model(sent_check1)
labels1,b=model(sent_check2)
print(labels,labels1)
swa1_poses=['JJ','JJR','JJS','VB','NNS','VBG','NN','NNS','RB','RBR','RBS','VBD','VBG','VBN','VBP','VBZ']
swa2_poses=['WDT','WRB','WP','WBR']
swa3_poses=['RP',]
iso_poses=['CD','DT','MD','PDT','PRP','PRP$','NNP','NNPS']
poses=[swa1_poses,swa2_poses,swa3_poses,iso_poses]
if labels[0]=='entail' and labels1[0]=='entail':
    pass
else:
    words = nltk.word_tokenize(string)
    pos=nltk.pos_tag(words)
    for po in poses:
        #print(po)
        mask_string=create_mask_set(mis_string,po)
        #print(mask_string)
        cand=[]
        for p in pos:
            if p[1] in po:
                cand.append(p[0])
        predict=[]
        precand=[]
        for m in mask_string:
            for c in cand:
                text = m.strip().split()
                mask_index = text.index('[MASK]')
                text[mask_index]=c
                new_sent = " ".join(text)
                predict.append(new_sent)
                precand.append(c)
        ans=[]
        lab=[]
        pcl=[]
        #print(predict)
        for p,pc in zip(predict,precand):
            sent_pairs = [(string,p)]
            sent_pairs1=[(sent_pairs[0][1],sent_pairs[0][0])]
            print(p)
            labels, probs = model(sent_pairs)
            labels1,probs1=model(sent_pairs1)
            print(labels,probs)
            print(labels1,probs1)
            if labels[0]=='entail' and labels1[0]=='entail':
                print(pc)
                lab.append(probs[0][1]+probs1[0][1])
                ans.append(p)
                pcl.append(pc)
        print('---------------------------')
        if len(ans)!=0:
            print(ans)
            max_index=lab.index(max(lab))
            if kflag==0:
                print(ans[max_index]+'.')
                break
            else:
                s=ans[max_index]
                if kst in s:
                    n=s.find(kst)
                    print(s[:n-1]+', '+s[n:]+'.')
                else:
                    n=s.find(pcl[max_index])
                    print(s[:n-1]+', '+s[n:]+'.')
                break
