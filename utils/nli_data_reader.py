from utils.input_example import InputExample
import pandas as pd
import csv
import gzip
import os

def load_csvdataset(filepath, encoding='utf-8'):
    df = pd.read_csv(filepath,
                     encoding=encoding,
                     )
    a=[]
    b=[]
    c=[]
    for d1 in df['t1']:
        a.append(d1)
    for d2 in df['t2']:
        b.append(d2)
    for l in df['label']:
        if l==0:
            ll='contradiction'
        elif l==1:
            ll='entailment'
        c.append(ll)
    return a,b,c
def load_csvdataset2(filepath, encoding='utf-8'):
    df = pd.read_csv(filepath,
                     encoding=encoding,
                     )
    a=[]
    b=[]
    c=[]
    for d1 in df['t1']:
        a.append(d1)
    for d2 in df['t2']:
        b.append(d2)
    for l in df['label']:
        c.append(l)
    return a,b,c

def load_jsondataset(filepath, encoding='utf-8'):
    df = pd.read_json(filepath,
                     encoding=encoding,
                     lines=True
                     )
    a=[]
    b=[]
    c=[]
    for d1 in df['hypothesis']:
        a.append(d1)
    for d2 in df['premise']:
        b.append(d2)
    for l in df['label']:
        if l=='not_entailment':
            ll='contradiction'
        c.append(ll)
    return a,b,c

class NLIDataReader(object):
    """
    Reads in the Stanford NLI dataset and the MultiGenre NLI dataset
    """
    def __init__(self, dataset_folder):
        self.dataset_folder = dataset_folder

    def get_hans_examples(self, filename, max_examples=0):
        df = pd.read_csv( os.path.join(self.dataset_folder,filename), sep='\t' )

        examples = []
        for idx,entry in df.iterrows():
            guid = 'hans-{}'.format(idx)
            examples.append( InputExample(guid=guid, texts=[entry['sentence1'], entry['sentence2']], label=self.map_hans_label(entry['gold_label'])) )

            if 0 < max_examples <= len(examples):
                break

        return examples

    def get_examples(self, filename, max_examples=0):
        """
        data_splits specified which data split to use (train, dev, test).
        Expects that self.dataset_folder contains the files s1.$data_split.gz,  s2.$data_split.gz,
        labels.$data_split.gz, e.g., for the train split, s1.train.gz, s2.train.gz, labels.train.gz
        """

        s1 = gzip.open(os.path.join(self.dataset_folder, 's1.' + filename),
                       mode="rt", encoding="utf-8").readlines()
        s2 = gzip.open(os.path.join(self.dataset_folder, 's2.' + filename),
                       mode="rt", encoding="utf-8").readlines()
        labels = gzip.open(os.path.join(self.dataset_folder, 'labels.' + filename),
                           mode="rt", encoding="utf-8").readlines()

        s1=[s.replace("\n", '') for s in s1]
        s2=[s.replace("\n", '') for s in s2]
        labels=[l.replace("\n", '') for l in labels]
        ads1,ads2,adlabels=load_csvdataset('/home/matsui/zemi/データ整形/データ/total.csv')
        adh,adp,adl=load_jsondataset('/home/matsui/ダウンロード/RTE/train.jsonl')
        s1+=ads1+adh
        s2+=ads2+adp
        labels+=adlabels+adl
        examples = []
        id = 0
        for sentence_a, sentence_b, label in zip(s1, s2, labels):
            #print(type(label))
            guid = "%s-%d" % (filename, id)
            #print(guid)
            id += 1
            examples.append(InputExample(guid=guid, texts=[sentence_a, sentence_b], label=self.map_label(label)))

            if 0 < max_examples <= len(examples):
                break

        return examples
    def get_testexamples(self, filename, max_examples=0):
        s1,s2,labels=load_csvdataset2(filename)
        examples = []
        id = 0
        for sentence_a, sentence_b, label in zip(s1, s2, labels):
            #print(type(label))
            guid = "%s-%d" % (filename, id)
            #print(guid)
            id += 1
            examples.append(InputExample(guid=guid, texts=[sentence_a, sentence_b], label=self.map_label(label)))

            if 0 < max_examples <= len(examples):
                break

        return examples

    @staticmethod
    def get_labels():
        return {"contradiction": 0, "entailment": 1, "neutral": 2}

    @staticmethod
    def get_hans_labels():
        return {"entailment": 1, "non-entailment": 2}

    def get_num_labels(self):
        #print(len(self.get_labels()))
        return len(self.get_labels())

    def map_label(self, label):
        return self.get_labels()[label.strip().lower()]

    def map_hans_label(self, label):
        return self.get_hans_labels()[label.strip().lower()]
