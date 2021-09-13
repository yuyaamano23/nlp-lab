import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import math
from difflib import SequenceMatcher
from bert_nli import BertNLIModel

det = ['the', 'a', 'an', 'this', 'that', 'these', 'those', 'my', 'your', 'his',
       'her', 'its', 'our', 'their', 'all', 'both', 'half', 'either', 'neither',
       'each', 'every', 'other', 'another', 'such', 'what', 'rather', 'quite']

# List of common prepositions
prep = ["about", "at", "by", "for", "from", "in", "of", "on", "to", "with",
        "into", "during", "including", "until", "against", "among",
        "throughout", "despite", "towards", "upon", "concerning"]

# List of helping verbs
helping_verbs = ['am', 'is', 'are', 'was', 'were', 'being', 'been', 'be',
                 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                 'shall', 'should', 'may', 'might', 'must', 'can', 'could']

def create_mask_set(spelling_sentences):
  sentences = []

  for sent in spelling_sentences:
    sent = sent.strip().split()
    for i in range(len(sent)):
      # (1) [MASK] each word
      new_sent = sent[:]
      new_sent[i] = '[MASK]'
      text = " ".join(new_sent)
      new_sent = '[CLS] ' + text + ' [SEP]'
      sentences.append(new_sent)

      # (2) [MASK] for each space between words
      new_sent = sent[:]
      new_sent.insert(i, '[MASK]')
      text = " ".join(new_sent)
      new_sent = '[CLS] ' + text + ' [SEP]'
      sentences.append(new_sent)

  return sentences

tokenizerLarge = BertTokenizer.from_pretrained('bert-large-uncased')
model = BertForMaskedLM.from_pretrained('bert-large-uncased')
org_sent='I go to a school.'
sentences=create_mask_set([org_sent])

n = len(sentences)
#print(sentences)
# what is the tokenized value of [MASK]. Usually 103
text = '[MASK]'
tokenized_text = tokenizerLarge.tokenize(text)
#print(tokenized_text)
mask_token = tokenizerLarge.convert_tokens_to_ids(tokenized_text)[0]

LM_sentences = []
new_sentences = []
spelling_sentences=[]
i = 0 # current sentence number
l = len(org_sent.strip().split())*2 # l is no of sentencees
mask = False # flag indicating if we are processing space MASK

for sent in sentences:
    i += 1

    # tokenize the text
    tokenized_text = tokenizerLarge.tokenize(sent)
    #print(tokenized_text)
    indexed_tokens = tokenizerLarge.convert_tokens_to_ids(tokenized_text)
    #print(indexed_tokens)
    # Create the segments tensors.
    segments_ids = [0] * len(tokenized_text)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    # Predict all tokens
    with torch.no_grad():
        predictions = model(tokens_tensor, segments_tensors)
    #print(predictions)
    # index of the masked token
    mask_index = (tokens_tensor == mask_token).nonzero()[0][1].item()
    #print(predictions[0, mask_index])
    # predicted token
    predicted_index = torch.argmax(predictions[0, mask_index]).item()
    _,predicted_index1 = torch.topk(predictions[0, mask_index],k=5,sorted=True)
    predicted_token = tokenizerLarge.convert_ids_to_tokens([predicted_index])[0]
    predicted_token1 = tokenizerLarge.convert_ids_to_tokens(predicted_index1.tolist())
    print(sent,predicted_token)
    #print(sent,predicted_token1)

    text = sent.strip().split()
    #print(text)
    # あくまで、ここのmaskは学習用ではなく、入力文を隠すため
    mask_index = text.index('[MASK]')

    if not mask:
      # case of MASKed words

      mask = True
      text[mask_index] = predicted_token
      try:
        # retrieve original word
        org_word = spelling_sentences[i//l].strip().split()[mask_index-1]
        #print(org_word)

      except:
        #print("!", end="")
        continue
      if SequenceMatcher(None, org_word, predicted_token).ratio() < 0.6:
        if org_word not in list(set(det + prep + helping_verbs)) or predicted_token not in list(set(det + prep + helping_verbs)):
          continue
      if org_word == predicted_token:
        #print(org_word)
        continue
    else:
      # case for MASKed spaces

      mask = False
  #     print("{0}".format(predicted_token))
      # only allow determiners / prepositions  / helping verbs in spaces

    text[mask_index] = predicted_token
    text.remove('[SEP]')
    text.remove('[CLS]')
    new_sent = " ".join(text)
    spelling_sentences.append(new_sent)
model = BertNLIModel('/home/matsui/bert_nli/output/nli_bert-large-2020-10-07_19-29-08/nli_model_acc0.8768068134109038.state_dict')
for s in spelling_sentences:
    sent_pairs = [('I go to a school.',s)]
    print(sent_pairs)
    labels, probs = model(sent_pairs)
    print(labels,probs)
