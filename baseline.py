#!/usr/bin/env python
# coding: utf-8

# # Sentence Boundary detection with NER features

# In[1]:


import os


# In[2]:


import torch
import pandas as pd
import numpy as np
import transformers
from tqdm import tqdm
import csv


# In[3]:


def set_newline_to_EOS(file_r, file_w):
    with open(file_w, 'w', encoding="UTF-8") as fw:
        with open(file_r, 'r', encoding="UTF-8") as fr:
            while True:
                line = fr.readline()
                if not line: break
                if line == '\n':
                    fw.write("<EOS>\t<EOS>\t<EOS>\t<EOS>\n")
                else:
                    fw.write(line)



train_file = './data/train.tsv'
test_file = './data/test.tsv'
train_EOS_file = './data/train_EOS.tsv'
test_EOS_file = './data/test_EOS.tsv'
#test_file = './data_v1/europarl-sbd-eval.tsv'

set_newline_to_EOS(train_file, train_EOS_file)
set_newline_to_EOS(test_file, test_EOS_file)
train_file = train_EOS_file
test_file = test_EOS_file


# In[4]:


train_df = pd.read_csv(train_file, delimiter='\t', engine='python', encoding='UTF-8', error_bad_lines=False, header=None, quoting=csv.QUOTE_NONE)
train_df.head()
test_df = pd.read_csv(test_file, delimiter='\t', engine='python', encoding='UTF-8', error_bad_lines=False, header=None, quoting=csv.QUOTE_NONE)
test_df.head()


# In[5]:


def set_sentence_num(df): 

    sent_num = 0
    df['sent_num'] = sent_num
    for idx in range(len(df)):
        df['sent_num'][idx] = sent_num
        #if df[0][idx]=='.' and df[1][idx]=="SENT":
        if df[0][idx]=='<EOS>' and df[1][idx]=="<EOS>":
            #df[0] = df[0].drop(df[0].index[idx])
            sent_num +=1
    df.head()
    
    df = df[df[0] != "<EOS>"] #get rid of the "<EOS> toekns"
    
    print(sent_num)
    
    return df


# In[6]:


train_df = set_sentence_num(train_df)
test_df = set_sentence_num(test_df)


# In[7]:


train_df = train_df[train_df['sent_num'] < 1000000]
test_df = test_df[test_df['sent_num'] < 10000]


# In[8]:


sentence = train_df[train_df['sent_num']==0]
token_list =  ' '.join([token for token in sentence[0]])
print(token_list)


# In[9]:


class FSBDataset():
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.idxtob = {'B-SENT': 1}
        
    def __len__(self):
        return self.data['sent_num'].max()
    
    def __getitem__(self, item):
        
        sentence = self.data[self.data['sent_num']==item]
        token_list =  [token for token in sentence[0]]
        target_list =  [target for target in sentence[3]]
        target_ids_list =  [1 if token=="B-SENT" else 0 for token in sentence[3]]
        
        encoded = self.tokenizer.encode_plus(' '.join(token_list),
                                            None,
                                            add_special_tokens=True,
                                            max_length=self.max_length,
                                            truncation=True,
                                            pad_to_max_length=True)
        
        ids = encoded['input_ids']
        mask = encoded['attention_mask']
        
        bpe_head_mask = [0]; upos_ids = [-1] # --> CLS token
        
        for word, target in zip(token_list, target_list):
            bpe_len = len(self.tokenizer.tokenize(word))
            head_mask = [1] + [0]*(bpe_len-1)
            bpe_head_mask.extend(head_mask)
            upos_mask = [self.idxtob.get(target,0)] + [-1]*(bpe_len-1)
            upos_ids.extend(upos_mask)
            #print("head_mask", head_mask)
        
        bpe_head_mask.append(0); upos_ids.append(-1) # --> END token
        bpe_head_mask.extend([0] * (self.max_length - len(bpe_head_mask))); upos_ids.extend([-1] * (self.max_length - len(upos_ids))) ## --> padding by max_len

        

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            #'target': torch.tensor(target_list, dtype=torch.long),
            'bpe_head_mask': torch.tensor(bpe_head_mask, dtype=torch.long),
            'target_ids': torch.tensor(upos_ids, dtype=torch.long)
        }
        
        
        
        


# In[10]:


class XLMRobertaBaseline(torch.nn.Module):
    def __init__(self):
        super(XLMRobertaBaseline, self).__init__()
        
        self.bert = transformers.XLMRobertaModel.from_pretrained('xlm-roberta-base')
        self.dropout = torch.nn.Dropout(0.33)
        self.classfier = torch.nn.Linear(768, 2)
        
        
    def forward(self, ids, mask):
        
        o1, o2 = self.bert(ids, mask)
        out = self.dropout(o1)
        logits = self.classfier(out)
        
        return logits
        
        
        


# In[11]:


MAX_LEN = 510
tokenizer = transformers.XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
train_dataset = FSBDataset(train_df, tokenizer, MAX_LEN)
train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=False, num_workers=4, batch_size=8)
test_dataset = FSBDataset(test_df, tokenizer, MAX_LEN)
test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, num_workers=4, batch_size=6)


# In[12]:


model = XLMRobertaBaseline()
model = torch.nn.DataParallel(model)
model = model.cuda()


# In[13]:


optimizer = transformers.AdamW(params=model.parameters(), lr=0.000005)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)


# In[14]:


def f1_score(total_pred, total_targ):
    
    p = 0 # (retrived SB and real SB) / retrived SB  # The percentage of (the number of correct predictions) / (the number of predction that system predicts as B-SENT)
    r = 0
    f1= 0
    
    #print("total_pred", total_pred, len(total_pred))
    #print("total_targ", total_targ, len(total_targ))
    
    np_total_pred = np.array(total_pred)
    np_total_tag = np.array(total_targ)
    
    #precision
    incidence_nopad = np.where(np_total_tag != -1) ## eliminate paddings
    #print("incidence_nopad", incidence_nopad)
    
    np_total_pred_nopad = np_total_pred[incidence_nopad]
    np_total_tag_nopad = np_total_tag[incidence_nopad]
    
    incidence_nopad_sb = np.where(np_total_pred_nopad == 1)
    np_total_pred_nopad_sb = np_total_pred_nopad[incidence_nopad_sb]
    np_total_tag_nopad_sb = np_total_tag_nopad[incidence_nopad_sb]
    
    count_active_tokens_p = len(np_total_pred_nopad_sb)
    count_correct_p = np.count_nonzero((np_total_pred_nopad_sb==np_total_tag_nopad_sb) == True)
    
    '''
    np_total_pred_incid = np_total_pred[incidence_p]
    print("np_total_pred_incid", np_total_pred_incid)
    ids_sb_pred_p = np.where(np_total_pred_incid==1)
    np_total_pred_p = np_total_pred_incid[ids_sb_pred_p]
    np_total_tag_p = np_total_tag[ids_sb_pred_p]
    
    print("ids_sb_pred_p", ids_sb_pred_p)
    print("np_total_pred_p", np_total_pred_p)
    print("np_total_tag_p", np_total_tag_p)
    
    count_active_tokens_p = len(np_total_pred_p)
    count_correct_p = np.count_nonzero((np_total_pred_p==np_total_tag_p) == True)
    '''
    
    print("count_correct_p", count_correct_p)
    print("count_active_tokens_p", count_active_tokens_p)
    
    
    try:
        p = count_correct_p/count_active_tokens_p
    except ZeroDivisionError:
        p = 0
    

    print("precision:", p)

    
    #recall
    ids_sb_pred_r = np.where(np_total_tag==1)
    np_total_pred_r = np_total_pred[ids_sb_pred_r]
    np_total_tag_r = np_total_tag[ids_sb_pred_r]
    
    #print("ids_sb_pred_r", ids_sb_pred_r)
    #print("np_total_pred_r", np_total_pred_r)
    #print("np_total_tag_r", np_total_tag_r)
    
    count_active_tokens_r = len(np_total_pred_r)
    count_correct_r = np.count_nonzero((np_total_pred_r==np_total_tag_r) == True)
    
    print("count_active_tokens_r", count_active_tokens_r)
    print("count_correct_r", count_correct_r)
    
    
    
    try:
        r = count_correct_r/count_active_tokens_r
    except ZeroDivisionError:
        r = 0
    
    print("recall:", r)
    
    
    #F1
    
    try:
        f1 = 2*(p*r) / (p+r)
    except ZeroDivisionError:
        f1 = 0
    

    print("F1:", f1)
    
    #count_active_tokens_recall = np.count_nonzero(np.array(total_targ) > -1)
    #print("count_active_tokens_recall", count_active_tokens_recall)
    #count_active_tokens_precision = np.count_nonzero(np.array(total_targ) > -1)
    
    #count_correct = np.count_nonzero((np.array(total_pred)==np.array(total_targ)) == True)
    #print("count_correct",count_correct)
    #print("ACCURACY:", count_correct/count_active_tokens)
    


# In[15]:


def train_loop_fn(train_loader, model, optimizer, DEVICE=None, scheduler=None):
    model.train()
    
    total_pred = []
    total_targ = []
    total_loss = []
    
    for idx, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
        optimizer.zero_grad()
        #print(batch['ids'], len(batch['ids']), batch['ids'].size() )
        #print(batch['mask'], len(batch['mask']))
        #print(batch['bpe_head_mask'], len(batch['bpe_head_mask']))
        #print(batch['upos_ids'], len(batch['upos_ids']))

        logists = model(batch['ids'].cuda(), batch['mask'].cuda())
        #print(logists, logists.size())
        #print(batch['upos_ids'], batch['upos_ids'].size())
        #print(logists.view(45,9), logists.view(45,9).size())
        #print(batch['upos_ids'].view(45), batch['upos_ids'].view(45).size())
        b,s,l = logists.size()
        loss = loss_fn(logists.view(b*s,l), batch['target_ids'].cuda().view(b*s))
        total_loss.append(loss.item())
        total_pred.extend(torch.argmax(logists.view(b*s,l), 1).cpu().tolist())
        total_targ.extend(batch['target_ids'].cuda().view(b*s).cpu().tolist())
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
            
        #print("batch",batch)
        #break
    #print(total_pred, len(total_pred))
    #print(total_targ, len(total_targ))
    count_active_tokens = np.count_nonzero(np.array(total_targ) > -1)
    count_correct = np.count_nonzero((np.array(total_pred)==np.array(total_targ)) == True)
    #print("TRAINING ACCURACY:", count_correct/count_active_tokens)
    f1_score(total_pred, total_targ)
    #f1_score(total_pred[2:], total_targ[2:])
    #print(count_active_tokens)
    #print(count_correct)

    
def dev_loop_fn(dev_loader, model, optimizer, DEVICE=None, scheduler=None):
    model.eval()
    
    total_pred = []
    total_targ = []
    total_loss = []
    total_middle_pred = []
    total_middle_targ = []

    with torch.no_grad():
        for idx, batch in tqdm(enumerate(dev_loader), total=len(dev_loader)):

            logists = model(batch['ids'].cuda(), batch['mask'].cuda())
            b,s,l = logists.size()
            #print(b,s,l)
            loss = loss_fn(logists.view(b*s,l), batch['target_ids'].cuda().view(b*s))
            total_loss.append(loss.item())
            total_pred.extend(torch.argmax(logists.view(b*s,l), 1).cpu().tolist())
            total_targ.extend(batch['target_ids'].cuda().view(b*s).cpu().tolist())
            

            logists2 = logists[:,2:,]
            b,s,l = logists2.size()
            #print(b,s,l)
            total_middle_pred.extend(torch.argmax(logists2.contiguous().view(b*s,l), 1).cpu().tolist())
            total_middle_targ.extend(batch['target_ids'][:,2:].cuda().contiguous().view(b*s).cpu().tolist())

    print("the number of total pred and targ", len(total_pred), len(total_targ))
    f1_score(total_pred, total_targ)
    f1_score(total_middle_pred, total_middle_targ)

    
    
    #count_active_tokens = np.count_nonzero(np.array(total_targ) > -1)
    #count_correct = np.count_nonzero((np.array(total_pred)==np.array(total_targ)) == True)
    #print("TESTING ACC:", count_correct/count_active_tokens)
    


# In[ ]:


for idx in range(100):
    train_loop_fn(train_loader, model, optimizer)
    dev_loop_fn(test_loader, model, optimizer)


# In[ ]:




