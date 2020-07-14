#!/usr/bin/env python
# coding: utf-8

# # Sentence Boundary detection with NER features

# In[ ]:


import torch
import pandas as pd
import numpy as np
import transformers
from tqdm import tqdm
import csv


# In[ ]:


train_file = './data/train.tsv'
test_file = './data/test.tsv'


# In[ ]:


train_df = pd.read_csv(train_file, delimiter='\t', engine='python', encoding='UTF-8', error_bad_lines=False, header=None, quoting=csv.QUOTE_NONE)
train_df.head()
#train_df = pd.read_excel(train_file, delimiter='\t', encoding='UTF-8', error_bad_lines=False, header=None)
#train_df.head()


# In[ ]:


test_df = pd.read_csv(test_file, delimiter='\t', engine='python', encoding='UTF-8', error_bad_lines=False, header=None, quoting=csv.QUOTE_NONE)
test_df.head()


# In[ ]:


def set_sentence_num(df): 

    sent_num = 0
    df['sent_num'] = sent_num
    for idx in range(len(df)):
        df['sent_num'][idx] = sent_num
        if df[0][idx]=='.' and df[1][idx]=="SENT":
            sent_num +=1
    df.head()
    print(sent_num)
    
    return df


# In[ ]:


train_df = set_sentence_num(train_df)
test_df = set_sentence_num(test_df)


# In[ ]:


train_df = train_df[train_df['sent_num'] < 10000]
test_df = test_df[test_df['sent_num'] < 100000]


# In[ ]:


sentence = train_df[train_df['sent_num']==2]
token_list =  ' '.join([token for token in sentence[0]])
print(token_list)


# In[ ]:


class FSBDataset():
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.idxtob = {'B-SENT': 1}
        
    def __len__(self):
        return self.data['sent_num'].max()
    
    def __getitem__(self, item):
        
        sentence = train_df[train_df['sent_num']==item]
        token_list =  [token for token in sentence[0]]
        target_list =  [target for target in sentence[3]]
        target_ids_list =  [1 if token=="B-SENT" else 0 for token in sentence[3]]
        
        encoded = self.tokenizer.encode_plus(' '.join(token_list),
                                            None,
                                            add_special_tokens=True,
                                            max_length=self.max_length,
                                            pad_to_max_length=True)
        
        ids = encoded['input_ids']
        mask = encoded['attention_mask']
        
        bpe_head_mask = [0]; upos_ids = [-1] # --> CLS token
        
        for word, target in zip(token_list, target_list):
            bpe_len = len(self.tokenizer.tokenize(word, add_special_token=False))
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
        
        
        
        


# In[ ]:


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
        
        
        


# In[ ]:


MAX_LEN = 640
tokenizer = transformers.XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
train_dataset = FSBDataset(train_df, tokenizer, MAX_LEN)
train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, num_workers=4, batch_size=10)
test_dataset = FSBDataset(test_df, tokenizer, MAX_LEN)
test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, num_workers=4, batch_size=4)


# In[ ]:


model = XLMRobertaBaseline()
model = torch.nn.DataParallel(model)
model = model.cuda()


# In[ ]:


optimizer = transformers.AdamW(params=model.parameters(), lr=0.000005)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)


# In[ ]:


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
    
    p = count_correct_p/count_active_tokens_p
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
    
    r = count_correct_r/count_active_tokens_r
    print("recall:", r)
    
    
    #F1
    f1 = 2*(p*r) / (p+r)
    print("F1:", f1)
    
    #count_active_tokens_recall = np.count_nonzero(np.array(total_targ) > -1)
    #print("count_active_tokens_recall", count_active_tokens_recall)
    #count_active_tokens_precision = np.count_nonzero(np.array(total_targ) > -1)
    
    #count_correct = np.count_nonzero((np.array(total_pred)==np.array(total_targ)) == True)
    #print("count_correct",count_correct)
    #print("ACCURACY:", count_correct/count_active_tokens)
    


# In[ ]:


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


'''
for batch in train_loader:
    
    model.train()
    
    print(batch)
    logits = model(batch['ids'], batch['mask'])
    print(logits)
    
    break
    
    '''


# In[ ]:


tmp = train_df.groupby('sent_num').max()
print(tmp)


# In[ ]:


tt = train_df[train_df['sent_num'] == 12025]


# In[ ]:


tt


# In[ ]:


tt.head(60)


# In[ ]:


tt.tail(60)


# In[ ]:




