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


def set_newline_to_EOS(file_r, file_w, eos="\n"):
    with open(file_w, 'w', encoding="UTF-8") as fw:
        with open(file_r, 'r', encoding="UTF-8") as fr:
            while True:
                line = fr.readline()
                if not line: break
                if line == eos:
                    fw.write("<EOS>\t<EOS>\t<EOS>\t<EOS>\n")
                else:
                    fw.write(line)
                    
def set_sentence_num(df): 

    sent_num = 0
    df['sent_num'] = sent_num
    for idx in range(len(df)):
        df['sent_num'][idx] = sent_num
        #if df[0][idx]=='.' and df[1][idx]=="SENT":
        if df[0][idx]=='<EOS>' and df[1][idx]=="<EOS>":
            sent_num +=1
    df.head()
    df = df[df[0] != "<EOS>"] #get rid of the "<EOS>" toekns
    print(sent_num)
    
    return df


train_file = './data/train.tsv'
test_file = './data/test.tsv'
#test_file = './data_v1/europarl-sbd-eval.tsv'
train_EOS_file = './data/train_EOS.tsv'
test_EOS_file = './data/test_EOS.tsv'
#test_EOS_file = './data_v1/europarl-sbd-eval_EOS.tsv'

set_newline_to_EOS(train_file, train_EOS_file, eos="\n")
set_newline_to_EOS(test_file, test_EOS_file, eos="\n")
#set_newline_to_EOS(test_file, test_EOS_file, eos="\t\n")
train_file = train_EOS_file
test_file = test_EOS_file


# In[4]:


train_df = pd.read_csv(train_file, delimiter='\t', engine='python', encoding='UTF-8', error_bad_lines=False, header=None, quoting=csv.QUOTE_NONE)
train_df.head()
test_df = pd.read_csv(test_file, delimiter='\t', engine='python', encoding='UTF-8', error_bad_lines=False, header=None, quoting=csv.QUOTE_NONE)
test_df.head()


# In[5]:


train_df = set_sentence_num(train_df)
test_df = set_sentence_num(test_df)


# In[6]:


train_df = train_df[train_df['sent_num'] < 1000000]
test_df = test_df[test_df['sent_num'] < 10000]


# In[7]:


sentence = train_df[train_df['sent_num']==0]
token_list =  ' '.join([token for token in sentence[0]])
print(token_list)


# In[8]:


class FSBDataset():
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.idxtob = {'B-SENT': 1}
        self.idxtoPOS = {'DET:ART': 0,'NAM': 1,'KON': 2,'PUN': 3,'DET:POS': 4,'NOM': 5,'VER:pres': 6,'PRP': 7,'PRO:PER': 8,'VER:infi': 9,'PRP:det': 10,'VER:simp': 11,'VER:pper': 12,'NUM': 13,'SENT': 14,'ABR': 15,'VER:futu': 16,'PRO:DEM': 17,'ADJ': 18,
 'PRO:REL': 19,'PRO:IND': 20,'ADV': 21,'SYM': 22,'PUN:cit': 23,'VER:impf': 24,'VER:subp': 25,'VER:subi': 26,'VER:ppre': 27,'VER:cond': 28,'PRO:POS': 29,'VER:impe': 30}
        self.idxtoNER = {'I-LOC': 0, 'I-PER': 1, 'O': 2, 'I-ORG': 3}
        
    def __len__(self):
        return self.data['sent_num'].max()
    
    def __getitem__(self, item):
        
        sentence = self.data[self.data['sent_num']==item]
        token_list =  [token for token in sentence[0]]
        sbd_list =  [target for target in sentence[3]]
        pos_list =  [target for target in sentence[1]]
        ner_list =  [target for target in sentence[2]]
        
        #target_ids_list =  [1 if token=="B-SENT" else 0 for token in sentence[3]]
        #pos_ids = [self.idxtoPOS.get(pos) for pos in sentence[1]]
        #ner_ids = [self.idxtoNER.get(ner) for ner in sentence[2]]
        
        encoded = self.tokenizer.encode_plus(' '.join(token_list),
                                            None,
                                            add_special_tokens=True,
                                            max_length=self.max_length,
                                            truncation=True,
                                            padding='max_length')
        
        ids = encoded['input_ids']
        mask = encoded['attention_mask']
        
        bpe_head_mask = [0]; sbd_ids = [-1]; pos_ids = [-1]; ner_ids = [-1] # --> CLS token
        
        for word, sbd, pos, ner in zip(token_list, sbd_list, pos_list, ner_list):
            bpe_len = len(self.tokenizer.tokenize(word))
            head_mask = [1] + [0]*(bpe_len-1)
            bpe_head_mask.extend(head_mask)
            
            sbd_mask = [self.idxtob.get(sbd,0)] + [-1]*(bpe_len-1)
            sbd_ids.extend(sbd_mask)
            pos_mask = [self.idxtoPOS.get(pos,0)] + [-1]*(bpe_len-1)
            pos_ids.extend(pos_mask)
            ner_mask = [self.idxtoNER.get(ner,0)] + [-1]*(bpe_len-1)
            ner_ids.extend(ner_mask)
            #print("head_mask", head_mask)
        
        bpe_head_mask.append(0)
        bpe_head_mask.extend([0] * (self.max_length - len(bpe_head_mask)))
        
        sbd_ids.append(-1) # --> END token
        sbd_ids.extend([-1] * (self.max_length - len(sbd_ids))) ## --> padding by max_len
        pos_ids.append(-1) # --> END token
        pos_ids.extend([-1] * (self.max_length - len(pos_ids))) ## --> padding by max_len
        ner_ids.append(-1) # --> END token
        ner_ids.extend([-1] * (self.max_length - len(ner_ids))) ## --> padding by max_len


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'bpe_head_mask': torch.tensor(bpe_head_mask, dtype=torch.long),
            'sbd_ids': torch.tensor(sbd_ids, dtype=torch.long),
            'pos_ids': torch.tensor(pos_ids, dtype=torch.long),
            'ner_ids': torch.tensor(ner_ids, dtype=torch.long)
        }
        
        
        
        


# In[9]:


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
        
        
    
class POS(torch.nn.Module):
    def __init__(self, num_pos=31):
        super(POS, self).__init__()
        
        self.dropout = torch.nn.Dropout(0.33)
        self.classfier = torch.nn.Linear(768, num_pos)
        
    def forward(self, bert_out):
        
        o1 = bert_out
        out = self.dropout(o1)
        logits = self.classfier(out)
        
        return logits
    
    
class NER(torch.nn.Module):
    def __init__(self, num_ner=4):
        super(NER, self).__init__()
        
        self.dropout = torch.nn.Dropout(0.33)
        self.classfier = torch.nn.Linear(768, num_ner)
        
    def forward(self, bert_out):
        
        o1 = bert_out
        out = self.dropout(o1)
        logits = self.classfier(out)
        
        return logits
    
    
class SBD(torch.nn.Module):
    def __init__(self, input_dim=768, num_sbd=2):
        super(SBD, self).__init__()
        
        self.dropout = torch.nn.Dropout(0.33)
        self.classfier = torch.nn.Linear(input_dim, num_sbd)
        
    def forward(self, bert_out):
        
        o1 = bert_out
        out = self.dropout(o1)
        logits = self.classfier(out)
        
        return logits
    
    
class XLMRobertaMultiTask(torch.nn.Module):
    def __init__(self):
        super(XLMRobertaMultiTask, self).__init__()
        
        num_pos = 31
        num_ner = 4
        num_sbd = 2
        pos_dim = 50
        ner_dim = 25
        
        self.bert = transformers.XLMRobertaModel.from_pretrained('xlm-roberta-base')
        
        self.pos = POS(num_pos)
        self.ner = NER(num_ner)
        self.sbd = SBD(num_sbd)
        
        self.pos_emb = torch.nn.Embedding(num_pos, pos_dim)
        self.ner_emb = torch.nn.Embedding(num_ner, ner_dim)
        
        
    def forward(self, ids, mask):
        
        o1, o2 = self.bert(ids, mask)
        out = o1 
        
        pos_logits = self.pos(out)
        ner_logits = self.ner(out)
        sbd_logits = self.sbd(out)

        
        return sbd_logits, pos_logits, ner_logits
    
    
    
class XLMRobertaMultiTaskPred(torch.nn.Module):
    def __init__(self):
        super(XLMRobertaMultiTaskPred, self).__init__()
        
        num_pos = 31
        num_ner = 4
        num_sbd = 2
        pos_dim = 50
        ner_dim = 25
        
        self.bert = transformers.XLMRobertaModel.from_pretrained('xlm-roberta-base')
        
        self.pos = POS(num_pos)
        self.ner = NER(num_ner)
        self.sbd = SBD(768+pos_dim+ner_dim, num_sbd)
        
        self.pos_emb = torch.nn.Embedding(num_pos, pos_dim)
        self.ner_emb = torch.nn.Embedding(num_ner, ner_dim)
        
        
    def forward(self, ids, mask):
        
        o1, o2 = self.bert(ids, mask)
        out = o1 #self.dropout(o1)
        
        #Step1: predict POS tags for the entire toekns
        pos_logits = self.pos(out)
        ner_logits = self.ner(out)
        
        pos_idx = torch.argmax(pos_logits, dim=2)
        ner_idx = torch.argmax(ner_logits, dim=2)
        
        pos_emb = self.pos_emb(pos_idx)
        ner_emb = self.ner_emb(ner_idx)

        concatenated = torch.cat([out, pos_emb, ner_emb], dim=2)
        sbd_logits = self.sbd(concatenated)
        
        return sbd_logits, pos_logits, ner_logits


# In[10]:


MAX_LEN = 510
tokenizer = transformers.XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
train_dataset = FSBDataset(train_df, tokenizer, MAX_LEN)
train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, num_workers=4, batch_size=8)
test_dataset = FSBDataset(test_df, tokenizer, MAX_LEN)
test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, num_workers=4, batch_size=6)


# In[11]:


#model = XLMRobertaBaseline()
#model = XLMRobertaMultiTask()
model = XLMRobertaMultiTaskPred()
model = torch.nn.DataParallel(model)
model = model.cuda()
optimizer = transformers.AdamW(params=model.parameters(), lr=0.000005)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)


# In[12]:


def f1_score(total_pred, total_targ):
    
    p = 0 # (retrived SB and real SB) / retrived SB  # The percentage of (the number of correct predictions) / (the number of predction that system predicts as B-SENT)
    r = 0
    f1= 0

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
    
    count_active_tokens_r = len(np_total_pred_r)
    count_correct_r = np.count_nonzero((np_total_pred_r==np_total_tag_r) == True)
    
    print("count_active_tokens_r", count_active_tokens_r)
    print("count_correct_r", count_correct_r)
    
    
    
    try:
        r = count_correct_r/count_active_tokens_r
    except ZeroDivisionError:
        r = 0
    
    print("recall:", r)
    
    
    try:
        f1 = 2*(p*r) / (p+r)
    except ZeroDivisionError:
        f1 = 0
    

    print("F1:", f1)
    
    
    
def ner_f1_score(total_pred, total_targ):
    
    p = 0 # (retrived SB and real SB) / retrived SB  # The percentage of (the number of correct predictions) / (the number of predction that system predicts as B-SENT)
    r = 0
    f1= 0

    np_total_pred = np.array(total_pred)
    np_total_tag = np.array(total_targ)
    
    #precision
    incidence_nopad = np.where(np_total_tag != -1) ## eliminate paddings
    #print("incidence_nopad", incidence_nopad)
    
    np_total_pred_nopad = np_total_pred[incidence_nopad]
    np_total_tag_nopad = np_total_tag[incidence_nopad]
    
    incidence_nopad_sb = np.where((np_total_pred_nopad != -1) & (np_total_pred_nopad != 2)) #np.where(np_total_pred_nopad == 1) 
    np_total_pred_nopad_sb = np_total_pred_nopad[incidence_nopad_sb]
    np_total_tag_nopad_sb = np_total_tag_nopad[incidence_nopad_sb]
    
    count_active_tokens_p = len(np_total_pred_nopad_sb)
    count_correct_p = np.count_nonzero((np_total_pred_nopad_sb==np_total_tag_nopad_sb) == True)
    
    print("count_correct_p", count_correct_p)
    print("count_active_tokens_p", count_active_tokens_p)
    
    
    try:
        p = count_correct_p/count_active_tokens_p
    except ZeroDivisionError:
        p = 0
    

    print("precision:", p)

    
    #recall
    ids_sb_pred_r = np.where((np_total_tag != -1) & (np_total_tag != 2)) #np.where(np_total_tag==1)
    np_total_pred_r = np_total_pred[ids_sb_pred_r]
    np_total_tag_r = np_total_tag[ids_sb_pred_r]
    
    count_active_tokens_r = len(np_total_pred_r)
    count_correct_r = np.count_nonzero((np_total_pred_r==np_total_tag_r) == True)
    
    print("count_active_tokens_r", count_active_tokens_r)
    print("count_correct_r", count_correct_r)
    
    
    
    try:
        r = count_correct_r/count_active_tokens_r
    except ZeroDivisionError:
        r = 0
    
    print("recall:", r)
    
    
    try:
        f1 = 2*(p*r) / (p+r)
    except ZeroDivisionError:
        f1 = 0
    

    print("F1:", f1)
    


# In[13]:


def train_loop_fn(train_loader, model, optimizer, DEVICE=None, scheduler=None, mode="all", s_r=1, p_r=1, n_r=1):
    model.train()
    
    sbd_pred = []
    sbd_targ = []
    sbd_loss = []
    
    pos_pred = []
    pos_targ = []
    pos_loss = []
    
    ner_pred = []
    ner_targ = []
    ner_loss = []
    
    for idx, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
        optimizer.zero_grad()

        sbd_logits, pos_logits, ner_logits = model(batch['ids'].cuda(), batch['mask'].cuda())
        
        #if mode = "all" or "sbd":
        b,s,l = sbd_logits.size()
        s_loss = loss_fn(sbd_logits.view(b*s,l), batch['sbd_ids'].cuda().view(b*s))
        sbd_loss.append(s_loss.item())
        sbd_pred.extend(torch.argmax(sbd_logits.view(b*s,l), 1).cpu().tolist())
        sbd_targ.extend(batch['sbd_ids'].cuda().view(b*s).cpu().tolist())

        b,s,l = pos_logits.size()
        p_loss = loss_fn(pos_logits.view(b*s,l), batch['pos_ids'].cuda().view(b*s))
        pos_loss.append(p_loss.item())
        pos_pred.extend(torch.argmax(pos_logits.view(b*s,l), 1).cpu().tolist())
        pos_targ.extend(batch['pos_ids'].cuda().view(b*s).cpu().tolist())
        
        b,s,l = ner_logits.size()
        n_loss = loss_fn(ner_logits.view(b*s,l), batch['ner_ids'].cuda().view(b*s))
        ner_loss.append(n_loss.item())
        ner_pred.extend(torch.argmax(ner_logits.view(b*s,l), 1).cpu().tolist())
        ner_targ.extend(batch['ner_ids'].cuda().view(b*s).cpu().tolist())
        
        loss = s_r*s_loss + n_r*n_loss + p_r*p_loss
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
            
    sbd_count_active_tokens = np.count_nonzero(np.array(sbd_targ) > -1)
    sbd_count_correct = np.count_nonzero((np.array(sbd_pred)==np.array(sbd_targ)) == True)
    pos_count_active_tokens = np.count_nonzero(np.array(pos_targ) > -1)
    pos_count_correct = np.count_nonzero((np.array(pos_pred)==np.array(pos_targ)) == True)
    ner_count_active_tokens = np.count_nonzero(np.array(ner_targ) > -1)
    ner_count_correct = np.count_nonzero((np.array(ner_pred)==np.array(ner_targ)) == True)
    
    f1_score(sbd_pred, sbd_targ)
    print("POS Accuracy:"+ str(pos_count_correct/pos_count_active_tokens))
    print("NER Accuracy:"+ str(ner_count_correct/ner_count_active_tokens))
    ner_f1_score(ner_pred, ner_targ)
    
    #f1_score(total_pred[2:], total_targ[2:])

    
def dev_loop_fn(dev_loader, model, optimizer, DEVICE=None, scheduler=None):
    model.eval()
    
    sbd_pred = []
    sbd_targ = []
    sbd_loss = []
    sbd_middle_pred = []
    sbd_middle_targ = []
    
    pos_pred = []
    pos_targ = []
    pos_loss = []
    
    ner_pred = []
    ner_targ = []
    ner_loss = []


    with torch.no_grad():
        for idx, batch in tqdm(enumerate(dev_loader), total=len(dev_loader)):

            sbd_logits, pos_logits, ner_logits = model(batch['ids'].cuda(), batch['mask'].cuda())

            #if mode = "all" or "sbd":
            b,s,l = sbd_logits.size()
            s_loss = loss_fn(sbd_logits.view(b*s,l), batch['sbd_ids'].cuda().view(b*s))
            sbd_loss.append(s_loss.item())
            sbd_pred.extend(torch.argmax(sbd_logits.view(b*s,l), 1).cpu().tolist())
            sbd_targ.extend(batch['sbd_ids'].cuda().view(b*s).cpu().tolist())

            b,s,l = pos_logits.size()
            p_loss = loss_fn(pos_logits.view(b*s,l), batch['pos_ids'].cuda().view(b*s))
            pos_loss.append(p_loss.item())
            pos_pred.extend(torch.argmax(pos_logits.view(b*s,l), 1).cpu().tolist())
            pos_targ.extend(batch['pos_ids'].cuda().view(b*s).cpu().tolist())

            b,s,l = ner_logits.size()
            n_loss = loss_fn(ner_logits.view(b*s,l), batch['ner_ids'].cuda().view(b*s))
            ner_loss.append(n_loss.item())
            ner_pred.extend(torch.argmax(ner_logits.view(b*s,l), 1).cpu().tolist())
            ner_targ.extend(batch['ner_ids'].cuda().view(b*s).cpu().tolist())
            

            sbd_logits_middle = sbd_logits[:,2:,]
            b,s,l = sbd_logits_middle.size()
            #print(b,s,l)
            sbd_middle_pred.extend(torch.argmax(sbd_logits_middle.contiguous().view(b*s,l), 1).cpu().tolist())
            sbd_middle_targ.extend(batch['sbd_ids'][:,2:].cuda().contiguous().view(b*s).cpu().tolist())
            

    sbd_count_active_tokens = np.count_nonzero(np.array(sbd_targ) > -1)
    sbd_count_correct = np.count_nonzero((np.array(sbd_pred)==np.array(sbd_targ)) == True)
    pos_count_active_tokens = np.count_nonzero(np.array(pos_targ) > -1)
    pos_count_correct = np.count_nonzero((np.array(pos_pred)==np.array(pos_targ)) == True)
    ner_count_active_tokens = np.count_nonzero(np.array(ner_targ) > -1)
    ner_count_correct = np.count_nonzero((np.array(ner_pred)==np.array(ner_targ)) == True)

    f1_score(sbd_pred, sbd_targ)
    f1_score(sbd_middle_pred, sbd_middle_targ)
    print("POS Accuracy:"+ str(pos_count_correct/pos_count_active_tokens))
    print("NER Accuracy:"+ str(ner_count_correct/ner_count_active_tokens))

    '''
    print("sbd_targ: ",sbd_targ)
    print("sbd_pred: ",sbd_pred)

    print("pos_targ: ",pos_targ)
    print("pos_pred: ",pos_pred)

    print("ner_targ: ",ner_targ)
    print("ner_pred: ",ner_pred)
    '''
    ner_f1_score(ner_pred, ner_targ)


    
    
    #count_active_tokens = np.count_nonzero(np.array(total_targ) > -1)
    #count_correct = np.count_nonzero((np.array(total_pred)==np.array(total_targ)) == True)
    #print("TESTING ACC:", count_correct/count_active_tokens)
    


# In[ ]:


for idx in range(100):
    train_loop_fn(train_loader, model, optimizer, s_r=1, p_r=1, n_r=1)
    dev_loop_fn(test_loader, model, optimizer)

'''
for idx in range(20):
    train_loop_fn(train_loader, model, optimizer, a=0,b=0,c=1)
    dev_loop_fn(test_loader, model, optimizer)
print("Finished NER training")
    
for idx in range(20):
    train_loop_fn(train_loader, model, optimizer, a=0,b=1,c=0)
    dev_loop_fn(test_loader, model, optimizer)
print("Finished POS training")
    
for idx in range(20):
    train_loop_fn(train_loader, model, optimizer, a=1,b=0,c=0)
    dev_loop_fn(test_loader, model, optimizer)
print("Finished SBD training")
'''


# In[ ]:


dev_loop_fn(test_loader, model, optimizer)


# In[ ]:





# In[ ]:


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


# In[ ]:


checkpoint = {
    "state_dict": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "train_dataset": train_dataset
}
save_checkpoint(checkpoint)


# In[ ]:




