#!/usr/bin/env python
# coding: utf-8

# In[16]:


import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.nn import functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
block_size = 8
batch_size = 4
learning_rate = 7e-4
max_iters = 50000
eval_iters = 1000


# In[17]:


with open('books.txt', 'r', encoding = 'utf') as f:
    text = f.read()
chars = sorted(set(text))
vocab_size = len(chars)


# In[18]:


string_to_int = { ch:i for i, ch in enumerate(chars)}
int_to_string = { i:ch for i, ch in enumerate(chars)}
encode = lambda s: [string_to_int[c] for c in s]
decode = lambda v: ''.join([int_to_string[c] for c in v])



# In[19]:


data = torch.tensor(encode(text), dtype = torch.long)


# In[20]:


n = int(0.8*len(data))
train_data = data[ :n]
valid_data = data[n: ]

def get_batch(split):
    data = train_data if split == "train" else valid_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+1+block_size] for i in ix])
    x, y = x.to(device), y.to(device)
    return x,y

x, y = get_batch("train")



# In[21]:


def evaluate_loss():
    out = {}
    model.eval()
    for i in ["train", "test"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(i)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[i] = losses.mean()
    model.train()
    return out



# In[22]:


class BiGramLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_emb_tb = nn.Embedding(vocab_size, vocab_size)

    def forward(self, index, targets = None):
        logits = self.token_emb_tb(index)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, index, max_new_tokens):

        for _ in range(max_new_tokens):

            logits, loss = self.forward(index)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim = -1)
            index_next = torch.multinomial(probs, num_samples = 1)
            index = torch.cat((index, index_next), dim = 1)

        return index

model = BiGramLM(vocab_size)
m = model.to(device)


# In[23]:


"""x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print("when input is ", context, "print" , target)"""


# In[24]:


optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)
step = []
train_loss = []
test_loss = []
for iter in range(max_iters):

    if iter % eval_iters == 0 :
        print(f"step: {iter}, {evaluate_loss()["train"]:.4f}, {evaluate_loss()["test"]:.4f}")
        step.append(iter)
        train_loss.append(evaluate_loss()["train"])
        test_loss.append(evaluate_loss()["test"])

    xb, yb = get_batch("train")

    logits, loss = model.forward(xb, yb)
    optimizer.zero_grad(set_to_none = True)
    loss.backward()
    optimizer.step()

print(loss.item())
plt.figure(figsize=(8, 5))
plt.plot(step, train_loss, label='Training Loss', color='blue')
plt.plot(step, test_loss, label='Test Loss', color='orange')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('LLM Loss Curve')
plt.legend()
plt.grid(True)
plt.show()


# In[25]:


context = torch.zeros((1,1), dtype = torch.long, device = device)
gen_chars = decode(m.generate(context, max_new_tokens = 1000)[0].tolist())
print(gen_chars)


# In[ ]:




