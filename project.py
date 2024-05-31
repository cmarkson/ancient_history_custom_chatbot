#!/usr/bin/env python
# coding: utf-8

# # Ancient History Chatbot Project
#I created an ancient history chatbot based off of information contained in the wiki page of ancient history.  this page had specific formatting that needed to be adjusted but provided relavent information for the custom chatbot.

# In[112]:




# ## Data Wrangling
# 
# loaded ancient history data to be able to test answering specific questions about history.

# In[2]:


import requests

from openai import OpenAI
    

OPENAI_API_KEY=''  #removed for security


# In[31]:


params = {
    "action": "query", 
    "prop": "extracts",
    "exlimit": 1,
    "titles": "Timeline_of_ancient_history",
    "explaintext": 1,
    "formatversion": 2,
    "format": "json"
}


# In[32]:


resp = requests.get("https://en.wikipedia.org/w/api.php", params=params)
response_dict = resp.json()

response_dict["query"]["pages"][0]["extract"].split("\n")


# In[33]:


import pandas as pd


# In[34]:


df = pd.DataFrame(
    data = response_dict["query"]["pages"][0]["extract"].split("\n"),
    columns = ['text']
)

df


# In[35]:


df = df[df.text.str.len()>0] # remove null rows
df


# In[36]:


df = df[~df.text.str.startswith('==')] # remove == row strings
df


# # remove any text that doesn't begin with c., late or /d

# In[51]:


df = df[(df.text.str.startswith('Late') #find rows that start with Late
           | df.text.str.startswith('c.') #find rows that start with c.
           | df.text.str.match('^\d'))] # find rows that start with digit
df


# In[56]:


df.reset_index(drop=True).to_csv('ancient.csv', index=False)


# In[57]:


df = pd.read_csv('ancient.csv')
df


# # create embeddings on ancient history data

# In[115]:



# In[59]:

client = OpenAI(api_key=OPENAI_API_KEY)
response = client.embeddings.create(
    model='text-embedding-3-small', #using the new embedding models
    input=df.text.tolist()
)


# In[60]:

print(response.data[0].embedding)



# In[62]:


len(response.data[0].embedding)


# In[63]:


embeddings = list(map(lambda x: x.embedding, response.data))
type(embeddings)


# In[65]:


len(embeddings)  #217 historical records


# In[64]:


df['embeddings'] = embeddings
df.to_csv('embeddings.csv', index=False)


# In[67]:


import numpy as np
import pandas as pd
df = pd.read_csv('embeddings.csv') # set index_col=0 if loaded with Unnamed:0
df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)
df


# ## Custom Query Completion
# 
# TODO: In the cells below, compose a custom query using your chosen dataset and retrieve results from an OpenAI `Completion` model. You may copy and paste any useful code from the course materials.

# In[75]:


question = 'What year and where did vandals sack Rome?'


# In[76]:

def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], 
model=model).data[0].embedding

question_embeddings = get_embedding(question, model='text-embedding-3-small')
question_embeddings


# # find cosine similarities from ancient history dataset

# In[77]:
from typing import List, Optional
def distances_from_embeddings(
    query_embedding: List[float],
    embeddings: List[List[float]],
    distance_metric="cosine",
) -> List[List]:
    """Return the distances between a query embedding and a list of embeddings."""
    from scipy import spatial
    from typing import List, Optional

    distance_metrics = {
        "cosine": spatial.distance.cosine,
        "L1": spatial.distance.cityblock,
        "L2": spatial.distance.euclidean,
        "Linf": spatial.distance.chebyshev,
    }
    distances = [
        distance_metrics[distance_metric](query_embedding, embedding)
        for embedding in embeddings
    ]
    return distances

# In[78]:
distances = distances_from_embeddings(question_embeddings, df['embeddings'].tolist(), distance_metric='cosine')
distances

df['distances'] = distances
df


# In[79]:


df.to_csv('distances.csv', index=False)


# In[2]:


import pandas as pd
pd.read_csv('distances.csv')
sorted_distances = df.sort_values(by='distances', ascending=True)
sorted_distances


# In[81]:


sorted_distances.to_csv('distances_sorted.csv', index=False)


# In[83]:


import tiktoken
tokenizer = tiktoken.get_encoding('cl100k_base')


# In[87]:


tokenized = tokenizer.encode(question)
tokenized, len(tokenized)
max_token_count = 500


# In[85]:


prompt_template = """
Answer the question based on the context below, and if the question
can't be answered based on the context, say "I don't know"

Context: 

{}

---

Question: {}
Answer:"""


# In[3]:


tokenized_question = tokenizer.encode(question)
tokenized_prompt = tokenizer.encode(prompt_template)
current_token_count = len(tokenized_question) + len(tokenized_prompt)
current_token_count


# In[4]:


import pandas as pd

df = pd.read_csv('distances_sorted.csv')
df

context = []
current_token_count = len(tokenized_question) + len(tokenized_prompt)

for text in df.text.values:
    text_token_count = len(tokenizer.encode(text))
    try:
       current_token_count += text_token_count
    except:
        print(text_token_count)

    if current_token_count <= max_token_count:
        context.append(text)
    else:
        break


# In[93]:


print(prompt_template.format('\n\n###\n\n'.join(context), question))


# In[6]:


import openai
client = OpenAI(api_key=OPENAI_API_KEY)



# In[ ]:


response = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[{"role": "user", "content": prompt_template.format('\n\n###\n\n'.join(context), question)}]
)
print(response.choices[0].message.content)

# ## Custom Performance Demonstration
# 

# ### Question 1

# In[ ]:

question = 'what year was the oldest song documented?'
question_embeddings = get_embedding(question, model='text-embedding-3-small')
df = pd.read_csv('embeddings.csv') # set index_col=0 if loaded with Unnamed:0
df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)


distances = distances_from_embeddings(question_embeddings, df['embeddings'].tolist(), distance_metric='cosine')
df['distances'] = distances

sorted_distances = df.sort_values(by='distances', ascending=True)


tokenized_question = tokenizer.encode(question)
tokenized_prompt = tokenizer.encode(prompt_template)
current_token_count = len(tokenized_question) + len(tokenized_prompt)
current_token_count

context = []
current_token_count = len(tokenized_question) + len(tokenized_prompt)
df=sorted_distances
for text in df.text.values:
    text_token_count = len(tokenizer.encode(text))
    try:
       current_token_count += text_token_count
    except:
        print(text_token_count)

    if current_token_count <= max_token_count:
        context.append(text)
    else:
        break


print(prompt_template.format('\n\n###\n\n'.join(context), question))

response = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[{"role": "user", "content": prompt_template.format('\n\n###\n\n'.join(context), question)}]
)
print(response.choices[0].message.content)

# In[ ]:





# ### Question 2

# In[ ]:

question = 'when was the Phoenician alphabet created'
question_embeddings = get_embedding(question, model='text-embedding-3-small')
df = pd.read_csv('embeddings.csv') # set index_col=0 if loaded with Unnamed:0
df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)


distances = distances_from_embeddings(question_embeddings, df['embeddings'].tolist(), distance_metric='cosine')

df['distances'] = distances
sorted_distances = df.sort_values(by='distances', ascending=True)


tokenized_question = tokenizer.encode(question)
tokenized_prompt = tokenizer.encode(prompt_template)
current_token_count = len(tokenized_question) + len(tokenized_prompt)
current_token_count

context = []
current_token_count = len(tokenized_question) + len(tokenized_prompt)
df=sorted_distances
for text in df.text.values:
    text_token_count = len(tokenizer.encode(text))
    try:
       current_token_count += text_token_count
    except:
        print(text_token_count)

    if current_token_count <= max_token_count:
        context.append(text)
    else:
        break


print(prompt_template.format('\n\n###\n\n'.join(context), question))

response = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[{"role": "user", "content": prompt_template.format('\n\n###\n\n'.join(context), question)}]
)
print(response.choices[0].message.content)



# In[ ]:




