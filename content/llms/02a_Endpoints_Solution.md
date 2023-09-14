```python
from pathlib import Path

import openai
```


```python
openaikey = Path('aetoken.txt').read_text()
openai.api_key = openaikey
openai.api_base = 'https://ray-summit-training-jrvwy.cld-kvedzwag2qa8i5bj.s.anyscaleuserdata-staging.com/v1'
model="meta-llama/Llama-2-70b-chat-hf"
```


```python
# Define the system message
system_msg = 'You are a helpful assistant.'

# Define the user message -- i.e., the prompt
user_msg = 'What is your favorite place to visit in San Francisco?'

# call GPT
response = openai.ChatCompletion.create(model=model, max_tokens=200,
                                        messages=[{"role": "system", "content": system_msg},
                                         {"role": "user", "content": user_msg}])

response.choices[0].message["content"]
```


```python
def quick_chat(user, temp=1.0):
    response = openai.ChatCompletion.create(model=model, temperature=temp, max_tokens=150,
                                        messages=[{"role": "system", "content": 'You are a helpful assistant.'},
                                         {"role": "user", "content": user}])
    return response.choices[0].message["content"]
```


```python
quick_chat(user_msg)
```


```python
quick_chat(user_msg, temp=0.1)
```


```python
quick_chat(user_msg, temp=1.5)
```


```python
quick_chat("Who is the CFO of Monkeylanguage LLC?")
```


```python
base_prompt = """
You are a helpful assistant who can answer questions about the team at Monkeylanguage LLC, an AI startup.

When answering questions, use the following facts about Monkeylanguage LLC employees:

1. Juan Williams is the CEO
2. Linda Johnson is the CFO
3. Robert Jordan is the CTO
4. Aileen Xin is Engineering Lead

If you don't have information to answer a question, please say you don't know. Don't make up an answer

"""
```


```python
def chat(system, user):
    response = openai.ChatCompletion.create(model=model, max_tokens=200,
                                        messages=[{"role": "system", "content": system},
                                         {"role": "user", "content": user}])
    return response.choices[0].message["content"]
```


```python
chat(base_prompt, "Who is the CFO of Monkeylanguage LLC?")
```


```python
chat(base_prompt, "Who are all of the technical staff members at Monkeylanguage LLC?")
```


```python
database = {
    'Monkeylanguage LLC' : ['Juan Williams is the CEO', 'Linda Johnson is the CFO', 'Robert Jordan is the CTO', 'Aileen Xin is Engineering Lead'],
    'FurryRobot Corp' : ['Ana Gonzalez is the CEO', 'Corwin Hall is the CFO', 'FurryRobot employs no technical staff', 'All tech is produced by AI'],
    'LangMagic Inc' : ["Steve Jobs' ghost fulfills all roles in the company"]
}
```


```python
prompt = 'Who is the CFO at Monkeylanguage LLC?'
```


```python
def lookup(prompt, database):
    for k in database.keys():
        if k in prompt:
            return database[k]
```


```python
docs = lookup(prompt, database)

docs
```


```python
def make_base_prompt(docs):
    return """
You are a helpful assistant who can answer questions about the team at some AI startup companies.

When answering questions, use the following facts about employees at the firm:
""" + '\n'.join([doc for doc in docs]) + """
If you don't have information to answer a question, please say you don't know. Don't make up an answer"""
```


```python
make_base_prompt(docs)
```


```python
def retrieve_and_chat(prompt, database):
    docs = lookup(prompt, database)
    base_prompt = make_base_prompt(docs)
    return chat(base_prompt, prompt)
```


```python
retrieve_and_chat(prompt, database)
```


```python
retrieve_and_chat('Who is the CFO at FurryRobot Corp?', database)
```


```python
retrieve_and_chat('Who is the CFO at LangMagic Inc?', database)
```


```python
tools = ['If you wish to email, return the function call EMAIL(email_subject, email_body), inserting the relevant email_subject and email_body.']
```


```python
def make_enhanced_base_prompt(docs, tools):
    return """
You are a helpful assistant who can answer questions about the team at some AI startup companies. 

When answering questions, use the following facts about employees at the firm:
""" + '\n'.join([doc for doc in docs]) + """
If you don't have information to answer a question, please say you don't know. Don't make up an answer.

You can also use tools to accomplish some actions.
""" + '\n'.join([tool for tool in tools]) + """

If you use a tool, return the tool function call and nothing else.
"""
```


```python
make_enhanced_base_prompt(docs, tools)
```


```python
chat(make_enhanced_base_prompt(docs, tools),
     'Please send an email advertising a new role as assistant to the CFO of Monkeylanguage LLC. Name the CFO, and send the email from the CEO')
```

Note that the model may not quite get the sense of "assistant to the CFO" ... if it doesn't get that right, we can be more specific:


```python
chat(make_enhanced_base_prompt(docs, tools),
     'Please send an email advertising a new role as assistant in the office of the CFO of Monkeylanguage LLC. Name the CFO, and send the email from the CEO')
```

This should have worked, but it's not returning a clean, usable API invocation. We can address that as well:


```python
chat(make_enhanced_base_prompt(docs, tools),
     'Please send an email advertising a new role as assistant in the office of the CFO of Monkeylanguage LLC. Name the CFO, and send the email from the CEO. Please return the command only, and no other text.')
```

Next we have to make some changes to add an email destination. The model can't possibly work if we don't add emails into our employee database.


```python
database = {
    'Monkeylanguage LLC' : ['Juan Williams (juanw@monkeylang.com) is the CEO', 'Linda Johnson (lindaj@monkeylang.com) is the CFO', 
                            'Robert Jordan (rjordan@monkeylang.com) is the CTO', 'Aileen Xin (axin@monkeylang.com) is Engineering Lead'],
    'FurryRobot Corp' : ['Ana Gonzalez is the CEO', 'Corwin Hall is the CFO', 'FurryRobot employs no technical staff', 'All tech is produced by AI'],
    'LangMagic Inc' : ["Steve Jobs' ghost fulfills all roles in the company"]
}
```


```python
retrieve_and_chat("What is the email address for the CTO of Monkeylanguage LLC?", database)
```


```python
tools = ['If you wish to email, return the function call EMAIL(receipient_address, email_subject, email_body), inserting the relevant '
         +' recipient_address, email_subject and email_body.']
```


```python
chat(make_enhanced_base_prompt(docs, tools),
     'Please send an email to Robert Jordan with a message advertising a new role as assistant in the office of the CFO of Monkeylanguage LLC. Name the CFO, and send the email from the CEO. Please return the command only, and no other text.')
```


```python
tools = ['If you wish to email, return the function call EMAIL(receipient_address, email_subject, email_body), inserting the relevant '
         +' recipient_address (in quotes), email_subject and email_body.']
```


```python
chat(make_enhanced_base_prompt(docs, tools),
     'Please send an email to Robert Jordan with a message advertising a new role as assistant in the office of the CFO of Monkeylanguage LLC. Name the CFO, and send the email from the CEO. Please return the command only, and no other text.')
```


```python

```
