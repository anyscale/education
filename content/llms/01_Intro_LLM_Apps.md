```python
from pathlib import Path

import openai
```

# LLM Applications
## Birds-eye view: end-to-end from first principles

We'll demo a micro illustration of a common LLM app pattern, using basic Python along with access to OpenAI's GPT-3.5-Turbo model (accessed via API).

The purpose is to understand the key ideas underlying more complex tools like vector databases, Langchain/LlamaIndex, structure data extraction and function calling, etc.

We won't cover creating and training new large language models -- we'll assume that we already have one.

<div class="alert alert-block alert-info">
    
__Roadmap: end-to-end overview__

1. Prompts and the OpenAI API
1. Limitations in LLM question answering
1. Providing additional information to the model via the prompt
1. The role of data stores and search in finding information for the prompt
1. Zero-shot tool use: prompting the model to use an API
</div>

### Getting started


```python
openaikey = Path('openaikey.txt').read_text()
openai.api_key = openaikey
model="gpt-3.5-turbo"
```

Most apps are based around two kinds of prompts: 
* a "system" prompt (basically the rules of the game for the AI)
* a "user" prompt (what the user or application submits ... in chat conversations, the conversation history is in the user prompt)

There are various tricks and techniques for eliciting various behaviors from different models ... but the basics are straightforward.


```python
# Define the system message
system_msg = 'You are a helpful assistant.'

# Define the user message -- i.e., the prompt
user_msg = 'What is your favorite place to visit in San Francisco?'
```

Now we can ask the LLM to respond. OpenAI's `ChatCompletion` API simplifies and implements the pattern.


```python
# call GPT
response = openai.ChatCompletion.create(model=model,
                                        messages=[{"role": "system", "content": system_msg},
                                         {"role": "user", "content": user_msg}])

response.choices[0].message["content"]
```

Since we'll be interacting a lot, we can wrap this logic in a helper function. We'll hide most of the params for now, but expose an optional "temperature" which specifies how creative (or chaotic) we would like the model to be.


```python
def quick_chat(user, temp=1.0):
    response = openai.ChatCompletion.create(model=model, temperature=temp, 
                                        messages=[{"role": "system", "content": 'You are a helpful assistant.'},
                                         {"role": "user", "content": user}])
    return response.choices[0].message["content"]
```


```python
quick_chat(user_msg)
```

A low temperature may produce more spare, conservative responses with less likelihood of hallucination


```python
quick_chat(user_msg, temp=0.1)
```

A higher temperature produces more creative responses ... but there may not be a huge difference


```python
quick_chat(user_msg, temp=1.8)
```

### Asking the AI harder questions by injecting facts into the prompt

Many common facts are heavily covered in the LLM training data, so the model can easily return them.

But what happens if we ask an unusual or impossible question?


```python
quick_chat("Who is the CFO of Monkeylanguage LLC?")
```

Well-tuned LLMs should decline to provide an answer ... although less-well-tuned ones may simply make up ("hallucinate") an answer.

A common category of LLM apps attempts to use the LLM as a sort of natural language user interface to query specific information. Where the information is not likely in the training data, and we don't want hallucinated answers, there is a simple trick:

Jam the relevant facts into the prompt.

Let's try that by adding in some organization info for a fictional company, Monkeylanguage LLC, into our chatbot prompt.


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

Since we're modifying the base prompt now, we'll need to update our quick chat shortcut function to allow us to pass the new system prompt along with a user prompt


```python
def chat(system, user):
    response = openai.ChatCompletion.create(model=model,
                                        messages=[{"role": "system", "content": system},
                                         {"role": "user", "content": user}])
    return response.choices[0].message["content"]
```

Now we can ask about our fictional company


```python
chat(base_prompt, "Who is the CFO of Monkeylanguage LLC?")
```


```python
chat(base_prompt, "Who are all of the technical staff members at Monkeylanguage LLC?")
```

### Flexible injection of facts via query from a supporting dataset

But how do we get the right content to insert into the prompt?

We use a trick:

1. look at the user prompt -- the actual question we want to answer
2. search a dataset of some kind for information that might match -- it doesn't need to be precise or exact, which is why this technique is more general than an old-fashioned database search for the final answer
3. insert matches or possible matches into the prompt

In production apps, we usually use a database that supports semantic matching to natural language texts via embedding vector similarity -- "vector databases"

But we can demonstrate this with a toy database


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

We'll define a trivial `lookup` helper that returns all of the facts for the first company whose name (the dict key) is in the query


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

We can code a helper to build the system prompt from a set of relevant documents


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

And now we can "chat" with our "data"


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

Some queries are "harder" ... and the model may not get it right on the first try without either more data or more sophisticated prompting.

But in this example, the model usually gets the right answer in one or two tries


```python
retrieve_and_chat('Who is the CFO at LangMagic Inc?', database)
```

The process we've just implemented -- albeit with more data, a more sophisticated approach to storing and querying, and more complex prompts -- is at the heart of many LLM-powered apps. 

It's a pattern called "Retrieval Augmented Generation" or RAG

### Tools: "but what about those AI assistants that can do things for me, like order my groceries?"

In order to interface the LLM to the "real world" we can ask the LLM to generate a function call or API call based on our interaction.

We can then use that API or function call to trigger a real-world result, like a grocery order.

__How does this work?__

The essence of teaching a LLM to use functions is just more prompt engineering. 

1. define a collection of "tools" -- functions or data patterns that the LLM can use in various contexts
2. provide all of tools, along with a description of when they might be useful, in the prompt
3. ask the LLM to do something and hope that it properly selects and uses the tool

We can either provide the LLM with all of the available tools, or we can retrieve relevant tools from a larger collection based on the user prompt. We can even have the LLM itself choose the tools via patterns like RAG that we saw earlier


```python
tools = ['If you wish to email, return the function call EMAIL(email_subject, email_body), inserting the relevant email_subject and email_body.']
```

We'll inject the tool description(s) into the base prompt


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

And now we can ask the AI to do something ... and hopefully it will produce the right invocation


```python
chat(make_enhanced_base_prompt(docs, tools),
     'Please send an email advertising a new role as assistant to the CFO of Monkeylanguage LLC. Name the CFO, and send the email from the CEO')
```

In a nutshell, that is many (maybe most) of the AI-powered apps that are being built today.

Packages like LlamaIndex, LangChain, and others help automating sophisticated patterns of prompt generation and content/tool merging.

And semantic vector databases (along with proper "chunking" and ingestion of relevant datasets) provide knowledge to the LLM beyond what it learned in training.
