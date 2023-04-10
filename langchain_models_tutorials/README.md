<br>

# LANGCHAIN <u>MODELS</u> TUTORIALS FROM THE <a href="https://python.langchain.com/en/latest/getting_started/getting_started.html">DOCS</a>

This section of the tutorials deals with different types of models that are used in LangChain.

* **LLMs**
    * Large Language Models (LLMs) are the first type of models we cover. 
    * * These models take a text string as input, and return a text string as output.
* **Chat Models**
  * Chat Models are the second type of models we cover. 
  * These models are usually backed by a language model, but their APIs are more structured. 
  * Specifically, these models take a list of Chat Messages as input, and return a Chat Message.
* **Text Embedding Models**
  * The third type of models we cover are text embedding models. 
  * These models take text as input and return a list of floats.

---

<br>

## 1. LLMs: 

Large Language Models (LLMs) are a core component of LangChain. 
* **Remember:** LangChain is not a provider of LLMs, but rather provides a standard interface through which you can interact with a variety of LLMs.

The following sections of documentation are provided:
* **Getting Started:** 
  * An overview of all the functionality the LangChain LLM class provides.
* **How-To Guides:** 
  * A collection of how-to guides. 
  * These highlight how to accomplish various objectives with our LLM class (streaming, async, etc).
* **Integrations:** 
  * A collection of examples on how to integrate different LLM providers with LangChain (OpenAI, Hugging Face, etc).

---

<br>

### 1.1 Getting Started: 

---


<details>
<summary><b>Getting Started:</b> Using the LLM class in LangChain</summary>

This notebook goes over how to use the LLM class in LangChain.

The LLM class is a class designed for interfacing with LLMs. There are lots of LLM providers (OpenAI, Cohere, Hugging Face, etc) - this class is designed to provide a standard interface for all of them. In this part of the documentation, we will focus on generic LLM functionality. For details on working with a specific LLM wrapper, please see the examples in the How-To section.

For this notebook, we will work with an OpenAI LLM wrapper, although the functionalities highlighted are generic for all LLM types.

```python
from langchain.llms import OpenAI
llm = OpenAI(model_name="text-ada-001", n=2, best_of=2)
```

<b>Generate Text:</b> The most basic functionality an LLM has is just the ability to call it, passing in a string and getting back a string.

```python
llm("Tell me a joke")
'\n\nWhy did the chicken cross the road?\n\nTo get to the other side.'
```

<b>Generate:</b> More broadly, you can call it with a list of inputs, getting back a more complete response than just the text. This complete response includes things like multiple top responses, as well as LLM provider specific information

```python
llm_result = llm.generate(["Tell me a joke", "Tell me a poem"]*15)
len(llm_result.generations)
30
llm_result.generations[0]
[Generation(text='\n\nWhy did the chicken cross the road?\n\nTo get to the other side!'),
 Generation(text='\n\nWhy did the chicken cross the road?\n\nTo get to the other side.')]
llm_result.generations[-1]
[Generation(text="\n\nWhat if love neverspeech\n\nWhat if love never ended\n\nWhat if love was only a feeling\n\nI'll never know this love\n\nIt's not a feeling\n\nBut it's what we have for each other\n\nWe just know that love is something strong\n\nAnd we can't help but be happy\n\nWe just feel what love is for us\n\nAnd we love each other with all our heart\n\nWe just don't know how\n\nHow it will go\n\nBut we know that love is something strong\n\nAnd we'll always have each other\n\nIn our lives."),
 Generation(text='\n\nOnce upon a time\n\nThere was a love so pure and true\n\nIt lasted for centuries\n\nAnd never became stale or dry\n\nIt was moving and alive\n\nAnd the heart of the love-ick\n\nIs still beating strong and true.')]
```

You can also access provider specific information that is returned. This information is NOT standardized across providers.

```python
llm_result.llm_output
{'token_usage': {'completion_tokens': 3903,
  'total_tokens': 4023,
  'prompt_tokens': 120}}
```

<b>Number of Tokens:</b> You can also estimate how many tokens a piece of text will be in that model. This is useful because models have a context length (and cost more for more tokens), which means you need to be aware of how long the text you are passing in is.

Notice that by default the tokens are estimated using a HuggingFace tokenizer.

```python
llm.get_num_tokens("what a joke")
3
```

</details>

---

<br>

### 1.2 Generic Functionality: 

---

<details>
<summary><b>The Async API for LLMs:</b> How to Use It!</summary>

LangChain provides async support for LLMs by leveraging the 
<b><a href="https://docs.python.org/3/library/asyncio.html">asyncio</a></b> library. 
Async support is particularly useful for calling multiple LLMs concurrently, as these calls are network-bound. 

Currently, `OpenAI`, `PromptLayerOpenAI`, `ChatOpenAI`, and `Anthropic` are supported, 
but async support for other LLMs is on the roadmap.

You can use the **`agenerate`** method to call an OpenAI LLM asynchronously.

```python
import time
import asyncio

from langchain.llms import OpenAI

def generate_serially():
    llm = OpenAI(temperature=0.9)
    for _ in range(10):
        resp = llm.generate(["Hello, how are you?"])
        print(resp.generations[0][0].text)


async def async_generate(llm):
    resp = await llm.agenerate(["Hello, how are you?"])
    print(resp.generations[0][0].text)


async def generate_concurrently():
    llm = OpenAI(temperature=0.9)
    tasks = [async_generate(llm) for _ in range(10)]
    await asyncio.gather(*tasks)


s = time.perf_counter()
# If running this outside of Jupyter, use asyncio.run(generate_concurrently())
await generate_concurrently() 
elapsed = time.perf_counter() - s
print('\033[1m' + f"Concurrent executed in {elapsed:0.2f} seconds." + '\033[0m')

s = time.perf_counter()
generate_serially()
elapsed = time.perf_counter() - s
print('\033[1m' + f"Serial executed in {elapsed:0.2f} seconds." + '\033[0m')
```

Example output:

```terminal
I'm doing well, thank you. How about you?

I'm doing well, thank you. How about you?

I'm doing well, how about you?

I'm doing well, thank you. How about you?

I'm doing well, thank you. How about you?

I'm doing well, thank you. How about yourself?

I'm doing well, thank you! How about you?

I'm doing well, thank you. How about you?

I'm doing well, thank you! How about you?

I'm doing well, thank you. How about you?
Concurrent executed in 1.39 seconds.

I'm doing well, thank you. How about you?

I'm doing well, thank you. How about you?

I'm doing well, thank you. How about you?

I'm doing well, thank you. How about you?

I'm doing well, thank you. How about yourself?

I'm doing well, thanks for asking. How about you?

I'm doing well, thanks! How about you?

I'm doing well, thank you. How about you?

I'm doing well, thank you. How about yourself?

I'm doing well, thanks for asking. How about you?
Serial executed in 5.77 seconds.
```

</details>

---

<details>
<summary><b>Custom LLM Wrappers:</b> Use Your Own LLM!</summary>

You can create a custom LLM wrapper, in case you want to use your own LLM or a different wrapper 
than one that is supported in LangChain.

There is only one required thing that a custom LLM needs to implement:
* A **`_call`** method that takes in a string, some optional stop words, and returns a string

There is a second optional thing it can implement:
* An **`_identifying_params`** property that is used to help with printing of this class. 
* Should return a dictionary.

Let’s implement a very simple custom LLM that just returns the first N characters of the input.

```python
from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any
```

```python
class CustomLLM(LLM):
    
    n: int
        
    @property
    def \_llm\_type(self) \-> str:
        return "custom"
    
    def \_call(self, prompt: str, stop: Optional\[List\[str\]\] \= None) \-> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        return prompt\[:self.n\]
    
    @property
    def \_identifying\_params(self) \-> Mapping\[str, Any\]:
        """Get the identifying parameters."""
        return {"n": self.n}
```

We can now use this as an any other LLM.

```python
llm \= CustomLLM(n\=10)
```

```python
llm("This is a foobar thing")
```

```python
'This is a '
```

We can also print the LLM and see its custom print.

```python
print(llm)
```

CustomLLM

```python
Params: {'n': 10}
```

</details>

---

<details>
<summary><b>The <i>Fake</i> LLM:</b> How and Why You Should Use a Fake LLM</summary>

We expose a fake LLM class that can be used for testing. This allows you to mock out calls to the LLM and simulate 
what would happen if the LLM responded in a certain way.

We start this with using the FakeLLM in an agent.

```python
from langchain.llms.fake import FakeListLLM
```

```python
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
```

```python
tools = load_tools(["python_repl"])
```

```python
responses=[
    "Action: Python REPL\nAction Input: print(2 + 2)",
    "Final Answer: 4"
]
llm = FakeListLLM(responses=responses)
```

```python
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
```

```python
agent.run("whats 2 + 2")
```

```terminal
> Entering new AgentExecutor chain...
Action: Python REPL
Action Input: print(2 + 2)
Observation: 4

Thought:Final Answer: 4

> Finished chain.
```

```terminal
'4'
```

</details>

---

<details>
<summary><b>Caching:</b> How to cache LLM calls</summary>

This script will cover how to cache results of individual LLM calls.

**In Memory Cache**

This section demonstrates how to use an in-memory cache for LLM calls. 
* When using an in-memory cache, the results of LLM calls are stored in memory for quick retrieval.
* The first time an LLM call is made, the result will be fetched from the API and then stored in the cache. 
* Subsequent calls with the same prompt will return the cached result, significantly reducing the response time.

```python
import langchain
from langchain.llms import OpenAI
from langchain.cache import InMemoryCache
langchain.llm_cache = InMemoryCache()
```

```python
# To make the caching really obvious, lets use a slower model.
llm = OpenAI(model_name="text-davinci-002", n=2, best_of=2)
```

```python
%%time
# The first time, it is not yet in cache, so it should take longer
llm("Tell me a joke")
```

```terminal
CPU times: user 30.7 ms, sys: 18.6 ms, total: 49.3 ms
Wall time: 791 ms
```

```python
"\n\nWhy couldn't the bicycle stand up by itself? Because it was...two tired!"
```

```python
%%time
# The second time it is, so it goes faster
llm("Tell me a joke")
```

```terminal
CPU times: user 80 µs, sys: 0 ns, total: 80 µs
Wall time: 83.9 µs
```

```python
"\n\nWhy couldn't the bicycle stand up by itself? Because it was...two tired!"
```

**SQLite Cache**

This section demonstrates how to use a SQLite cache for LLM calls. 
* SQLite caching stores the results of LLM calls in an SQLite database file. 
* This allows for persistent caching, even if the program is restarted. 
* The first time an LLM call is made, the result will be fetched from the API and then stored in the cache. 
* Subsequent calls with the same prompt will return the cached result, reducing response time.

```python
!rm .langchain.db
```

```python
# We can do the same thing with a SQLite cache
from langchain.cache import SQLiteCache
langchain.llm_cache = SQLiteCache(database_path=".langchain.db")
```

```python
%%time
# The first time, it is not yet in cache, so it should take longer
llm("Tell me a joke")
```

```terminal
CPU times: user 17 ms, sys: 9.76 ms, total: 26.7 ms
Wall time: 825 ms
```

```python
'\n\nWhy did the chicken cross the road?\n\nTo get to the other side.'
```

```python
%%time
# The second time it is, so it goes faster
llm("Tell me a joke")
```

```terminal
CPU times: user 2.46 ms, sys: 1.23 ms, total: 3.7 ms
Wall time: 2.67 ms
```
    
```python
'\n\nWhy did the chicken cross the road?\n\nTo get to the other side.'
```

**Redis Cache**

This section demonstrates how to use a Redis cache for LLM calls. 
* Redis caching stores the results of LLM calls in a Redis data store. 
* This allows for distributed caching, making it useful for applications running on multiple servers. 
* The first time an LLM call is made, the result will be fetched from the API and then stored in the cache. 
* Subsequent calls with the same prompt will return the cached result, reducing response time. 
* *Note that a local Redis instance must be running to use this cache.*

```python
# We can do the same thing with a Redis cache
# (make sure your local Redis instance is running first before running this example)
from redis import Redis
from langchain.cache import RedisCache
langchain.llm_cache = RedisCache(redis_=Redis())
```

```python
%%time
# The first time, it is not yet in cache, so it should take longer
llm("Tell me a joke")
```

```python
%%time
# The second time it is, so it goes faster
llm("Tell me a joke")
```

**SQLAlchemy Cache**

This section shows how to use an SQLAlchemy Cache to cache LLM calls in any SQL database supported by SQLAlchemy. 
* This enables you to use a variety of SQL databases, including PostgreSQL, MySQL, and SQLite, for caching purposes. 
* To use this cache, you must create an appropriate database connection using SQLAlchemy's **`create_engine`** function.

```python
# You can use SQLAlchemyCache to cache with any SQL database supported by SQLAlchemy.

# from langchain.cache import SQLAlchemyCache
# from sqlalchemy import create_engine

# engine = create_engine("postgresql://postgres:postgres@localhost:5432/postgres")
# langchain.llm_cache = SQLAlchemyCache(engine)
```

**Custom SQLAlchemy Schemas**

This section demonstrates how to create a custom SQLAlchemy schema for caching LLM calls. 
* By defining your own declarative **`SQLAlchemyCache`** child class, you can customize the schema used for caching. 

This example shows how to create a full-text indexed LLM cache using PostgreSQL.

```python
# You can define your own declarative SQLAlchemyCache child class to customize the schema used for caching. For example, to support high-speed fulltext prompt indexing with Postgres, use:

from sqlalchemy import Column, Integer, String, Computed, Index, Sequence
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy_utils import TSVectorType
from langchain.cache import SQLAlchemyCache

Base = declarative_base()


class FulltextLLMCache(Base):  # type: ignore
    """Postgres table for fulltext-indexed LLM Cache"""

    __tablename__ = "llm_cache_fulltext"
    id = Column(Integer, Sequence('cache_id'), primary_key=True)
    prompt = Column(String, nullable=False)
    llm = Column(String, nullable=False)
    idx = Column(Integer)
    response = Column(String)
    prompt_tsv = Column(TSVectorType(), Computed("to_tsvector('english', llm || ' ' || prompt)", persisted=True))
    __table_args__ = (
        Index("idx_fulltext_prompt_tsv", prompt_tsv, postgresql_using="gin"),
    )

engine = create_engine("postgresql://postgres:postgres@localhost:5432/postgres")
langchain.llm_cache = SQLAlchemyCache(engine, FulltextLLMCache)
```

**Optional Caching**

This section demonstrates how to disable caching for specific LLMs. 
* You may want to disable caching for certain LLMs in cases where you expect the results to change frequently or when you want to save memory or storage space. 
* In this example, caching is disabled for a specific LLM, and you can see that the response time is consistent between the first and second calls.
  * NOTE: In the example below, even though global caching is enabled, we turn it off for a specific LLM

```python
llm = OpenAI(model_name="text-davinci-002", n=2, best_of=2, cache=False)
```

```python
%%time
llm("Tell me a joke")
```

```terminal
CPU times: user 5.8 ms, sys: 2.71 ms, total: 8.51 ms
Wall time: 745 ms
```

```python
'\n\nWhy did the chicken cross the road?\n\nTo get to the other side!'
```

```python
%%time
llm("Tell me a joke")
```

```terminal
CPU times: user 4.91 ms, sys: 2.64 ms, total: 7.55 ms
Wall time: 623 ms
```

```python
'\n\nTwo guys stole a calendar. They got six months each.'
```

**Optional Caching in Chains**

This section demonstrates how to disable caching for particular nodes in chains. 
* You can control caching behavior at different stages of a chain, allowing you to optimize caching for specific parts of your pipeline. In this example, caching is enabled for the map-step of a map-reduce chain but disabled for the reduce step, demonstrating how caching can be fine-tuned within a single chain.

You can also turn off caching for particular nodes in chains. 
* Because of certain interfaces, its often easier to construct the chain first, and then edit the LLM afterwards.

As an example, we will load a summarizer map-reduce chain. 
* We will cache results for the map-step, but then not freeze it for the combine step.

```python
llm = OpenAI(model_name="text-davinci-002")
no_cache_llm = OpenAI(model_name="text-davinci-002", cache=False)
```

```python
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain

text_splitter = CharacterTextSplitter()
```

```python
with open('../../../state_of_the_union.txt') as f:
    state_of_the_union = f.read()
texts = text_splitter.split_text(state_of_the_union)
```

```python
from langchain.docstore.document import Document
docs = [Document(page_content=t) for t in texts[:3]]
from langchain.chains.summarize import load_summarize_chain
```

```python
chain = load_summarize_chain(llm, chain_type="map_reduce", reduce_llm=no_cache_llm)
```

```python
%%time
chain.run(docs)
```

```terminal
CPU times: user 452 ms, sys: 60.3 ms, total: 512 ms
Wall time: 5.09 s
```

```python
'\n\nPresident Biden is discussing the American Rescue Plan and the Bipartisan Infrastructure Law, which will create jobs and help Americans. He also talks about his vision for America, which includes investing in education and infrastructure. In response to Russian aggression in Ukraine, the United States is joining with European allies to impose sanctions and isolate Russia. American forces are being mobilized to protect NATO countries in the event that Putin decides to keep moving west. The Ukrainians are bravely fighting back, but the next few weeks will be hard for them. Putin will pay a high price for his actions in the long run. Americans should not be alarmed, as the United States is taking action to protect its interests and allies.'
```

When we run it again, we see that it runs substantially faster but the final answer is different. 
This is due to caching at the map steps, but not at the reduce step.

```python
%%time
chain.run(docs)
```

```terminal
CPU times: user 11.5 ms, sys: 4.33 ms, total: 15.8 ms
Wall time: 1.04 s
```

```python
'\n\nPresident Biden is discussing the American Rescue Plan and the Bipartisan Infrastructure Law, which will create jobs and help Americans. He also talks about his vision for America, which includes investing in education and infrastructure.'
```

</details>

---

<details>
<summary><b>TBD</b>TBD</summary>

TBD

</details>

---