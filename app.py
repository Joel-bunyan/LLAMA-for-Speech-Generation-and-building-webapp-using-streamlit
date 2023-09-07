import os 
import streamlit as st 
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain 
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper 
from langchain.llms import Replicate

from langchain import PromptTemplate, HuggingFaceHub, LLMChain
import os
template = """Question: {question}

Answer: """
prompt = PromptTemplate(template=template, input_variables=["question"])

os.environ["REPLICATE_API_TOKEN"] = 'r8_GQaPZbEMScSQTzqOSd5Yk5bVdJEQMVh2yfRHh'
llm = Replicate(model="replicate/dolly-v2-12b:ef0e1aefc61f8e096ebe4db6b2bacc297daf2ef6899f0f7e001ec445893500e5")



st.title('Speech Generator')
prompt = st.text_input('Plug in your prompt here') 


title_template = PromptTemplate(
    input_variables = ['topic'], 
    template='write me a title of a speech in the field of {topic}'
)

script_template = PromptTemplate(
    input_variables = ['title'], 
    template='write me a complete speech based on this title TITLE: {title}'
)


title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')



title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory=script_memory)

wiki = WikipediaAPIWrapper()

if prompt: 
    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt) 
    script = script_chain.run(title=title, wikipedia_research=wiki_research)

    st.write(title) 
    st.write(script) 

    with st.expander('Title'): 
        st.info(title_memory.buffer)

    with st.expander('Script'): 
        st.info(script_memory.buffer)

    with st.expander('Wikipedia Research'): 
        st.info(wiki_research)
