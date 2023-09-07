
import streamlit as st 
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
import os

st.title('Speech Generator')
prompt = st.text_input('Plug in your prompt here') 


import replicate
os.environ["REPLICATE_API_TOKEN"] = "r8_GQaPZbEMScSQTzqOSd5Yk5bVdJEQMVh2yfRHh"

output = replicate.run('a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5', # LLM model
                        input={"prompt": f"{prompt} Assistant: ", 
                        "temperature":0.1, "top_p":0.9,  "repetition_penalty":1})
o = ""
for i in output:
    o += i
st.write(o) 