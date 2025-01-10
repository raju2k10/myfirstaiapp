from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import LLMChain
from langchain import PromptTemplate

import streamlit as st
import os

# Set up the environment variable for the Google API key
os.environ['GOOGLE_API_KEY'] = st.secrets['GOOGLE_API_KEY']

# Create a prompt template for generating stock recommendations
stock_template = "Give me the top {number} stocks to look for today with a brief reason for each recommendation."

stock_prompt = PromptTemplate(template=stock_template, input_variables=['number'])

# Initialize Google's Gemini model
gemini_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")

# Create LLM chain using the prompt template and model
stock_chain = stock_prompt | gemini_model

# Streamlit UI for the application
st.header("Stock Recommendation Generator")

st.subheader("Get the top stocks to watch for today using Generative AI")

number = st.number_input("Number of stocks", min_value=1, max_value=10, value=5, step=1)

if st.button("Generate"):
    stocks = stock_chain.invoke({"number": number})
    st.write(stocks.content)
