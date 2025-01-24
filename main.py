from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import PromptTemplate
import streamlit as st
import os

# Set up API key for Google's Generative AI
os.environ['GOOGLE_API_KEY'] = st.secrets['GOOGLE_API_KEY']

# Create a prompt template for generating recommendations
recommendation_template = "Give me the top 5 recommendations to visit in {country}"

recommendation_prompt = PromptTemplate(template=recommendation_template, input_variables=['country'])

# Initialize Google's Gemini model
gemini_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")

# Create the LLM chain for generating recommendations
recommendation_chain = recommendation_prompt | gemini_model

# Streamlit app setup
st.header("Top 5 Travel Tourist Recommendations")

st.subheader("Discover the must-visit places in multiple countries")

# Allow the user to select multiple countries
countries = st.multiselect(
    "Select countries to generate recommendations:",
    options=["Canada", "Cuba", "France", "Italy", "Japan", "Australia", "USA"],
    default=["Canada", "Cuba"]  # Pre-select Canada and Cuba
)

if st.button("Generate Recommendations"):
    if countries:
        for country in countries:
            st.subheader(f"Top 5 recommendations for {country}:")
            recommendations = recommendation_chain.invoke({"country": country})
            st.write(recommendations.content)
    else:
        st.warning("Please select at least one country.")
