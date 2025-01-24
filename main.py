from fastapi import FastAPI, HTTPException
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import PromptTemplate
import os

# Initialize FastAPI app
app = FastAPI()

# Set up API key for Google's Generative AI
os.environ['GOOGLE_API_KEY'] = "AIzaSyB_bnprNdELA3lsOfkVkVZ-SC3ToqA-QDM"

# Create a prompt template for generating recommendations
recommendation_template = "Give me the top 5 recommendations to visit in {country}"

recommendation_prompt = PromptTemplate(template=recommendation_template, input_variables=['country'])

# Initialize Google's Gemini model
gemini_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Travel Recommendation API!"}

@app.post("/recommendations/")
async def get_recommendations(countries: list[str]):
    if not countries:
        raise HTTPException(status_code=400, detail="Please provide at least one country.")
    
    recommendation_chain = recommendation_prompt | gemini_model
    recommendations_response = {}

    try:
        for country in countries:
            recommendations = recommendation_chain.invoke({"country": country})
            recommendations_response[country] = recommendations.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

    return {"recommendations": recommendations_response}
