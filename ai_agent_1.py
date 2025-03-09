import streamlit as st
from google import genai
import os

# Initialize the API client
client = genai.Client(api_key=os.environ['GOOGLE_GEMINI_API_KEY'])

# Streamlit App UI
st.title("SWOT Analysis Generator")
st.write("""The SWOT Analysis Generator is a tool that automates the creation of detailed SWOT analyses for any organization, helping businesses make informed strategic decisions. By simply providing the organization’s name and context, the tool generates an insightful evaluation of strengths, weaknesses, opportunities, and threats. 
         (Powered by - Gemini API) \n

Advantages:
- Time-saving: Automates the SWOT analysis process.
- Data-driven: Provides objective insights based on business context.
- Customizable: Tailored analysis based on organization-specific details.
- Scalable: Can be used for multiple organizations or projects.\n
Uses:
* Strategic planning
* Market research
* Competitor analysis
* Risk management\n
This tool provides analysis of the situation and aids businesses to make strategic decisions faster and with confidence. \n""")

# User input
organization_name = st.text_input("Enter the organization name:")
context = st.text_area("Please provide the current situation of the organization")

if organization_name and context:
    # Create the content prompt for Gemini API based on user input
    content = f"""Provide a detailed SWOT analysis for {organization_name}. 
    The situation of the orgnization is as follows - {context}. 
    Include top 4 of each of strengths, weaknesses, opportunities, and threats in the analysis. 
    Give 3 short summary of real examples of similar situations context which can be referred. Give a formal tone to the response."""

    # Call Gemini API to generate the response
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash", contents=content
        )

        # Display the response
        st.subheader(f"SWOT Analysis for {organization_name}:")
        st.write(response.text)
    except Exception as e:
        st.error(f"Error: {str(e)}")
