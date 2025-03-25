import pandas as pd
import streamlit as st
from serpapi import Client
import google.generativeai as genai
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Hardcoded API keys for demonstration purposes
SERPAPI_API_KEY = "6d2fbf993e072a1adf9b076450fff9b7f820fac3fecaf882434a9ddf9ee2f2ee"
GEMINI_API_KEY = "AIzaSyB9mjHevsOWZ1I3LCYIQjNbq9nrk5WzWpk"

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# Function to perform web search using SerpApi
def web_search(query, api_key):
    if not query:
        raise ValueError("Query cannot be empty")
    
    client = Client(api_key=api_key)
    params = {
        "q": query,
        "engine": "google",  # Specify the search engine
    }
    search = client.search(params)
    return search  # Return the search results directly

# Function to extract information using Gemini API
def extract_information(search_results, entity, query_template):
    prompt = query_template.format(entity=entity, search_results=search_results)
    response = model.generate_content(prompt)
    
    # Check for response content
    if response and hasattr(response, 'text'):
        return response.text.strip()
    else:
        return "No information found"

# Streamlit App
def main():
    st.title("AI Agent Dashboard")
    st.write("Upload a CSV file, select a column for queries, and retrieve structured data.")

    # File Upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Data Preview:")
        st.write(data.head())

        # Select Column for Queries
        search_column = st.selectbox("Select the Column for Queries", data.columns)
        query_template = st.text_input("Define your query", "Extract the email address of {entity} from the following web results: {search_results}")

        if st.button("Submit"):
            queries = data[search_column]
            results = []
            for entity in queries:
                search_query = query_template.replace("{entity}", entity)
                if not search_query:
                    st.error(f"Query for entity '{entity}' is empty. Skipping this entity.")
                    continue
                
                try:
                    search_results = web_search(search_query, SERPAPI_API_KEY)
                    parsed_result = extract_information(search_results, entity, query_template)
                    if parsed_result:
                        results.append({"Entity": entity, "Extracted Information": parsed_result})
                except Exception as e:
                    st.error(f"Error processing entity '{entity}': {e}")

            # Display Results
            st.write("Search Results:")
            for result in results:
                st.write(f"{result['Entity']}: {result['Extracted Information']}")

            # Download CSV
            if results:
                df = pd.DataFrame(results)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download results as CSV",
                    data=csv.encode('utf-8'),
                    file_name='results.csv',
                    mime='text/csv',
                )
            else:
                st.write("No results to download.")

if __name__ == '__main__':
    main()