import streamlit as st
from sentence_transformers import SentenceTransformer, util
from geopy.distance import geodesic
import pandas as pd
import re

# Load the SentenceTransformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Sample data (replace this with your actual DataFrame loading)
agents_df = pd.read_excel("C:\Data Science\Find_top_3_agents\Agents.xlsx")

# Function to parse POINT string to (latitude, longitude) tuple
def parse_location(point_str):
    match = re.search(r'POINT\(([-\d\.]+) ([-\d\.]+)\)', point_str)
    if match:
        return float(match.group(2)), float(match.group(1))  # return as (latitude, longitude)
    return None

# Function to find the top 3 relevant agents for a query
def find_top_agents(query_text, location_str, type_of_agent, max_distance_km=50):
    query_embedding = model.encode(query_text, convert_to_tensor=True)
    relevant_agents = []

    for _, agent in agents_df.iterrows():
        if agent['Type of Agent'] == type_of_agent:
            agent_embedding = model.encode(agent['Description of Agent'], convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(query_embedding, agent_embedding).item()
            relevant_agents.append((agent, similarity))
    
    # Sort agents by similarity score
    relevant_agents.sort(key=lambda x: x[1], reverse=True)
    
    # If the query requires an Onsite agent, apply the location filter
    if type_of_agent == 'Onsite':
        query_location = parse_location(location_str)
        relevant_agents = [
            (agent, similarity) for agent, similarity in relevant_agents
            if geodesic(query_location, parse_location(agent['Location of Deployment'])).km <= max_distance_km
        ]
    
    # Return top 3 agents by similarity
    return [agent['Name of Agent'] for agent, _ in relevant_agents[:3]]

# Streamlit UI
st.title("Agent Matching App")

# Input fields
query_text = st.text_area("Query Text", "Looking for help with organizing a large corporate event.")
location_str = st.text_input("Location of Requirement (in POINT format)", "POINT(69.223216 73.998725)")
type_of_agent = st.selectbox("Type of Agent", ["Onsite", "Online"])

# Button to trigger the search
if st.button("Find Top Agents"):
    top_agents = find_top_agents(query_text, location_str, type_of_agent)
    st.write("Top 3 Agents:")
    for agent in top_agents:
        st.write(agent)
