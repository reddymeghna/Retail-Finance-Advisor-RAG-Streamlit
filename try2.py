import pandas as pd
import json
import streamlit as st
import mysql.connector
import os
import requests
from langchain_groq import ChatGroq
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from PIL import Image
import cv2
from streamlit_lottie import st_lottie


def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)


def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        st.error(f"Failed to load URL: {url} with status code: {r.status_code}")
        return None
    try:
        return r.json()
    except json.JSONDecodeError as e:
        st.error(f"JSON decode error: {e}")
        st.error(f"Response content is not JSON: {r.text}")
        return None


lottie_hello = load_lottieurl(
    "https://lottie.host/0edcc7b1-fa80-4949-b56a-89aced683664/eKwMsyXHk2.json"
)

if lottie_hello:
    st_lottie(lottie_hello, key="hello")


# Configure database connection
def get_db_connection():
    return mysql.connector.connect(
        host="localhost", user="root", password="", database="streamlit_db"
    )


# Function to load the uploaded file with proper encoding handling
def load_file(uploaded_file):
    if uploaded_file.name.endswith(".csv"):
        encodings = ["utf-8", "iso-8859-1", "latin-1"]
        for encoding in encodings:
            try:
                df = pd.read_csv(uploaded_file, encoding=encoding)
                return df
            except UnicodeDecodeError:
                continue
        st.error("Unable to decode the CSV file with available encodings.")
        return None
    elif uploaded_file.name.endswith(".xlsx"):
        try:
            df = pd.read_excel(uploaded_file)
            return df
        except Exception as e:
            st.error(f"Error reading Excel file: {e}")
            return None
    else:
        st.error("Unsupported file type")
        return None


# Function to save file details to the database
def save_file_details_to_db(filename, file_path):
    conn = get_db_connection()
    cursor = conn.cursor()
    file_id = None
    try:
        cursor.execute(
            "INSERT INTO files (filename, file_path) VALUES (%s, %s)",
            (filename, file_path),
        )
        conn.commit()
        file_id = cursor.lastrowid
        st.success("File details saved to database.")
    except mysql.connector.Error as err:
        st.error(f"Error: {err}")
    finally:
        cursor.close()
        conn.close()
    return file_id


# Function to create and query the agent
def query_data(df, query):
    # Save the DataFrame to a CSV file temporarily
    csv_file_path = "temp_file.csv"
    df.to_csv(csv_file_path, index=False)

    # Create the CSV agent
    groq_api_key = st.secrets["GROQ"]["API_KEY"]
    llm = ChatGroq(temperature=0, model="llama3-70b-8192", api_key=groq_api_key)
    agent = create_csv_agent(
        llm, csv_file_path, verbose=True, allow_dangerous_code=True
    )

    # Query the agent
    response = agent.invoke(query)
    return response


# Function to save chat history to the database
def save_chat_history_to_db(file_id, query, response):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO chat_history (file_id, query, response) VALUES (%s, %s, %s)",
            (file_id, query, response),
        )
        conn.commit()
    except mysql.connector.Error as err:
        st.error(f"Error: {err}")
    finally:
        cursor.close()
        conn.close()


# Function to fetch chat history from the database
def fetch_chat_history(file_id):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(
            "SELECT query, response, timestamp FROM chat_history WHERE file_id = %s ORDER BY timestamp",
            (file_id,),
        )
        history = cursor.fetchall()
    except mysql.connector.Error as err:
        st.error(f"Error: {err}")
        history = []
    finally:
        cursor.close()
        conn.close()
    return history


# Streamlit app
st.title("MART ANALYST👾")

uploaded_file = st.file_uploader("Upload the Mart data file", type=["csv", "xlsx"])
if uploaded_file is not None:
    df = load_file(uploaded_file)
    if df is not None:
        st.write("Data Preview:")
        st.write(df.head())

        # Save the file to a directory
        save_directory = "uploads"
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        file_path = os.path.join(save_directory, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Save file details to the database and get the file_id
        file_id = save_file_details_to_db(uploaded_file.name, file_path)

        if file_id:
            # Display chat history
            st.sidebar.title("Chat History")
            chat_history = fetch_chat_history(file_id)
            for entry in chat_history:
                st.sidebar.write(f"{entry['timestamp']}")
                st.sidebar.write(f"User: {entry['query']}")
                st.sidebar.write(f"Assistant: {entry['response']}")
                st.sidebar.write("---")

            # User input for new query
       
            query = st.text_input("Enter your query:")
            if st.button("Submit Query"):
                if query:
                    st.write("Loading..")
                    response = query_data(df, query)
                    st.write("Query Result:")
                    st.write(response)

                    # Save chat history to the database
                    save_chat_history_to_db(file_id, query, response)

                    # Update chat history display
                    st.sidebar.write(f"{pd.Timestamp.now()}")
                    st.sidebar.write(f"User: {query}")
                    st.sidebar.write(f"Assistant: {response}")
                    st.sidebar.write("---")
                else:
                    st.error("Please enter a query")
