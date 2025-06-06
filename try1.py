import pandas as pd
import streamlit as st
import mysql.connector
import os
import json
from langchain_groq import ChatGroq
from langchain_experimental.agents.agent_toolkits import create_csv_agent


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA


# Configure database connection
def get_db_connection():
    return mysql.connector.connect(
        host="localhost", user="root", password="MinnuSri_4444", database="streamlit_db"
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


# Function to create and query the CSV agent (existing, unchanged)
def query_data(df, query):
    # Save the DataFrame to a CSV file temporarily
    csv_file_path = "temp_file.csv"
    df.to_csv(csv_file_path, index=False)

    # Create the CSV agent
    groq_api_key = st.secrets["GROQ"]["API_KEY"]
    llm = ChatGroq(temperature=0, model="llama3-70b-8192", api_key=st.secrets["GROQ"]["API_KEY"])
    agent = create_csv_agent(
        llm, csv_file_path, verbose=True, allow_dangerous_code=True
    )

    # Query the agent
    response = agent.invoke(query)
    return response

def query_data_rag(df, query):
    # Convert dataframe to CSV string
    csv_text = df.to_csv(index=False)

    # Split text into chunks for embeddings
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.create_documents([csv_text])

    # Create embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI"]["OPENAI_API_KEY"])

    # Create FAISS vectorstore from documents
    vectorstore = FAISS.from_documents(docs, embeddings)

    # Create retriever and RAG chain
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatGroq(temperature=0, model="llama3-70b-8192", api_key=st.secrets["GROQ"]["API_KEY"]),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
    )

    # Query the RAG chain
    result = qa_chain.run(query)
    return result


# Function to save chat history to the database
def save_chat_history_to_db(file_id, query, response):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Convert the response dictionary to a JSON string
        if isinstance(response, dict):
            response = json.dumps(response)

        # Insert into the database
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
st.title("Chatbot")

button_container = st.container()

col1, col2 = button_container.columns([1, 3])

with col1:
    st.markdown(
        """
        <a href="http://localhost/RetailLLM-main/view_uploads.php" target="_blank">
            <button style="
                background-color: #4CAF50;
                color: white;
                padding: 10px 24px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 16px;
                margin: 4px 2px;
                cursor: pointer;
                border: none;
                border-radius: 4px;
            ">View</button>
        </a>
        """,
        unsafe_allow_html=True,
    )

# Main file upload and chat interaction in the center
with col2:
    uploaded_file = st.file_uploader("Upload your file", type=["csv", "xlsx"])
    if uploaded_file is not None:
        df = load_file(uploaded_file)
        if df is not None:
            st.write("Data Preview:")
            st.write(df.head())

            # Save the file to a directory
            save_directory = "C:/xampp/htdocs/RetailLLM-main/uploads"

            if not os.path.exists(save_directory):
                os.makedirs(save_directory)

            file_path = os.path.join(save_directory, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            relative_file_path = f"uploads/{uploaded_file.name}"    

            # Save file details to the database and get the file_id
            file_id = save_file_details_to_db(uploaded_file.name, relative_file_path)

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
                if query:  # Debugging output
                    st.write("Query text input captured.")
                if st.button("Submit Query"):
                    if query:
                        st.write("Query button pressed.")

                        
                        response = query_data_rag(df, query)

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

