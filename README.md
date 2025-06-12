# RetailLLM
This chatbot application is used to provide small mart owners summary and financial details of their marts.

# 💼 Retail Finance Advisor 🧠📊

A smart AI-based chatbot system for **financial data understanding and decision-making** tailored to small retail store owners. This project uses **Retrieval-Augmented Generation (RAG)** powered by **LLaMA-3**, **FAISS similarity search**, and **Groq API** to extract and answer questions from user-uploaded PDFs (bank statements, reports, etc.).

---

## 🚀 Key Features

- 📄 **PDF Upload & Parsing**: Upload financial documents like retail statements or invoices.
- 🔍 **FAISS Similarity Search**: Retrieve the most relevant chunks from parsed documents.
- 🧠 **LLM (LLaMA 3 via Groq API)**: Generate answers to financial queries based on context.
- 🧾 **RAG Pipeline**: Combines document retrieval with LLM reasoning.
- 🖥️ **Streamlit Interface**: Simple frontend for user interaction.
- 🗄️ **PHP + MySQL**: Stores user queries and financial summaries.

---

## 📚 Use Case

> Retail branch performance analysis and financial decision-making require efficiency and accuracy. Traditional methods often fall short. Leveraging **Large Language Models** (LLMs) provides a seamless, automated approach for extracting insights from complex documents, empowering small business owners to make smarter financial decisions.

---

## 🧰 Tech Stack

| Layer          | Tech Used                                   |
|----------------|---------------------------------------------|
| UI             | Streamlit                                   |
| Backend        | Python (RAG pipeline)                       |
| LLM Inference  | LLaMA 3 (via Groq API)                      |
| Vector Store   | FAISS (for similarity search)               |
| Storage        | PHP + MySQL (for storing results/queries)  |
| Parsing        | PyMuPDF / PDFMiner / LangChain Document Loaders |

---

## 🧪 System Architecture:



![Screenshot (188)](https://github.com/user-attachments/assets/a2ba36d7-a6db-4b6d-9133-b2404986ff9d)


## Home and Login Page:

![Screenshot (189)](https://github.com/user-attachments/assets/38ea1c95-3f17-40f0-ba16-858c6ed61d7c)


## Chatbot Page(upload pdf or .csv dataset):


![Screenshot (190)](https://github.com/user-attachments/assets/3cdaa151-4f01-4225-85dd-ce7e80bc5f67)


## Saves chat history and output page:



![Screenshot (191)](https://github.com/user-attachments/assets/10ca4836-c7bd-4bc3-acb5-16cbfa5688ad)



## Sample Prompts:

"What is the net profit for the last quarter?"

"Summarize expenses category-wise from this PDF"

"How much GST was paid?"

"Which months had negative cash flow?"





## ⚙️ Setup Instructions
## 🔧 Backend (Python)

git clone https://github.com/reddymeghna/Retail-Finance-Advisor-RAG-Streamlit.git
cd Retail-Finance-Advisor-RAG-Streamlit

Add your Groq API key in .env:
GROQ_API_KEY=your_groq_key

Run the Streamlit app:
streamlit run try2.py
