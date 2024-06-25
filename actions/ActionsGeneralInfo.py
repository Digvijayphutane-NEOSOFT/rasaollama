import json
import os
import re
import PyPDF2 
import numpy as np
import pandas as pd
import pymysql
pymysql.install_as_MySQLdb()
import sqlalchemy
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import OllamaEmbeddings
from sklearn.feature_extraction.text import TfidfVectorizer
import openai
from langchain.memory import ConversationSummaryMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch, FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI, ChatOllama
from langchain.document_loaders import PyPDFLoader, JSONLoader, CSVLoader
from langchain.docstore.document import Document

# Set API Key
# os.environ['OPENAI_API_KEY'] = 'YOUR-KEY'
# openai.api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOllama(model='llama2')

dialogues = [
    {"messages": [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "I'm your personal banking assistant. How can I assist you today?"}
    ]},
    {"messages": [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "I'm your personal banking assistant. How can I assist you today?"}
    ]},
    {"messages": [
        {"role": "user", "content": "Can you help me with Math or Physics?"},
        {"role": "assistant", "content": "Sorry, I am a banking assistant chatbot, can only help with banking staffs."}
    ]},
    {"messages": [
        {"role": "user", "content": "Can you tell me my account balance?"},
        {"role": "assistant", "content": "Sure, could you please specify which account you’d like to check? For example, save or pay?"}
    ]},
    {"messages": [
        {"role": "user", "content": "Save account, please."},
        {"role": "assistant", "content": "Your save account balance is $5,000. Is there anything else I can assist you with?"}
    ]},
    {"messages": [
        {"role": "user", "content": "Show me my last transactions."},
        {"role": "assistant", "content": "Your most recent transaction was Transaction No. 5000010 on 2024/10/14, lt was a debit transaction for -200 CHF at BCC restaurant. After this transaction, your balance in the Pay account was 2.000 CHF． Would you like to see more transactions or need help with something else?"}
    ]},
    {"messages": [
        {"role": "user", "content": "Show me my last 2 transactions."},
        {"role": "assistant", "content": """Your most recent 2 transaction as follow:
            1. Transaction Number: No. 5000010
              Date: 2024/10/14
              Change: -200 CHF
              Balance after: 2,000 CHF
              Information: BCC restaurant bills

            2. Transaction Number: No. 5000009
              Date: 2024/10/13
              Change: -60 CHF
              Balance after: 2,200 CHF
              Information: COOP City cost
        Would you like to see more transactions or need help with something else?"""}
    ]},
    {"messages": [
        {"role": "user", "content": "No, thank you."},
        {"role": "assistant", "content": "You're welcome! If you have more questions in the future, feel free to ask. Have a great day!"}
    ]},
    {"messages": [
        {"role": "user", "content": "What is my account number?"},
        {"role": "assistant", "content": "Sure, Your account number is [Account Number]. Would you like to see more transactions or need help with something else?"}
    ]},
    {"messages": [
        {"role": "user", "content": "Thank you."},
        {"role": "assistant", "content": "You're welcome! If you have more questions in the future, feel free to ask. Have a great day!"}
    ]},
    {"messages": [
        {"role": "user", "content": "Thanks!"},
        {"role": "assistant", "content": "You're welcome! If you have any more questions in the future, feel free to ask. Have a great day!"}
    ]},
    {"messages": [
        {"role": "user", "content": "What is the interest rate on a savings account?"},
        {"role": "assistant", "content": "Our current interest rate for a savings account is 2% APY. Do you have other questions?"}
    ]},
    {"messages": [
        {"role": "user", "content": "What is the interest rate on 1 year deposit?"},
        {"role": "assistant", "content": "Our current interest rate for 1-year time deposit is 3 per cent. Do you have other questions?"}
    ]},
    {"messages": [
        {"role": "user", "content": "Thanks! Bye!"},
        {"role": "assistant", "content": "You're welcome! If you have any more questions in the future, feel free to ask. Have a great day!"}
    ]}
]

def extract_text_from_pdf(pdf_path):
    # Initialize a variable for text
    text = ""

    # Open the PDF file
    with open(pdf_path, 'rb') as file:
        # Create a PDF reader object
        pdf_reader = PyPDF2.PdfReader(file)

        # Extract text from each page
        for page_num in range(len(pdf_reader.pages)):
            # Get a page
            page = pdf_reader.pages[page_num]

            # Extract text
            text += page.extract_text()

    return text

def extract_text_from_json_recursive(data, indent=""):
    text = ""

    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                text += f"{indent}{key}:\n"
                text += extract_text_from_json_recursive(value, indent + "  ")
            else:
                text += f"{indent}{key}: {value}\n"
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, (dict, list)):
                text += extract_text_from_json_recursive(item, indent + "  ")
            else:
                text += f"{indent}- {item}\n"
    else:
        text += f"{indent}{data}\n"

    return text

def extract_keywords_tfidf(docs, max_features=50):
    if len(docs) == 1:
        docs = [sentence.strip() for sentence in docs[0].split('.') if sentence]

    # Create and configure the TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer(max_df=1.0, stop_words='english', max_features=None)

    # Train the model
    tfidf_matrix = tfidf_vectorizer.fit_transform(docs)

    # Retrieve the vocabulary
    feature_names = np.array(tfidf_vectorizer.get_feature_names_out())

    # Aggregate the scores for each feature
    aggregated_scores = np.sum(tfidf_matrix.toarray(), axis=0)

    # Sort indices in descending order of aggregated_scores
    sorted_indices = np.argsort(aggregated_scores)[::-1]

    # Extract top features based on sorted_indices, ensuring we don't exceed the number of available features
    top_n = min(max_features, len(feature_names))
    top_features = feature_names[sorted_indices[:top_n]]

    refined_keywords = [word for word in top_features if re.search('[a-zA-Z]', word)]

    return set(refined_keywords)

def create_documents_from_json(data, parent_key=''):
    documents = []

    if isinstance(data, dict):
        for key, value in data.items():
            nested_key = f'{parent_key}.{key}' if parent_key else key
            documents.extend(create_documents_from_json(value, nested_key))
    elif isinstance(data, list):
        for item in data:
            documents.extend(create_documents_from_json(item, parent_key))
    else:
        # Combine key and value into page_content
        content = f"{parent_key}: {json.dumps(data)}" if parent_key else json.dumps(data)
        document = Document(page_content=str(content))
        documents.append(document)

    return documents

def load_data_from_database():
    # Connect to the MySQL database using pymysql
    engine = sqlalchemy.create_engine('mysql+pymysql://root:password@localhost:3306/bank')
    query = "SELECT * FROM Transactions"
    df = pd.read_sql(query, engine)
    
    # Combine data into a single string for creating a Document object
    data_str = df.to_string(index=False)
    document = Document(page_content=str(data_str))
    print("##########################")
    print(document)
    print("##########################")

    return [document]

def qaRetrival():
    # Paths
    # json_path = './sources/questionAnswers.json'
    # csv_path = './sources/Annual_report.csv'
    # pdf_path = './sources/Annual_report.pdf'
    json_path = './data/client_data.json'
    csv_path = './data/Transactions.csv'
    pdf_path = './data/data.pdf'

    # Loaders
    loaders = []

    # JSON Loader
    # if os.path.exists(json_path):
    #     with open(json_path) as f:
    #         data = json.load(f)
    #     documents = create_documents_from_json(data)
    #     json_loader = JSONLoader(file_path=json_path, jq_schema='.messages[].content')
    #     loaders.append(json_loader)

    # CSV Loader
    # if os.path.exists(csv_path):
    #     csv_loader = CSVLoader(file_path=csv_path)
    #     loaders.append(csv_loader)

    # PDF Loader
    if os.path.exists(pdf_path):
        pdf_text = extract_text_from_pdf(pdf_path)
        pdf_loader = Document(page_content=str(pdf_text))
        loaders.append(pdf_loader)

    # Database Loader
    database_document = load_data_from_database()
    loaders.extend(database_document)
    print(loaders)


    # Create documents and embeddings
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    # documents = []
    # documents = [Document(page_content=item) for item in loaders]

    # for loader in loaders:
    #     # documents.extend(loader.load())
    #     documents.extend(loader)

    # docs = text_splitter.split_documents(loaders)
    embeddings = OllamaEmbeddings()
    db = FAISS.from_documents(loaders, embeddings)

    # Memory
    memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history", return_messages=True)

    # Conversational Retrieval Chain
    qa = ConversationalRetrievalChain.from_llm(llm, retriever=db.as_retriever(), memory=memory)
    print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
    print(qa)
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    return qa

keywords = {"account", "balance", "transaction", "transfer"}
common_greetings = {"hi", "hello", "thank you", "thanks", "goodbye", "bye", "Yes", "No", "OK", "Sure","Transaction","Transactions"}

def is_question_relevant(question: str, keywords: set, common_phrases: set) -> bool:
    # Combine the keywords and common phrases
    all_relevant_words = keywords.union(common_phrases)

    # Check if the question contains any of the relevant words
    return any(word.lower() in question.lower() for word in all_relevant_words)