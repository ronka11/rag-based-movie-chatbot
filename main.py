import os
from pypdf import PdfReader
from langchain_community.document_loaders import TextLoader
from langchain_community.llms import HuggingFaceHub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
import torch

# Clear CUDA cache
torch.cuda.empty_cache()

# Create embeddings model outside the function
instructor_embeddings = HuggingFaceInstructEmbeddings(
    model_name='hkunlp/instructor-xl', 
    model_kwargs={'device': 'cuda'},
    encode_kwargs={'batch_size': 8, 'show_progress_bar': True}
)

def process_pdfs_from_folder(folder_path, embeddings):
    vector_stores = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            pdf_name = os.path.splitext(filename)[0]
            save_path = f'vector_store/{pdf_name}'
            
            # Check if vector store already exists
            if os.path.exists(save_path):
                print(f"Vector store for {filename} already exists. Loading...")
                db = FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)  # Add this parameter
                vector_stores.append(db)
                continue
            
            # Process the PDF if vector store doesn't exist
            print(f"Processing {filename}...")
            reader = PdfReader(file_path)
            document_text = ""
            for page in reader.pages:
                document_text += page.extract_text()

            chunk_size = 500
            chunk_overlap = 50
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            split_docs = splitter.create_documents([document_text])

            db = FAISS.from_documents(split_docs, embeddings)

            db.save_local(save_path)
            print(f"Vector store saved for {filename} at {save_path}")

            vector_stores.append(db)

    return vector_stores

# Read PDFs from the 'data' folder and create vector stores
pdf_folder = 'data'
vector_stores = process_pdfs_from_folder(pdf_folder, instructor_embeddings)

# Merge all vector stores
if vector_stores:
    merged_db = vector_stores[0]
    for db in vector_stores[1:]:
        merged_db.merge_from(db)
else:
    print("No PDFs found in the data folder.")
    exit()

# Load LLM
max_length = 500
model_temperature = 0.75
llm_model = 'tiiuae/falcon-7b-instruct'
token = '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'

llm = HuggingFaceHub(
    repo_id=llm_model,
    model_kwargs={'temperature': model_temperature, 'max_length': max_length},
    huggingfacehub_api_token=token
)

# Set up conversation memory
memory = ConversationBufferWindowMemory(
    k=2,
    memory_key="chat_history",
    output_key="answer",
    return_messages=True,
)

# Create the conversational retrieval chain
qa_conversation = ConversationalRetrievalChain.from_llm(
    llm=llm,
    chain_type='stuff',
    retriever=merged_db.as_retriever(),
    return_source_documents=True,
    memory=memory,
)

# Function to get answers from the chatbot
def get_answer(question):
    response = qa_conversation({"question": question})
    return response['answer']

# Main chat loop
print("Welcome to the RAG-based LLM Chatbot. Type 'exit' to end the conversation.")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print("Chatbot: Goodbye!")
        break
    response = get_answer(user_input)
    print("Chatbot:", response)
    print("\n")