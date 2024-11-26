import os
import chainlit as cl
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import torch

# Constants
DB_FAISS_PATH = os.path.join(os.getcwd(), 'vectorstore', 'db_faiss')
MODEL_NAME = "TheBloke/Llama-2-7B-Chat-GGML"
MODEL_TYPE = "llama"
MODEL_DIR = os.path.expanduser("~/.cache/transformers")

# Custom prompt for QA
custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

# Ensure model is downloaded at runtime
# def ensure_model_downloaded():
#     os.makedirs(MODEL_DIR, exist_ok=True)
#     model_path = os.path.join(MODEL_DIR, MODEL_NAME)
#     if not os.path.exists(model_path):
#         print(f"Downloading model: {MODEL_NAME}")
#         CTransformers(model=MODEL_NAME, model_type=MODEL_TYPE)
#     else:
#         print(f"Model {MODEL_NAME} is already present.")
def ensure_model_downloaded():
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, MODEL_NAME)
    
    if not os.path.exists(model_path):
        try:
            print(f"Downloading model: {MODEL_NAME}")
            CTransformers(model=MODEL_NAME, model_type=MODEL_TYPE)
            print(f"Model {MODEL_NAME} downloaded successfully.")
        except Exception as e:
            print(f"Error occurred during model download: {str(e)}")
            print("Please check your internet connection or try downloading the model manually.")
    else:
        print(f"Model {MODEL_NAME} is already present.")


# Set custom prompt for QA
def set_custom_prompt():
    print("Setting custom prompt template...")
    return PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])

# Load the language model
def load_llm():
    print("Loading LLM...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    return CTransformers(
        model=os.path.join(MODEL_DIR, MODEL_NAME),
        model_type=MODEL_TYPE,
        max_new_tokens=256,
        temperature=0.5,
        device=device
    )

# Create QA Bot
def qa_bot():
    print("Creating QA bot...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': device}
    )
    print(f"Loading FAISS database from: {DB_FAISS_PATH}")
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    print("FAISS database loaded successfully.")
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': qa_prompt}
    )

# Chainlit handlers
@cl.on_chat_start
async def start():
    print("Starting chat session...")
    ensure_model_downloaded()  # Ensure model is ready
    chain = qa_bot()
    cl.user_session.set("chain", chain)
    print("Bot initialized and session data set.")
    await cl.Message(content="Hi, Welcome to CareConnect. What is your query?").send()
    print("Greeting message sent.")

@cl.on_message
async def main(message: cl.Message):
    print("Received message:", message.content)
    chain = cl.user_session.get("chain")
    conversation_context = message.content

    try:
        # Generate response using the LLM
        print("Generating response using the LLM...")
        result = chain({"query": conversation_context})
        print("Response generated.")
        answer = result.get("result", "Sorry, I couldn't find an answer.")
        sources = result.get("source_documents", [])

        # Format the sources
        formatted_sources = "\n\n**Sources:**\n" if sources else "\nNo sources found."
        for doc in sources:
            source_name = doc.metadata.get('source', 'Unknown Source')
            page_number = doc.metadata.get('page', 'N/A')
            formatted_sources += f"- {source_name} (Page {page_number})\n"

        # Combine the answer with formatted sources
        final_output = f"{answer}{formatted_sources}"

        # Send the response
        print("Sending response...")
        await cl.Message(content=final_output).send()
        print("Response sent.")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        await cl.Message(content=f"An error occurred: {str(e)}").send()
