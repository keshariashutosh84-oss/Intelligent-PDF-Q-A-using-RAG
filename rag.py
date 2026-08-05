import os
import tempfile
from dotenv import load_dotenv

load_dotenv()

from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate


# Get Mistral API key
api_key = os.getenv("MISTRAL_API_KEY")

if not api_key:
    raise ValueError("MISTRAL_API_KEY not found in .env file")


# Mistral models
embedding_model = MistralAIEmbeddings(
    model="mistral-embed",
    api_key=api_key
)

llm = ChatMistralAI(
    model="mistral-small-2506",
    api_key=api_key
)


# Prompt
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful AI assistant.

Answer the user's question using only the provided context.

If the answer is not available in the context, say:
"I could not find the answer in the document."

Do not make up information."""
    ),
    (
        "human",
        """Context:
{context}

Question:
{question}"""
    )
])


def process_pdf(uploaded_file):
    # Save uploaded PDF temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_path = temp_file.name

    try:
        # Load PDF
        loader = PyPDFLoader(temp_path)
        documents = loader.load()

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(documents)

        # Create vector store
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model
        )

        return vectorstore, len(documents), len(chunks)

    finally:
        # Delete temporary PDF
        if os.path.exists(temp_path):
            os.remove(temp_path)


def ask_question(vectorstore, question):

    # Create MMR retriever
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )

    # Find relevant chunks
    docs = retriever.invoke(question)

    # Combine retrieved chunks
    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    # Add context and question to prompt
    final_prompt = prompt.invoke({
        "context": context,
        "question": question
    })

    # Generate response
    response = llm.invoke(final_prompt)

    return response.content, docs