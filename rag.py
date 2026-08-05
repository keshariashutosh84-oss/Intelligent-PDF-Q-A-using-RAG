import os
import tempfile

from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate


embedding_model = MistralAIEmbeddings(model="mistral-embed")
llm = ChatMistralAI(model="mistral-small-2506")


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
    # PyPDFLoader needs a file path, so save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(documents)

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model
        )

        return vectorstore, len(documents), len(chunks)

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def ask_question(vectorstore, question):
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )

    docs = retriever.invoke(question)

    context = "\n\n".join(doc.page_content for doc in docs)

    final_prompt = prompt.invoke({
        "context": context,
        "question": question
    })

    response = llm.invoke(final_prompt)

    return response.content, docs