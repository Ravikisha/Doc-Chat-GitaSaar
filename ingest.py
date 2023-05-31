"""Load html from files, clean up, split, ingest into Weaviate."""
import pickle

from langchain.document_loaders import ReadTheDocsLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders import PyPDFLoader
import json
import pandas as pd
from langchain.document_loaders import DataFrameLoader
import sqlite3




def ingest_docs(docs,vector):
    """Get documents from web pages."""
    
    loader = PyPDFLoader(docs)
    raw_documents = loader.load()
    # print("Raw Document: ",raw_documents)
    
    # text_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=1000,
    #     chunk_overlap=200,
    # )
    
    # documents = text_splitter.split_documents(raw_documents)
    
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(raw_documents, embeddings)

    # Save vectorstore
    with open(vector, "wb") as f:
        pickle.dump(vectorstore, f)

if __name__ == "__main__":
    ingest_docs()

