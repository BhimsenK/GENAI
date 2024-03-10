from PyPDF2 import PdfReader
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from typing_extensions import Concatenate

import os
# add your API key.
os.environ["OPENAI_API_KEY"] = " "


pdfreader = PdfReader('F:/GENAI/documents/budget_speech_new.pdf')


# read text from pdf
raw_text = ''
for i, page in enumerate(pdfreader.pages):
    content = page.extract_text()
    if content:
        raw_text += content

# print(f'RAW : {raw_text}')


# We need to split the text using Character Text Split such that it sshould not increse token size
text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 800,
    chunk_overlap  = 200,
    length_function = len,
)
texts = text_splitter.split_text(raw_text)
print(f'Texts : {texts}')
print(f'length : {len(texts)}')


# Download embeddings from OpenAI
embeddings = OpenAIEmbeddings()
document_search = FAISS.from_texts(texts, embeddings)

print(f'Doc search : {document_search}')
#------------------

from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

chain = load_qa_chain(OpenAI(), chain_type="stuff")

query = "Vision for Amrit Kaal"
docs = document_search.similarity_search(query)
chain.run(input_documents=docs, question=query)

#= ==== Online ===


from langchain.document_loaders import OnlinePDFLoader
loader = OnlinePDFLoader("https://arxiv.org/pdf/1706.03762.pdf")
#!pip install unstructured
data = loader.load()
print(data)

# Download embeddings from OpenAI
embeddings = OpenAIEmbeddings()
#!pip install chromadb
from langchain.indexes import VectorstoreIndexCreator
index = VectorstoreIndexCreator().from_loaders([loader])

query = "Explain me about Attention is all you need"
index.query(query)


