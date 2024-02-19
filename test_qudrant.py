# from langchain.llms import AzureOpenAI
# llm = AzureOpenAI(deployment_name= config.DEPLOYMENT_NAME,openai_api_key=config.AZURE_OPENAI_KEY,api_version=config.AZURE_API_VERSION,
#     azure_endpoint=config.AZURE_OPENAI_ENDPOINT)

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader

# from openai import AzureChatOpenAI
# from openai import AzureOpenAI
from langchain.llms import OpenAIChat
import os
from dotenv import load_dotenv

dotenv_path = '/home/kic/yskids/service/app/credentials/.env'
load_dotenv(dotenv_path)


def rag(path):
    # Define the directory containing the text files
    sample_data_dir = f"{path}/sample_data"

    # Iterate over each text file in the directory
    for filename in os.listdir(sample_data_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(sample_data_dir, filename)
            
            # Load text from the current file
            loader = TextLoader(file_path)
            documents = loader.load()
            
            # Define namespace based on the filename
            namespace = os.path.splitext(os.path.basename(file_path))[0]
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=20)
            texts = text_splitter.split_documents(documents)
            
            # Embed and store the texts
            persist_directory = f'{path}/db'
            embedding = OpenAIEmbeddings(
                model='text-embedding-ada-002',
                openai_api_key=os.getenv('OPENAI_API_KEY')
            )
            chroma_db = Chroma.from_documents(
                documents=texts, 
                embedding=embedding,
                persist_directory=persist_directory,
                collection_name=namespace
            )
            
            # Define OpenAIChat instance
            llm = OpenAIChat(temperature=0, model_name='gpt-3.5-turbo', api_key=os.getenv('OPENAI_API_KEY'))
            
            # Define the RetrievalQA instance from the chain
            chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=chroma_db.as_retriever()
            )
            
            # Define a query
            query = "이 사람이 AI에 관련해서 전문가"
            
            # Get response for the query
            response = chain(query)
            
            # Print response
            print(f"Response for {filename}: {response}")


