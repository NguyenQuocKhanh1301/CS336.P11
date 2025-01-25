from PyPDF2 import PdfReader
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from langchain_groq import ChatGroq
import groq
import numpy as np

def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def creat_embedding():
    modelPath = "dangvantuan/vietnamese-document-embedding"
    model_kwargs = {'device':'cpu'}
    encode_kwargs = {'normalize_embeddings': False}

    embeddings = HuggingFaceEmbeddings(
    model_name=modelPath, 
    model_kwargs={**model_kwargs, "trust_remote_code": True}, 
    encode_kwargs=encode_kwargs,
    )
    return embeddings
# Create retriever with chunking
def create_retriever_with_chunks(text, chunk_size=500, chunk_overlap = 10):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(text)
    
    vector_store = FAISS.from_texts(chunks, creat_embedding())
    return vector_store

# Initialize custom LLM pipeline
def pipeline_llm():
    # tokenizer = AutoTokenizer.from_pretrained("minhtoan/gpt2-vietnamese", padding = True,  truncation=True, max_length = 1024)
    # model = AutoModelForCausalLM.from_pretrained("minhtoan/gpt2-vietnamese")
    model = groq.Groq(api_key='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')
    # pipe = pipeline(
    #     "text-generation",
    #     model=model,
    #     # tokenizer=tokenizer,
    #     model_kwargs={"torch_dtype": "auto"},
    #     # max_new_tokens = 1024,
    #     device_map="auto",
    # )
    # llm = HuggingFacePipeline(pipeline=pipe,
    #                           model_kwargs = {"temperature": 0.3,  "max_new_tokens": 1024})
    llm = ChatGroq(api_key='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',
                   model='llama-3.3-70b-versatile')
    return llm

# Use LLM for answering questions
def use_llm_for_answering(question, retriever):
    llm = pipeline_llm()
    embedding = creat_embedding()
    query_embedding = embedding.embed_query(question)
    # Step 2: Retrieve documents from the retriever
    results = retriever.similarity_search(question, k=3)  
    
    # Step 3: Compute embeddings for all retrieved documents
    doc_embeddings = [embedding.embed_query(doc.page_content) for doc in results]  # Generate embeddings dynamically
    
    # Step 4: Compute cosine similarity
    similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
    
    # Find the best match
    best_match_idx = np.argmax(similarities)
    best_match = results[best_match_idx]  # Retrieve the best matching document
    best_score = similarities[best_match_idx]
        
    # Generate response using retrieved content as context
    if best_match:
        retrieved_content = best_match.page_content
        prompt = f"Dựa vào ngữ cảnh {retrieved_content}\n trả lời câu hỏi sau: {question}\n "
        response = llm.invoke(prompt).content
        
    return response