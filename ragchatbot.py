import os

import streamlit as st
import pdfplumber
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.header("My First Chatbot")
HF_API_KEY = os.environ.get("HUGGINGFACE_API_KEY")

with st.sidebar:
    st.title("Your Document")
    uploaded_file = st.file_uploader("Upload a document", type="pdf")
    submit = st.button("Submit")


    if submit:
        if uploaded_file is not None:
            with pdfplumber.open(uploaded_file) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"        
            # st.write(text)
            text_splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", ". ", " ", ""],
                chunk_size=1000,
                chunk_overlap=200,
            )
            chunks = text_splitter.split_text(text)
            st.success(f"Split into {len(chunks)} chunks")
            #st.write(chunks)
            #Operating embeddings
            embeddings = HuggingFaceEndpointEmbeddings(
                model = "sentence-transformers/all-MiniLM-L6-v2",
                huggingfacehub_api_token=HF_API_KEY,
            )
            #store embeddings in vector DB
            vector_store = FAISS.from_texts(chunks, embeddings)
            st.session_state.vector_store = vector_store
            st.success("Vector store created and ready for retrieval!")
        else:
            st.warning("Please upload a PDF first before clicking Submit.")

            #get user query
            user_query = st.text_input("Enter your question:")

            def format_docs(docs):
                return "\n\n".join([doc.page_content for doc in docs])

            #generate response based on user query
            vector_store = st.session_state.get("vector_store")
            if vector_store is None:
                st.warning("Submit a PDF before asking a question.")
                st.stop()

            Retriever = vector_store.as_retriever(
                search_type="mmr", 
                search_kwargs={"k": 4}
            )

            llm = HuggingFaceEndpoint(
                endpoint_url="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2",
                model="sentence-transformers/all-MiniLM-L6-v2",
                Temperature=0.3,
                Max_new_tokens=1000,
                huggingfacehub_api_token=HF_API_KEY
            )

            #provide a prompt template for the LLM
            prompt = ChatPromptTemplate.from_messages
            ([
                ("system", "You are a helpful assistant that answers questions based on the provided PDF document.\n\n"
                 "Guidelines:\n"
                 "1. Use the context from the document to answer questions.\n"
                    "2. If the answer is not in the document, respond with 'I don't know.'\n"
                    "3. Provide concise and accurate answers.\n"
                    "4. Avoid adding information not present in the document.\n"
                    "5. If the question is unrelated to the document, politely inform the user that you can only answer questions based on the provided document.\n"
                    "6. If the question is ambiguous, ask for clarification.\n"
                    "7. If the question is too broad, ask the user to narrow it down.\n"
                    "8. If the question is too specific and not covered in the document, respond with 'I don't know.'\n"
                 ),
                ("human", "{context}\n\nQuestion: {question}\nAnswer:")
            ])


            chain = (
                {"context": Retriever | format_docs,"question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
            if user_query:
                response = chain.invoke({"question": user_query})
                st.write("Answer:", response)



        
      
            

