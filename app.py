import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from sentence_transformers import SentenceTransformer,util
import openai

with st.sidebar:
    st.title('🤗💬 LLM Chat App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
 
    ''')
    st.write('Made for Students')

load_dotenv()

def main():
    st.header("Chat with PDF 💬")
 
 
    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')
 
    # st.write(pdf)
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
 
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
            )
        chunks = text_splitter.split_text(text=text)
 
        # # embeddings
        store_name = pdf.name[:-4]
        st.write(f'{store_name}')
        # st.write(chunks)
 
        """if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
            # st.write('Embeddings Loaded from the Disk')s
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            save_data = {'faiss_index': VectorStore._index, 'other_data': VectorStore._other_data}
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(save_data, f)"""
 
        embeddings1 = OpenAIEmbeddings()
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings1)
 
        # Accept user questions/query
        query = st.text_input("Ask questions about your PDF file:")
        # st.write(query)
 
        if query:
            docs = VectorStore.similarity_search(query=query, k=3)
 
            llm = OpenAI()
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            st.write(response)
            user_answer = st.text_input("Enter your Answer:")
            actual_answer = response
            if user_answer:
                #sentences = [actual_answer,user_answer]

                model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
                #embeddings = model.encode(sentences)
                actual_answer_embeddings = model.encode([actual_answer], convert_to_tensor=True)
                user_generated_answer_embeddings = model.encode([user_answer], convert_to_tensor=True)
                similarity_scores = util.pytorch_cos_sim(actual_answer_embeddings, user_generated_answer_embeddings)

                st.write(f"Similarity Score: {similarity_scores[0][0].item()}")

                openai.api_key = "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"
                prompt = f"""You have to give missing points to a baed only on the user-generated answer based on the following information:
                -This is the question the user tried to answer: {user_answer}
                -This is the correct answer: {actual_answer},
                and Provide detailed feedback on what points are missing in a user-generated answer"""


                response1 = openai.ChatCompletion.create(
                    model='gpt-3.5-turbo-0125',
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=150
                )

                completion_text = response1.choices[0].message['content'].strip()
                st.write("Feedback:- ")
                st.write(completion_text)


 
if __name__ == '__main__':
    main()
