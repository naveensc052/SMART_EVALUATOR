# SMART_EVAL

# 🤗💬 Smart Guru - PDF Chatbot

## About
Smart Guru is an interactive chatbot designed to help students extract information from PDF documents. It is built using Streamlit, LangChain, and OpenAI LLM model.

## Features
- Upload your PDF document and ask questions about its content.
- Get model-generated answers based on the extracted text.
- Receive detailed feedback on missing points in user-generated answers.

## Technologies Used
- [Streamlit](https://streamlit.io/): For building the interactive web app.
- [LangChain](https://python.langchain.com/): Provides text splitting and embeddings.
- [OpenAI](https://platform.openai.com/docs/models): LLM model for question answering.
- [PyPDF2](https://pythonhosted.org/PyPDF2/): For reading PDF files.
- [Sentence Transformers](https://www.sbert.net/): For sentence embeddings.
- [dotenv](https://pypi.org/project/python-dotenv/): For environment variables management.

## Usage
1. Install the required packages by running `pip install -r requirements.txt`.
2. Set up your OpenAI API key in the `main.py` and `.env` file.
3. Run the Streamlit app by executing `streamlit run main.py` in your terminal.
4. Upload your PDF document and ask questions about it.
5. Get model-generated answers and feedback on your input.

## Feedback and Contributions
Feedback and contributions are welcome! Feel free to [open an issue](https://github.com/your-username/smart-guru/issues) or submit a pull request.
