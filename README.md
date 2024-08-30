# Vecihi

This application was developed as part of the TravelX Ideathon. It's a RAG-LLM-GenAI based chatbot designed to answer questions on customers' minds. You send your question and the chatbot generates the most appropriate answer from vectordb or agent.

### Features 
- **Buy ticket**: Finding and purchasing tickets can be done by just talking
- **RAG:** Can answer all questions about THY and aviation
- **Dictionary:** Provides definitions for aviation terms
- **Native:** Speaks a native language
- **Trip plan:** You can create a detailed trip plan for the destination
- **Seat matching**: Pairs people with similar tastes for a better travel experience

### Technologies Used
- [OpenAI](https://platform.openai.com/docs/api-reference/introduction) - OpenAI version: 1.35.14
- [LangChain](https://python.langchain.com/v0.2/docs/introduction/) - LangChain version: 0.2.5
- [Streamlit](https://docs.streamlit.io/) - Streamlit version: 1.38.0



---

## Requirements

### Environment

Ensure that your Python version is set to `3.12.3` (pip version is `24.0`):

```bash
python --version
```
- Setting up Virtualenv:

```bash
pip install virtualenv
```
- Creating a Virtual Environment:
```bash
virtualenv venv
```
- Activating the Virtual Environment:
```bash
source venv/bin/activate
```
- Installing the necessary libraries:
```bash
pip install -r requirements.txt
```

#### Configuration

- Set up your .env file:

```bash
cd <project-directory>
```

```bash
- Create the .env file and add your OPENAI_API_KEY:

    OPENAI_API_KEY='key' # .env file

```
#### Create VectorDB

```bash
python3 create_database.py
```

#### Run

- Launch the Streamlit app in terminal:
```bash
streamlit run app.py
```
----


