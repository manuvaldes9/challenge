# 🤖 Promtior Chatbot

A specialized chatbot designed to answer questions about **Promtior**, built on Retrieval-Augmented Generation (RAG).

---

## ✨ Key Features

* **RAG Architecture**
* **Automatic Web Scraping**
* **PDF Document Processing**
* **RESTful API**
* **Interactive Playground**
* **Semantic Search**
* **Vector Storage with FAISS**

---

## 🛠️ Technology Stack

* **LangChain**
* **LangServe**
* **FastAPI**
* **OpenAI API**
* **FAISS**
* **Railway**

---

## 📌 Prerequisites (to run it locally)

Ensure you have the following installed:

* Python 3.11 or higher
* OpenAI API Key
* Git

---

## 🛠️ Installation

1\. **Clone the Repository**

```bash
git clone https://github.com/manuvaldes9/challenge.git
cd challenge
```

2\. **Create a Virtual Environment**

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3\. **Install Dependencies**

```bash
pip install -r requirements.txt
```

4\. **Configure Environment Variables**

Create a `.env` file in the root of the project:

```bash
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

---

## 📂 Data Preparation (Optional)

* To enhance your knowledge base, place additional PDF documents in the `data/` folder.
* Web scraping runs automatically when the application starts.

---

## 🚀 Running the Application

**Local Development**

Launch the FastAPI server:

```bash
uvicorn main:app --reload
```

Access the chatbot via:

* **Playground:** [http://localhost:8000/chat/playground](http://localhost:8000/chat/playground)

---

## 🚀 Deployed Version (Railway)

* **Access the live chatbot:** [rag-production-1a46.up.railway.app/chat/playground](https://rag-production-1a46.up.railway.app/chat/playground)
