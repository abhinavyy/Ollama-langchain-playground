
````markdown
# LangChain + Ollama Demo 🤖

A **basic demo project** integrating [LangChain](https://www.langchain.com/) with [Ollama](https://ollama.ai/) for a simple local chatbot experience.  

---

## Features

- Simple Q&A demo using LangChain + Ollama
- Easy local setup; no API keys required
- Demonstrates basic LangChain functionality
-<img width="2877" height="1799" alt="Screenshot 2025-08-21 014028" src="https://github.com/user-attachments/assets/533b2127-0b64-4fc5-be4b-34b69dea8eb7" />
-<img width="2879" height="1459" alt="Screenshot 2025-08-21 014129" src="https://github.com/user-attachments/assets/0d19a784-a973-4671-8391-a307b501f270" />



---

## Prerequisites

- Python 3.10+  
- [Ollama installed](https://ollama.ai/download)  
- pip package manager  

---

## Setup

1. **Clone the repository**

```bash
git clone https://github.com/YOUR-USERNAME/langchain-ollama-demo.git
cd langchain-ollama-demo
````

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Run the app**

* If using a Python script (`app.py`):

```bash
python app.py
```

* If using Streamlit:

```bash
streamlit run app.py
```

---

## Project Structure

```
langchain-ollama-demo/
├─ app.py              # Main demo app
├─ requirements.txt    # Python dependencies
├─ .env         # Langchain api. auto del in 30 days 
├─ 1.2.1-Simpleapp.ipynb
```

---

## Notes

* Accuracy is **limited** — this is purely experimental.
* LangChain + Ollama models run **locally**, so performance depends on your machine.

---
