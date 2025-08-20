
# LangChain + Ollama Demo 🤖

A **basic demo project** integrating [LangChain](https://www.langchain.com/) with [Ollama](https://ollama.ai/) for a simple local chatbot experience.

---

## Features

- Simple Q&A demo using LangChain + Ollama
- Easy local setup; no API keys required
- Demonstrates basic LangChain functionality

![Screenshot 1](https://github.com/user-attachments/assets/533b2127-0b64-4fc5-be4b-34b69dea8eb7)
![Screenshot 2](https://github.com/user-attachments/assets/0d19a784-a973-4671-8391-a307b501f270)

---

## Prerequisites

- Python 3.10+  
- [Ollama installed](https://ollama.ai/download)  
- pip package manager  

---

## Setup

1. **Clone the repository**

```bash
git clone https://github.com/abhinavyy/Ollama-langchain-playground.git
cd langchain-ollama-demo
````

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Run the app**

* Python script:

```bash
python app.py
```

* Streamlit:

```bash
streamlit run app.py
```

---

## Project Structure

```
langchain-ollama-demo/
├─ app.py                   # Main demo app
├─ requirements.txt         # Python dependencies
├─ .env                     # LangChain API (auto delete in 30 days)
├─ 1.2.1-Simpleapp.ipynb    # Optional notebook demo
```

---

## Notes

* Accuracy is **limited** — purely experimental.
* LangChain + Ollama models run **locally**, so performance depends on your machine.

```
```
