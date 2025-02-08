# 🌟 Roman Urdu Poetry Generator  

## 🚀 Project Description  
This project is a web-based **Roman Urdu Poetry Generator** built using AI and TensorFlow. It allows users to generate creative poetry in real time by entering a simple seed text.  

The backend is powered by **FastAPI**, and the user-friendly frontend features a sleek dark theme with a responsive design for a smooth experience.  

---

## ⚙️ Features  
- **Real-time poetry generation**  
- Powered by custom-trained TensorFlow models  
- Interactive and modern dark-themed UI  
- Efficient backend using FastAPI  

---

## 🛠️ Tech Stack  
- **Frontend:** HTML, CSS (gradient-based UI)  
- **Backend:** FastAPI  
- **AI Model:** TensorFlow  
- **Deployment:** Local or cloud environments  

---

## 🚀 Installation and Usage  

### 1. Clone the Repository  
```bash
git clone <repository_url>
cd <repository_name>
```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
### 3. Run the Application
```bash
uvicorn app:app --reload
```
### 4. Open the Application
Go to http://127.0.0.1:8000 in your browser.

```bash
.
├── app.py                # FastAPI backend  
├── poetry_model/          # Model and mapping files  
│   ├── roman_urdu_poetry_model.keras  
│   ├── char2idx.json  
│   └── idx2char.npy  
├── templates/  
│   └── index.html         # Frontend UI  
├── requirements.txt       # Python dependencies  
└── README.md              # Project documentation  
```
