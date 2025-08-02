# 🎬 Anime & Movie Recommender App

> Netflix-style intelligent recommendation system for Anime & Movies built with Streamlit and Deep Learning.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Made by Prathamesh](https://img.shields.io/badge/Made%20By-Prathamesh%20Jadhav%20Abd-ff69b4)

---
---

## 🚀 Features

- 🔍 **Search-Based Recommendations**  
  Smart content suggestions using TF-IDF + cosine similarity.

- 👥 **Deep Hybrid Recommendations**  
  Based on collaborative filtering + deep learning trained on user-item ratings.

- 💖 **Pick by Genre**  
  Get curated anime or movie suggestions by your favorite genre.

- 🎲 **Random Suggestions**  
  Bored? Get fun random suggestions every time!

- 🔥 **Netflix-Style UI**  
  Modern UI with responsive card layouts, filters, dark mode, and emojis!

---

## 🧠 Tech Stack

| Category       | Tools / Frameworks                         |
|----------------|---------------------------------------------|
| 💻 Frontend     | [Streamlit](https://streamlit.io/)          |
| 🧮 ML Models    | TF-IDF, Nearest Neighbors, Deep Learning    |
| 🧠 Deep Model   | TensorFlow (Keras Functional API)           |
| 🧹 Preprocessing| Pandas, Numpy, Scikit-learn                 |
| 💅 Styling      | Custom CSS (Netflix-Inspired Design)       |

---

## 📁 Project Structure
anime-recomd/
│
├── app.py # Main Streamlit app


├── css/
│ └── style.css # Custom UI styling


├── data/
│ ├── anime.csv
│ ├── movies.csv
│ └── rating.csv # User-item rating matrix


├── model/
│ ├── hybrid_model.h5 # Trained hybrid deep model
│ ├── user_map.csv
│ └── item_map.csv
└── README.md


---

## ⚙️ How to Run Locally

1. **Clone the Repository**
```bash
git clone https://github.com/prxthu04/anime-recomd.git
cd anime-recomd

```
2. Install Dependencies
```bash
pip install -r requirements.txt
```
3.Run the App
```bash
streamlit run app.py
```

# 👨‍💻 Author
Prathamesh Jadhav 
📍 Akola, India
GitHub: @prxthu04
Gmail: Prathunotfound@gmail.com

