# Crop Disease Prediction Platform

![AgriScan Logo/Banner](placeholder-for-banner.png) <!-- 📸 Add a screenshot or banner here -->

A full-stack, AI-powered web application for identifying crop diseases from plant leaf images. This platform allows users to select a crop type (Chilli, Rice, Finger Millet, Sugarcane) and upload an image of a leaf to receive an instant disease diagnosis and a confidence score.

## ✨ Features
- **Intelligent Disease Detection**: Utilizes deep learning Keras (`.h5`) models.
- **Offline Capable**: Models are loaded directly from the local filesystem—no external APIs or internet connection required for inference.
- **Crop Variety**: Supports multiple crops including Paddy (Rice), Chilli, Finger Millet (Ragi), and Sugarcane.
- **Modern UI/UX**: Professional, responsive frontend built with Next.js/React.
- **Fast Backend**: Asynchronous inference using FastAPI.

## 🛠️ Technology Stack
- **Frontend**: Next.js, React, CSS
- **Backend**: FastAPI, Uvicorn, Python
- **Machine Learning**: TensorFlow / Keras
- **Image Processing**: Pillow, NumPy

## 📂 Project Structure
```
CROP_prediction/
├── backend/
│   ├── main.py            # FastAPI application logic and API endpoints
│   ├── utils.py           # Image preprocessing utilities
│   ├── requirements.txt   # Python dependencies
│   └── models/            # Directory containing the ML models (*.h5)
├── frontend/
│   ├── public/            # Static assets
│   ├── src/               # React components, Next.js pages, styles
│   ├── package.json       # Node.js dependencies
│   └── next.config.js     # Next.js configuration
├── .gitignore             # Ignored files and directories
├── .env.example           # Example environment variables
├── README.md              # Project documentation
└── LICENSE                # MIT License
```

## 🚀 Getting Started

### Prerequisites
- Node.js (v18+)
- Python (3.9+)
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/Gnaneshwar9611/CROP_prediction.git
cd CROP_prediction
```

### 2. Configure Machine Learning Models
Due to GitHub's file size limits, the deep learning `.h5` models may need to be downloaded separately if not using Git LFS. 
Ensure your models are placed in the `backend/models/` directory:
- `backend/models/chilli_model.h5`
- `backend/models/rice_model.h5`
- `backend/models/finger_millet_model.h5`
- `backend/models/sugarcane_model.h5`

### 3. Backend Setup
Navigate to the backend directory, create a virtual environment, and start the server:
```bash
cd backend
python -m venv venv
venv\Scripts\activate      # On Windows
source venv/bin/activate   # On Mac/Linux
pip install -r requirements.txt
uvicorn main:app --reload
```
The FastAPI backend will start running on `http://127.0.0.1:8000`.

### 4. Frontend Setup
Open a new terminal, navigate to the frontend directory, and start the development server:
```bash
cd frontend
npm install
npm run dev
```
The Next.js frontend will start running on `http://localhost:3000`.

### 5. Access the Application
Open your web browser and navigate to `http://localhost:3000` to interact with the platform.

## 📸 Screenshots
<!-- 📸 Add actual screenshots of your application here -->
*Add screenshots of the landing page, upload workflow, and prediction results.*

## 🔮 Future Improvements
- [ ] Add more crops (e.g., Tomato, Potato).
- [ ] Incorporate treatment recommendations and agricultural prevention tips.
- [ ] Implement user authentication to track historical diagnosis.
- [ ] Deploy the platform to managed cloud services (Vercel, Render, AWS).

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
