# AyurGenX — AI-Powered Ayurvedic Face Scan & Assessment

AyurGenX combines advanced Computer Vision (EfficientNet) with an intelligent, adaptive assessment flow to provide users with a personalised Ayurvedic health profile from a single face scan.

## 🚀 Features

- **Live Face Scan**: Upload a selfie. The system uses CLAHE and denoising to counter phone camera beautification.
- **10-Class Disease Detection**: Uses a custom-trained EfficientNet model to detect conditions like Acne, Pigmentation, Redness, and Wrinkles with confidence and severity scores.
- **Biomarker Extraction**: Extracts extra CV metrics (Skin Glow, Roughness, Dullness, Redness Index).
- **Hierarchical Question Engine**: No more boring 30-question forms. The system asks 3 to 5 targeted questions *based on what the face scan finds*.
- **Ayurvedic Synthesis**: Maps visual conditions and lifestyle answers to Doshas (Vata, Pitta, Kapha) and provides tailored Ayurvedic remedies.
- **Premium Glassmorphism UI**: Beautiful, engaging, responsive dark-theme design.

---

## 🛠️ Local Setup

### 1. Install Dependencies
Ensure you have Python 3.9+ installed.

```bash
cd backend
pip install -r requirements.txt
```

### 2. Add the Model Weights
Your `efficientnet_best.pt` file is too large for GitHub. You must download it. 

If it's on Google Drive, use the setup script from the root folder:

```bash
python setup_model.py --drive_id YOUR_GOOGLE_DRIVE_FILE_ID
```
*(Replace `YOUR_GOOGLE_DRIVE_FILE_ID` with the actual file ID from your Google Drive link).*

Alternatively, manually place the `efficientnet_best.pt` file inside the `backend/models/` folder.

### 3. Run the Server
The FastAPI server serves both the API and the static frontend pages.

```bash
# from the project root
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. View the App
Open your browser and navigate to:
[http://localhost:8000](http://localhost:8000)

---

## ☁️ Deployment (Render / Railway)

The project includes a `render.yaml` and a `Procfile` for easy deployment on free-tier platforms.

### Important Note on the Model Size
Because the `.pt` file is not in Git, you have two options for deployment:

**Option A: Add model download to the build step (Recommended for Render free tier)**
Update the `buildCommand` in `render.yaml` (or your build settings) to run the download script before starting:
```yaml
buildCommand: pip install -r backend/requirements.txt && python setup_model.py --drive_id YOUR_DRIVE_ID
```

**Option B: Persistent Disk (Render paid tier)**
If you use a paid Render plan, a 2GB persistent disk is configured in `render.yaml`. You can mount it to `backend/models` and manually upload the model file once.

---

## 📁 Project Structure

```text
├── backend/
│   ├── main.py              # FastAPI server (API + Frontend serving)
│   ├── cv_model.py          # PyTorch wrapper, preprocessing & inference
│   ├── question_engine.py   # State machine for the adaptive questions
│   ├── question_tree.json   # The logic tree driving the assessment
│   ├── label_map.json       # Class definitions for the ML model
│   ├── requirements.txt     # Python dependencies
│   └── models/              # Directory for efficientnet_best.pt
├── frontend/
│   ├── index.html           # Landing page
│   ├── scan.html            # Upload & Scanning UI
│   ├── chat.html            # Adaptive assessment chat UI
│   ├── results.html         # Final report & recommendations
│   └── static/css/style.css # Global Design System
├── setup_model.py           # Google Drive downloader script
├── Procfile                 # Deployment config
└── render.yaml              # Render.com deployment config
```

---
Built with ❤️ for AyurGenX
