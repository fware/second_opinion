# 🚗 Independent Auto: Second Opinion Engine

## Overview
The **Second Opinion Engine** is an AI-powered lead generation and sales tool designed specifically for independent auto repair shops. It allows customers to upload expensive dealership repair estimates (via PDF or smartphone photo) and instantly generates a competitive counter-offer based on the independent shop's actual pricing. 

By leveraging cutting-edge Vision LLMs and fuzzy-matching algorithms, the app completely automates the traditional "quote matching" process, building customer trust through transparency and generating highly qualified leads for the shop.

## ✨ Key Features
* **AI-Powered Document Extraction:** Utilizes Google's `gemini-2.5-flash` model to accurately read messy PDFs and low-light smartphone photos, extracting the vehicle Year/Make/Model and itemized repair services.
* **Smart "Best Match" Pricing Engine:** Employs a weighted fuzzy-matching algorithm (via `difflib` and set intersections) to accurately map vague dealership terminology (e.g., "Front Wiper Blade Replacement") to the shop's internal pricing list (e.g., "wiper blades - front").
* **Live Admin Dashboard:** Shop owners can update their pricing dynamically by uploading a new `pricing.csv` directly through the Streamlit sidebar—no code changes required.
* **Intelligent API Caching:** Built-in session state memory caches multiple parsed documents, ensuring lightning-fast UI interactions and zero wasted API costs when users switch between files or fill out forms.
* **Instant PDF Generation:** Automatically generates a professional, downloadable PDF comparison report for the customer using `fpdf`.
* **Integrated Lead Capture:** Seamlessly captures customer contact information and triggers SMS alerts to the shop owner when a high-value lead is secured.

## 🛠️ Tech Stack
* **Frontend/Framework:** Streamlit
* **AI/LLM:** Google GenAI SDK (`google-genai`), Gemini 2.5 Flash
* **Data Processing:** Pandas, Python built-in `difflib`
* **File Handling:** PyPDF, Pillow (PIL)
* **Document Generation:** FPDF

## 🚀 Installation & Local Setup

### 1. Environment Setup (Python 3.12+)
It is highly recommended to run this application within an isolated Conda environment.
```bash
# Create and activate a new Conda environment
conda create -n second-opinion-app python=3.12 -y
conda activate second-opinion-app