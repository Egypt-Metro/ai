# Egypt Metro AI Models

## Overview

The **Egypt Metro AI Models** repository contains machine learning models used for **crowd prediction**, **route optimization**, and **chatbot support**. These models are powered by **TensorFlow**, **Keras**, and **Natural Language Processing (NLP)** techniques. The AI component helps to provide real-time insights into crowd density and suggests optimized routes for users based on historical data and AI predictions.

## Features

- **Crowd Prediction**: Predicts train congestion based on time of day and historical data.
- **Route Optimization**: Suggests the most efficient travel routes based on user preferences and live data.
- **Chatbot**: NLP-based chatbot to assist users with FAQs and navigation.
- **Anomaly Detection**: Detects unusual patterns in fault reporting data and flags them for further analysis.

## Installation

### Prerequisites
- Python 3.9+
- TensorFlow 2.x
- Scikit-learn
- Pandas
- NumPy

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/egypt-metro/egypt-metro-ai.git
2. Navigate to the project directory:

cd egypt-metro-ai
Create and activate a virtual environment:

bash
Copy code
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Running the Models
Crowd Prediction Model: To train the crowd prediction model, run:

bash
Copy code
python train_crowd_model.py
Route Optimization Model: To train the route optimization model, run:

bash
Copy code
python train_route_model.py
Chatbot: To train and run the chatbot model, use:

bash
Copy code
python train_chatbot.py
python run_chatbot.py
Example Usage
Crowd Prediction: To make a prediction on crowd density:

bash
Copy code
python predict_crowd.py --time "08:00" --train_line "Line 1"
Route Optimization: To suggest optimized routes:

bash
Copy code
python predict_route.py --start_station "Station A" --end_station "Station B"
