# 👥 Customer Segmentation with PySpark & Streamlit

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://https://customersegmentation-pyspark.streamlit.app)

An end-to-end data science project that performs customer segmentation on a large retail dataset. This project leverages **PySpark** for scalable data processing and feature engineering, **Gaussian Mixture Models (GMM)** for clustering, and **Streamlit** for an interactive web-based dashboard to visualize the results.

---

## ## Project Overview

The goal of this project is to analyze customer purchasing behavior and group them into distinct segments. By understanding these segments, a business can create targeted marketing strategies, improve customer retention, and personalize user experience.

The analysis is based on the popular **RFM (Recency, Frequency, Monetary)** model, enhanced with other behavioral features. The entire data pipeline is built to be scalable, making it suitable for datasets that are too large to fit into memory on a single machine.

### ## Key Features

* **Scalable Data Processing with PySpark**: The entire data cleaning, transformation, and feature engineering (RFM calculation, behavioral metrics) pipeline is built using PySpark to handle large datasets efficiently.
* **Advanced Clustering**: Uses Gaussian Mixture Models (GMM) to create nuanced, probability-based customer segments.
* **Model Optimization**: Automatically determines the optimal number of clusters using the Silhouette Score.
* **Interactive Dashboard**: A user-friendly web app built with Streamlit to visualize and explore the customer segments, their key characteristics, and a sample of the underlying data.
* **Customer Personas**: Segments are assigned descriptive personas (e.g., "🏆 Champions", " dormant Customers") for easy business interpretation.

---

## ## Tech Stack

* **Data Processing**: Apache Spark (via **PySpark**)
* **Machine Learning**: PySpark MLlib
* **Dashboard & Visualization**: Streamlit, Plotly, Matplotlib
* **Core Libraries**: Pandas, NumPy

---

## ## Architecture

The project is divided into two main components:

1.  **Batch Processing Pipeline (`segmentation.py`)**:
    * Loads the raw retail data into a Spark DataFrame.
    * Performs extensive data cleaning and preprocessing.
    * Engineers RFM (Recency, Frequency, Monetary) and other behavioral features.
    * Uses a PySpark ML Pipeline to scale features and train a Gaussian Mixture Model.
    * Saves the final segmented customer data and visualization assets (plots) to disk.

2.  **Interactive Dashboard (`app.py`)**:
    * A lightweight Streamlit application that loads the pre-processed results.
    * Provides an interactive UI with tabs, KPI cards, and charts to explore each customer segment.
    * **Does not** run the heavy Spark computations, ensuring the dashboard is fast and responsive.

---

## ## Setup and Installation

To run this project locally, please follow these steps.

### ### Prerequisites

* Python 3.8+
* Java 8 or later (required for PySpark)
* A package manager like `pip`

### ### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
    cd YOUR_REPO_NAME
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: If you don't have a `requirements.txt` file, you can create one with `pip freeze > requirements.txt` after installing the necessary packages like `pyspark`, `streamlit`, `pandas`, `plotly`)*

---

## ## How to Use

Running the project is a two-step process.

### ### Step 1: Run the PySpark Processing Pipeline

First, you need to execute the main segmentation script. This will process the raw data and generate the output files needed by the Streamlit dashboard.

```bash
spark-submit segmentation.py
```
This script will:
1.  Read the input data from `data/`.
2.  Perform all cleaning and feature engineering steps.
3.  Train the GMM model.
4.  Save the segmented data and plots to the `output/` directory.

### ### Step 2: Launch the Streamlit Dashboard

Once the processing is complete, you can launch the interactive dashboard.

```bash
streamlit run app.py
```

Your web browser will automatically open a new tab with the dashboard, where you can explore the customer segments.
