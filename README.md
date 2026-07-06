# Google Maps Rating Prediction from Reviews

> A machine learning project to predict Google Maps ratings from review text and business metadata, comparing baseline, linear, and transformer-based (DistilBERT) models.

---

## About The Project

This project analyzes a dataset of Google Maps reviews for businesses in Vermont to build a model that accurately predicts user ratings. The primary goal is to understand the key drivers of ratings by leveraging natural language processing on review text and incorporating contextual business metadata. This work has applications in sentiment analysis, business intelligence, and enhancing recommendation systems.

The analysis begins with thorough data cleaning and exploratory data analysis (EDA) to uncover insights, followed by a systematic progression of model development, from simple baselines to a state-of-the-art DistilBERT model.

---

## Analysis and Visualization

A key part of this project was understanding the data through exploratory analysis.

*   **Rating Distribution:** The dataset is heavily skewed towards positive reviews, with over 60% of all ratings being 5-stars. This highlights the importance of using a metric like MSE that penalizes large errors on less frequent, lower ratings.
    ![Rating Distribution](rating_distribution.png)

*   **Text Length vs. Rating:** A clear trend emerged where lower ratings are associated with significantly longer review texts. This suggests that users tend to be more descriptive and provide more detail when sharing negative feedback.
    ![Text Length Analysis](text_length_analysis.png)

*   **Feature Correlation:** A correlation analysis was performed on engineered features. The business's historical average rating showed the strongest positive correlation (0.331) with a new rating, while review text length had the strongest negative correlation (-0.168).
    ![Correlation Heatmap](correlation_heatmap.png)

---

## Key Features Implemented

*   **Data Preprocessing & Cleaning:** Merged review and business metadata from raw JSON files, handled missing values, removed duplicates, and recalculated business-level statistics (`avg_rating`, `num_of_reviews`) to ensure data integrity.
*   **Feature Engineering:** Created new features for analysis and modeling, including `text_length`, `open_num_days` (from business hours), `price_numeric` (from price symbols), and a leak-free `avg_rating` for each business.
*   **Baseline Modeling:** Implemented Global Mean, User Mean, and Item Mean models to establish a performance floor and demonstrate the value of personalization.
*   **Linear Models with Text Features:**
    *   Utilized **TF-IDF Vectorization** (with unigrams and bigrams) to convert review text into a high-dimensional feature space of 50,000 features.
    *   Implemented and fine-tuned **Ridge Regression** and **LinearSVR** models, using a dedicated validation set to find optimal hyperparameters (`alpha` for Ridge, `C` and `epsilon` for SVR).
*   **Hybrid Modeling (Text + Metadata):**
    *   Combined sparse TF-IDF features with dense, scaled metadata features (`avg_rating`, `text_length`) to create a richer input for the linear models.
*   **Transformer-Based Deep Learning Model:**
    *   Fine-tuned a pre-trained **DistilBERT** model for the rating prediction task. The problem was framed as a 5-class classification, leveraging the power of transformers to understand the semantic context of review text.

---

## What We Achieved

*   **High-Performance Predictive Model:** The fine-tuned **DistilBERT model achieved the lowest Mean Squared Error (MSE) of 0.4632** on the test set, outperforming all other models.
*   **Significant Improvement Over Baselines:** The best model demonstrated a **57.6% relative improvement in MSE** compared to a simple global mean baseline, showcasing the strong predictive power of review text.
*   **Effective Feature Engineering:** Adding just two metadata features (`avg_rating` and `text_length`) to the linear models consistently provided a **~1.8% performance boost**, confirming that combining text with context is beneficial.
*   **Comprehensive Model Evaluation:** Systematically evaluated eight different models, from simple baselines to a complex transformer. The results clearly show the progression in performance as model complexity and feature richness increase.
    !Model Performance Comparison

---

## Tech Stack

| Data Science & ML          | Deep Learning           | Visualization          | Environment       |
| -------------------------- | ----------------------- | ---------------------- | ----------------- |
| Pandas, NumPy              | PyTorch                 | Matplotlib             | Jupyter Notebook  |
| Scikit-learn               | Transformers (Hugging Face) | Seaborn                |                   |
| (TfidfVectorizer, Ridge)   | (DistilBERT)            |                        |                   |
| (LinearSVR, StandardScaler)|                         |                        |                   |

---

## Getting Started

This project is contained within a Jupyter Notebook (`Assignment2.ipynb`).

### Prerequisites

*   Python 3.x
*   Jupyter Notebook or JupyterLab
*   Required Python libraries

### Installation

1.  Clone the repository:
    ```sh
    git clone https://github.com/your_username/your_repository.git
    ```
2.  Install the required packages:
    ```sh
    pip install pandas numpy scikit-learn matplotlib seaborn torch transformers
    ```

### Execution

1.  Download the dataset files (`review-Vermont_10.json.gz` and `meta-Vermont.json.gz`) from the [https://mcauleylab.ucsd.edu/public_datasets/gdrive/googlelocal] (UCSD Google Local Datasets page) and place them in the root directory of the project.
2.  Launch Jupyter and open `Assignment2.ipynb`.
3.  Run the cells sequentially to reproduce the analysis, model training, and evaluation.
