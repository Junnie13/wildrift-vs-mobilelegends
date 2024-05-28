# 🧠 NLP Data Science Project: Analyzing Google Play Store Reviews 📊

## 📝 Overview
This repository contains code and documentation for an NLP (Natural Language Processing) project that analyzes user reviews from the Google Play Store for two popular mobile games: **League of Legends: Wild Rift** and **Mobile Legends: Bang Bang**.

## 🏗️ Project Structure
- `data/`: Contains the raw review data downloaded from the Google Play Store.
- `notebooks/`: Jupyter notebooks for data preprocessing, exploratory data analysis, and modeling.
- `src/`: Python scripts for data cleaning, feature extraction, and model training.
- `results/`: Stores intermediate results and final model outputs.
- `README.md`: This file, providing an overview of the project.

## 📊 Data Collection
1. **Google Play Store Reviews**:
   - We scraped user reviews for both games using the Google Play Store API.
   - Each review includes the user's rating, text, and timestamp.

## 🛠️ Data Preprocessing
1. **Cleaning and Tokenization**:
   - Removed special characters, emojis, and HTML tags.
   - Tokenized the review text into individual words.

2. **Feature Extraction**:
   - Extracted features such as sentiment scores, word frequencies, and review length.
   - Created a bag-of-words representation for modeling.

## 🔍 Exploratory Data Analysis (EDA)
1. **Distribution of Ratings**:
   - Visualized the distribution of ratings (1 to 5 stars) for both games.
   - Explored any patterns or anomalies.

2. **Word Clouds**:
   - Generated word clouds to visualize frequently occurring words in positive and negative reviews.

## 💡 Sentiment Analysis
1. **Sentiment Classification**:
   - Trained a Naive Bayes classifier to predict sentiment (positive/negative) based on review text.
   - Evaluated model performance using cross-validation.

## 📈 Topic Modeling (LLMS)
1. **Latent Dirichlet Allocation (LDA)**:
   - Applied LDA to discover latent topics within the reviews.
   - Identified key terms associated with each topic (e.g., gameplay, graphics, bugs).

## 🌟 Rating Prediction
1. **Regression Model**:
   - Built a regression model to predict user ratings based on review features.
   - Features include sentiment scores, review length, and topic proportions.

## 📋 Results
1. **Sentiment Distribution**:
   - Compared the proportion of positive and negative reviews for each game.
   - Identified areas for improvement based on negative sentiment.

2. **Top Keywords**:
   - Extracted keywords associated with positive and negative sentiments.
   - Example: "exciting," "smooth controls" (positive) vs. "bugs," "lag" (negative).

3. **Topic Insights**:
   - Discovered topics related to gameplay, graphics, and user experience.
   - Explored how these topics impact user ratings.

## 🏁 Conclusion
- **League of Legends: Wild Rift**:
  - Generally positive sentiment.
  - Strengths: Exciting gameplay, strategy, and team dynamics.
  - Areas for improvement: Address bugs and optimize performance.

- **Mobile Legends: Bang Bang**:
  - Mixed sentiment.
  - Strengths: Vibrant graphics, memorable characters.
  - Areas for improvement: Balance gameplay and enhance user experience.

## 🔜 Next Steps
- Investigate correlations between reviews and in-game events (e.g., updates, events, patches).
- Explore more advanced NLP techniques for sentiment analysis and topic modeling.
