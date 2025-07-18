# NLP: IMDB Review Analysis

This project analyzes 620 IMDB movie reviews to identify key terms and patterns using Natural Language Processing (NLP). It demonstrates skills in text preprocessing, feature extraction, and visualization, applicable to workforce analytics tasks like analyzing employee feedback or survey responses.

## Files
- `Group_121_IMDB_(1).ipynb`: Jupyter Notebook with the complete NLP analysis, including preprocessing, TF-IDF vectorization, bigram modeling, and visualizations.
- `data/IMDB (1).csv`: Dataset containing 620 IMDB reviews (or [Google Drive link](#) if hosted externally).

## Project Overview
The notebook performs the following tasks:
- **Text Preprocessing**: Removes HTML tags, punctuation, and stop words; applies lemmatization for standardized text.
- **TF-IDF Analysis**: Extracts important words using Term Frequency-Inverse Document Frequency.
- **Bigram Modeling**: Analyzes word pairs to understand common phrases and compute sentence probabilities.
- **Visualizations**:
  - Word cloud of top TF-IDF words.
  - Bar plot of top 30 TF-IDF words and top 10 bigrams.
  - Interactive PCA scatter plot of top words using Plotly.
  - Sentiment distribution plot (if sentiment labels are available).

## Key Insights
- **Top Words**: Identifies dominant terms (e.g., "movie," "great," "story") driving review content.
- **Bigrams**: Highlights common phrases (e.g., "great movie," "well done") in reviews.
- **Applications**: Techniques can be applied to workforce management, such as extracting trends from employee survey responses to improve engagement.

## How to Run
1. **Environment**: Open the notebook in Google Colab or Jupyter Notebook.
2. **Dataset**:
   - If `IMDB (1).csv` is in `data/`, ensure it’s accessible.
   - If hosted externally, download it using:
     ```python
     !gdown 'your_google_drive_link' -O IMDB.csv
     ```
3. **Install Dependencies**:
   ```bash
   pip install pandas numpy scikit-learn nltk matplotlib seaborn wordcloud plotly adjustText
   ```
4. **NLTK Resources**:
   ```python
   import nltk
   nltk.download('punkt_tab')
   nltk.download('stopwords')
   nltk.download('wordnet')
   nltk.download('omw-1.4')
   ```
5. Run all cells sequentially.

## Dependencies
See `../requirements.txt` for a complete list. Key packages:
- Python 3.8+
- pandas, numpy, scikit-learn, nltk
- matplotlib, seaborn, wordcloud, plotly, adjustText

## Notes
- The notebook assumes `IMDB (1).csv` has columns `ID` and `review`. If it includes a `sentiment` column, additional visualizations (e.g., sentiment distribution) are generated.
- For large datasets, host `IMDB (1).csv` on Google Drive and update the `gdown` link.
- Visualizations are optimized for clarity and interactivity to appeal to technical and non-technical audiences.