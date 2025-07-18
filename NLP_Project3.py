{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# IMDB Movie Review NLP Analysis\n",
    "\n",
    "**NLP Assignment Solution**\n",
    "\n",
    "> _A data science project performing EDA, preprocessing, feature engineering (TF-IDF), similarity analysis, and more on IMDB movie reviews._\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Table of Contents\n",
    "- [Project Overview](#Project-Overview)\n",
    "- [Setup & Installation](#Setup--Installation)\n",
    "- [Data Loading](#Data-Loading)\n",
    "- [Exploratory Data Analysis (EDA)](#Exploratory-Data-Analysis-EDA)\n",
    "- [Preprocessing](#Preprocessing)\n",
    "- [Feature Extraction & Similarity](#Feature-Extraction--Similarity)\n",
    "- [Dimensionality Reduction](#Dimensionality-Reduction)\n",
    "- [Results & Discussion](#Results--Discussion)\n",
    "- [References](#References)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Project Overview\n",
    "This project analyzes **IMDB movie reviews** using various NLP techniques, including:\n",
    "- Data cleansing and preprocessing\n",
    "- Feature extraction via TF-IDF\n",
    "- Cosine similarity computation\n",
    "- Dimensionality reduction (e.g., PCA)\n",
    "- Data visualization\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a id='Setup--Installation'></a>\n",
    "\n",
    "## Setup & Installation\n",
    "\n",
    "**Install requirements:**\n",
    "```
    "# !pip install adjustText --quiet\n",
    "```\n",
    "\n",
    "**Import libraries:**\n",
    "```
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.metrics.pairwise import cosine_similarity\n",
    "from sklearn.decomposition import PCA\n",
    "import matplotlib.pyplot as plt\n",
    "import nltk\n",
    "from nltk.tokenize import word_tokenize\n",
    "from nltk.corpus import stopwords\n",
    "from nltk.stem import PorterStemmer, WordNetLemmatizer\n",
    "import string\n",
    "from adjustText import adjust_text\n",
    "from collections import defaultdict, Counter\n",
    "import math\n",
    "import warnings\n",
    "warnings.filterwarnings(\"ignore\")\n",
    "import textwrap\n",
    "```\n",
    "\n",
    "**NLTK resource download:**\n",
    "```
    "nltk.download('punkt')\n",
    "nltk.download('stopwords')\n",
    "nltk.download('wordnet')\n",
    "nltk.download('omw-1.4')\n",
    "```\n",
    "\n",
    "**Pandas display settings:**\n",
    "```
    "pd.set_option('display.max_rows', 100)\n",
    "pd.set_option('display.max_columns', 50)\n",
    "pd.set_option('display.max_colwidth', 200)\n",
    "```\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a id='Data-Loading'></a>\n",
    "\n",
    "## Data Loading\n",
    "\n",
    "```
    "# For Google Colab use:\n",
    "# from google.colab import files\n",
    "# uploaded = files.upload()\n",
    "\n",
    "df = pd.read_csv('/content/IMDB.csv')\n",
    "print(df.shape)\n",
    "df.head()\n",
    "```\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a id='Exploratory-Data-Analysis-EDA'></a>\n",
    "\n",
    "## Exploratory Data Analysis (EDA)\n",
    "\n",
    "- **Check missing values:**\n",
    "```
    "df.isnull().sum()\n",
    "```\n",
    "- **Distribution of labels:**\n",
    "```
    "df['sentiment'].value_counts().plot(kind='bar', title='Sentiment Distribution')\n",
    "plt.show()\n",
    "```\n",
    "- **Sample reviews:**\n",
    "```
    "df.sample(5)\n",
    "```\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a id='Preprocessing'></a>\n",
    "\n",
    "## Preprocessing\n",
    "\n",
    "**Define text cleaning and normalization:**\n",
    "```
    "def clean_text(text):\n",
    "    # Lowercase\n",
    "    text = text.lower()\n",
    "    # Remove punctuation\n",
    "    text = text.translate(str.maketrans('', '', string.punctuation))\n",
    "    # Tokenize\n",
    "    tokens = word_tokenize(text)\n",
    "    # Remove stopwords\n",
    "    stop_words = set(stopwords.words('english'))\n",
    "    tokens = [w for w in tokens if w not in stop_words]\n",
    "    return ' '.join(tokens)\n",
    "\n",
    "df['clean_review'] = df['review'].apply(clean_text)\n",
    "df[['review', 'clean_review']].head()\n",
    "```\n",
    "\n",
    "**Optional: Stemming and Lemmatization:**\n",
    "```
    "stemmer = PorterStemmer()\n",
    "lemmatizer = WordNetLemmatizer()\n",
    "\n",
    "def stem_tokens(text):\n",
    "    return ' '.join([stemmer.stem(word) for word in text.split()])\n",
    "\n",
    "def lemmatize_tokens(text):\n",
    "    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])\n",
    "\n",
    "df['stemmed_review'] = df['clean_review'].apply(stem_tokens)\n",
    "df['lemmatized_review'] = df['clean_review'].apply(lemmatize_tokens)\n",
    "```\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a id='Feature-Extraction--Similarity'></a>\n",
    "\n",
    "## Feature Extraction & Similarity\n",
    "\n",
    "**TF-IDF Vectorization:**\n",
    "```
    "tfidf = TfidfVectorizer(max_features=5000)\n",
    "X = tfidf.fit_transform(df['lemmatized_review'])\n",
    "print(X.shape)\n",
    "```\n",
    "\n",
    "**Cosine Similarity (between first 5 reviews):**\n",
    "```
    "sample_cosine_sim = cosine_similarity(X[:5], X[:5])\n",
    "sample_cosine_sim\n",
    "```\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a id='Dimensionality-Reduction'></a>\n",
    "\n",
    "## Dimensionality Reduction\n",
    "\n",
    "**PCA for Visualization:**\n",
    "```
    "pca = PCA(n_components=2)\n",
    "X_pca = pca.fit_transform(X.toarray())\n",
    "\n",
    "plt.figure(figsize=(8,5))\n",
    "colors = {'positive':'green','negative':'red'}\n",
    "for sentiment in df['sentiment'].unique():\n",
    "    idx = df['sentiment']==sentiment\n",
    "    plt.scatter(X_pca[idx,0], X_pca[idx,1],\n",
    "                label=sentiment, alpha=0.5, c=colors[sentiment])\n",
    "plt.title('PCA of Reviews TF-IDF Embedding')\n",
    "plt.xlabel('PC1')\n",
    "plt.ylabel('PC2')\n",
    "plt.legend()\n",
    "plt.show()\n",
    "```\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a id='Results--Discussion'></a>\n",
    "\n",
    "## Results & Discussion\n",
    "- **Observations from EDA:** …\n",
    "- **Preprocessing impact:** …\n",
    "- **TF-IDF/Vectors inspection:** …\n",
    "- **Visual separation of sentiments:** …\n",
    "\n",
    "> _Add your commentary and results here._\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a id='References'></a>\n",
    "\n",
    "## References\n",
    "- [IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)\n",
    "- [Scikit-learn Documentation](https://scikit-learn.org/)\n",
    "- [NLTK Documentation](https://www.nltk.org/)\n",
    "- [Text preprocessing tutorial](https://www.kaggle.com/code/alvations/basic-nlp-with-nltk)\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": ""
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
