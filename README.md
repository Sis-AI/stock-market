# Correlating BBC News Sentiments With S&P 500 Stock Movements Using Machine Learning

**Project: Predicting Stock Market Trends Using Integrated News Sentiment and Machine Learning**

*Motivation & Overview:*  
I set out to investigate whether daily BBC news headlines could provide meaningful signals to predict changes in the S&P 500 closing prices. The goal was to build an integrated system that combined traditional financial data analysis with unsupervised sentiment extraction from news. Ultimately, I developed multiple models—ranging from classic classifiers (Logistic Regression, Random Forest, XGBoost) to a deep-learning LSTM time-series model—and an unsupervised sentiment engine that leverages both pre-trained language models (USE, Sentence-BERT and VADER) and a custom deep learning network to extract and aggregate daily sentiment labels.

*Project Workflow & Technical Details:*

1. **Market Data Analysis & Financial Models:**  
   - **Data Collection & EDA:**  
     I gathered historical S&P 500 data using yfinance and performed extensive EDA, including plotting the closing prices, examining correlation matrices, and decomposing the time series to reveal trends, seasonality, and residuals.  
   - **Feature Engineering & Labeling:**  
     I computed daily price changes and created a target variable (`shift_label`) with three classes (–1 for a downward move, 1 for upward, and 0 for negligible change). For parts of the analysis, I remapped these to binary labels (0 and 1) to facilitate certain model comparisons.  
   - **Classic Models:**  
     I built baseline models such as Logistic Regression (with L1 regularization), Random Forest, and XGBoost (tuned via grid search) to predict stock movement based solely on scaled financial features.

2. **Unsupervised Sentiment Analysis on BBC News Headlines:**  
   - **Preprocessing:**  
     The BBC news data was cleaned and preprocessed (tokenization, lemmatization, stop-word removal). I then tokenized and padded the cleaned headlines (with most sequences padded to a length of 13) to create inputs for deep-learning models.  
   - **Deep Learning Sentiment Model:**  
     I built an unsupervised LLM for sentiment analysis with the architecture:  
       - **Input & Embedding:** Converts padded sequences into dense representations.  
       - **Bidirectional LSTM Layers:** Two layers capture forward and backward contextual information.  
       - **Dense + Dropout Layers:** A dense layer with ReLU activation and dropout helps learn complex features while preventing overfitting.  
       - **Output Layer:** A Dense layer with tanh activation outputs a continuous sentiment score in the range [–1, 1].  
     After training (using self-supervised reconstruction targets), I apply thresholding (e.g., scaled outputs ≤0.5 as –1, >0.5 as 1) to obtain binary sentiment labels.
     
   - **Clustering & Alternative Approaches:**  
     To provide additional insights, I also generated embeddings using Universal Sentence Encoder (USE) and Sentence-BERT (all-MiniLM-L6-v2) and applied K-Means clustering (k=2) to derive alternative unsupervised sentiment labels.  
     
   - **Aggregation:**  
     Given that each day may contain numerous news articles, I grouped the sentiment labels by publication date—using aggregation methods (e.g., sum or mode) based on the Central Limit Theorem—to obtain a single daily sentiment score for each approach (from the deep model, USE, and BERT).

3. **Integration & Cross-Model Correlation:**  
   - **Merging Datasets:**  
     I merged the daily aggregated sentiment scores with the stock data (aligned by date).  
   - **Enhanced Prediction Models:**  
     I then incorporated the daily sentiment labels as additional features into the classic prediction models and also into the LSTM-based time-series forecasting model (using a sliding-window approach with 60-day sequences).  
   - **Evaluation:**  
     The performance of each model was rigorously evaluated using accuracy, F1-score, and confusion matrices. Results indicated that models incorporating sentiment—particularly those using daily aggregated unsupervised labels from the deep NLP model and USE clusters—showed improved correlation with the S&P 500 movements compared to using financial data alone.

*Outcome & Impact:*  
This multi-model system uncovered subtle correlations between BBC news sentiment and daily S&P 500 movements. In particular, the integration of daily aggregated sentiment labels into both classification and LSTM time-series models improved prediction accuracy, offering actionable insights for investment strategies. Overall, the project demonstrated that blending traditional financial analysis with advanced NLP techniques can yield significant improvements in market prediction performance.
