<<<<<<< HEAD
# Sentiment analysis of financial news headlines

This GitHub repository provides code and resources for sentiment analysis of financial news. The goal of this project is to analyze financial news articles and classify them as positive, negative, or neutral.

## Approaches

The main approaches proposed to tackle the sentiment analysis task are:

- SVM with TD-IDF features.
- Naive-Bayes.
- Fine-tuning the transformer-based RoBERTa architecture.

The RoBERTa model was fine-tuned in two different ways:

- Only on the target dataset, the Financial Phrase Bank dataset.
- On both the Financial Phrase Bank and a dataset that contains around 124M tweets.

## How to Run

**Attention:** to execute the workflow you need to download the weights of the fine-tuned RoBERTa (check the section below 'Weights for RoBERTa' to see how).

To ensure you have the same development environment as the project, I have provided a **conda** environment file **environment.yml** in the root of the repository. To create the environment and activate it, run the following command:
<pre>
conda env create -f environment.yml
conda activate NLP
</pre>
<!-- **conda env create -f environment.yml**
**conda activate NLP** -->

To then execute the main workflow, run the following command:

<!-- **python main.py** -->
<pre>
python main.py
</pre>

## Repository Structure

The repository is structured as follows:

- `src`: This folder contains the source code for the project. The code is organized into several modules that handle different aspects of the sentiment analysis task.
- `models`: This folder contains pre-trained models and fine-tuned models.
- `utils`: This folder contains utility functions and scripts used for preprocessing, feature engineering, and evaluation of models.
- `project_logs`: This folder contains logs generated by the project during training and testing phases.


## Weights for RoBERTa

The weights for the fine-tuned RoBERTa model are too large to be uploaded to GitHub. Therefore, we have provided a Google Drive link to download a zip file containing the **`WEIGHTS_ROBERTA`** folder. Please download the zip file from the following link, extract its contents, and place the resulting **`WEIGHTS_ROBERTA`** folder in the same directory as **`main.py`**:

**https://drive.google.com/file/d/1EUNRutkLIc0-cp9Bvz7nb3Xb4A9a1h2M/view?usp=share_link**

