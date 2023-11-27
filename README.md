*This is the business section. If you want the code section scroll past through this*
# Sentiment Analysis for Airlines on Twitter: Enhancing Customer Experience
# Business Objective
The main focus of this project is to do sentiment analysis on airline-related Twitter data. The main goal is to gather insightful information about how customers feel about various airlines as stated on social media. The objective is to create tools that can accurately anticipate a tweet's emotion based on its content by utilizing machine learning models.
### Project Focus
**Understanding Customer Sentiments:**
>  - studying tweets to understand the emotions and opinions people have about different airlines.
> - recognizing patterns and trends in consumer feedback to identify areas of contentment and discontent.

**Model Development for Sentiment Prediction:**
> - Developing machine learning algorithms capable of properly predicting tweet sentiment.
> - Improving the capacity to categorize tweets as positive, negative, or neutral based on the text content.

**Enhancing the Customer Experience:**
> - Providing airlines with actionable insights based on sentiment analysis results.
> - Providing airlines with the ability to address particular problems, improve services, and improve overall customer experience.

### Key Outcomes
**Key Results Data-Driven Decision-Making:**
> - Providing airlines with the ability to make informed decisions based on a thorough understanding of customer opinions.
> - Identifying opportunities for improvement in order to adjust services to consumer preferences.

**Proactive Problem Solving:**

> - Taking a proactive response to possible difficulties raised by clients on social media.
> - Resolving concerns and exhibiting attentiveness to reduce unfavorable feelings.

**Competitive Advantage:**
> - Giving airlines a competitive advantage by monitoring customer feedback and sentiment patterns.
> - Using sentiment analysis to distinguish services and keep a favorable brand image.

### Business Impact
This project's successful completion is expected to contribute to an overall improvement in consumer satisfaction in the airline industry. Airlines may effectively respond to consumer issues, improve service quality, and maintain a favorable online reputation by employing sentiment analysis models. Finally, this endeavor seeks to match business goals with customer expectations, encouraging long-term loyalty and market competitiveness.

# CODE
### Requirements
To run this project locally, you need to have Python installed. Additionally, install the required libraries using the following command:

```bash
pip install -r requirements.txt
```
### Project Structure
**Data Exploration:**
> - Checked the shape, missing values, data types, and duplicates in the dataset.
> - Explored the statistical summary of both numeric and text columns.

**Exploratory Data Analysis (EDA):**
> - Visualized the distribution of airlines and sentiments in the dataset.
> - Investigated the causes of negative feelings.

**Feature Engineering:**

> - Removed duplicate entries from the dataset.
> - Created a new dataframe with relevant columns for model building.
> - Encoded the target variable (airline_sentiment) using Label Encoder.

**Model Building:**
> - Split the data into training and testing sets.
> - Used TF-IDF vectorization to convert text data into numerical features.
> - Built a Logistic Regression model and evaluated its performance.
> - Utilized Stochastic Gradient Descent (SGD) Classifier for comparison and evaluation.

### Implementation
To run the code step-by-step, launch the Jupyter Notebook  *twitter_sentiment_analysis.ipynb*. If your dataset is in a separate directory, be sure to correct the file path.

### Dataset
The dataset can be found in kaggle using the following link:  https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment

