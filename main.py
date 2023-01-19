from pprint import pprint
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
from transformers import pipeline
import os
from datetime import datetime as dd

class CSSA: 

    def __init__(self, data):
        self.data = data

    @staticmethod
    def load_model(model_name):
        # load Customer Segmentation Model
        with open(model_name , 'rb') as f:
            model = pickle.load(f)
        return model
    
    @staticmethod
    def load_sentiment_model():
        sentiment_pipeline = pipeline("sentiment-analysis")
        return sentiment_pipeline

    def customer_segmentation(self, data, segmentation_model):
        # Predicting the output for custmer segmentation
        pred = segmentation_model.predict(data[:, :2])
        print(f'The Prediction for Above input is cluster no. {pred[0]}')
        return pred

    def sentiment_analysis(self, Review, sentiment_pipeline):
        result = sentiment_pipeline(Review)
        return result

    @staticmethod
    def looping(l):
        print('')
        for i, j in enumerate(l):
            print(f"{i+1}. {j}")

    def product(self, cluster):
        print('Potential Buyer for Below Product')
        if cluster == 0:
            # print('Belongs to High average annual income, low spending')
            l = ['Milk', 'Vegetable', 'Grocery', 'Wheat', 'Rice']
            CSSA.looping(l)
        elif cluster == 1:
            # print('Belongs to Low to mid average income, average spending capacity.')
            l = ['Cookies', 'Sandwich', 'Pizza', 'Burger', 'Coffee']
            CSSA.looping(l)
        elif cluster == 2:
            # print('Belongs to Low average income, high spending score.')
            l = ['Bicycle', 'TV', 'Fridge', 'Mobile', 'Bike']
            CSSA.looping(l)
        elif cluster == 3:
            # print('Belongs to High average income, high spending score.')
            l = ['Oddi', 'Apartment', 'Ferrari', 'Thar', 'Fortuner']
            CSSA.looping(l)
        else: 
            print('')
            print('No match found...')
        return l
    
    def buy_record(self, things):
        with open('record.txt', 'a') as f:
            f.write(f'Buy {str(things).lower()} on {dd.now()}\n')

    @staticmethod
    def extract_features(word_list):
        return dict([(word, True) for word in word_list])

    def start(self):

        # Loading Data
        df=pd.read_csv(self.data)
        col_names = ['Annual Income (k$)', 'Age', 'Spending Score (1-100)']
        features = df[col_names]
        # Training StandardScaler on test data
        scaler = StandardScaler().fit(features.values)

        # Loading Sentiment analysis pipeline
        sentiment_pipeline = CSSA.load_sentiment_model()
        
        # Loading the model of customer segmentation
        segmentation_model = CSSA.load_model('model_pkl')
        return scaler, sentiment_pipeline, segmentation_model

def Main_App(Income, Age, Spend, scaler, sentiment_pipeline, segmentation_model, C1):
    
    # Converting the input into an array
    x_array = np.array([[Income, Age, Spend]])

    # Normalizing the input using Standard Scaler
    normalized_arr = scaler.transform(x_array)

    # Predicting the output for custmer segmentation
    pred = C1.customer_segmentation(normalized_arr, segmentation_model)

    # Finding the Product that customer buy
    buy_list = C1.product(pred[0])
    
    # Return the list of buying products
    return buy_list

def Main_App2(sentiment_pipeline, C1, review):

    # Find the Sentiment of Review
    probdist = C1.sentiment_analysis([review], sentiment_pipeline)

    # Predicting the result of positive, negative or neutral review.
    print("Predicted Sentiments: ", probdist[0]['label'])                   # Sentiment
    print("Probability: ", round(probdist[0]['score'], 2))                  # Probability of Sentiment
    
    return probdist[0]['label'], round(probdist[0]['score'], 2)

if __name__ == '__main__':

    # Loading Data and Standardizing the data
    C1 = CSSA('Test.csv')
    scaler, sentiment_pipeline, segmentation_model = C1.start()

    # While loop for continuous input and output
    while True:

        # Clearning Screen before using    
        os.system('cls')

        # Getting the Input
        Income = int(input('\nEnter Monthly Income (in Thousand): '))
        Income = Income / 1000
        Age = int(input('Enter Age (1-100): '))
        a = input('Enter Spend Score: ') # Spend Score
        
        # Converting the input into an array
        x_array = np.array([[Income, Age, a]])

        # Normalizing the input using Standard Scaler
        normalized_arr = scaler.transform(x_array)

        # Predicting the output for custmer segmentation
        pred = C1.customer_segmentation(normalized_arr)

        # Finding the Product that customer buy
        buy = C1.product(pred[0])
        C1.buy_record(buy)

        # Enter Review for product
        review = input('\nEnter the Review: ')
        probdist = C1.sentiment_analysis([review])

        # Predicting the result of positive, negative or neutral review.
        print("Predicted Sentiments: ", probdist[0]['label'])                   # Sentiment
        print("Probability: ", round(probdist[0]['score'], 2))                  # Probability of Sentiment
        x = input('')
        if x.lower() == 'q':
            print('***** THANK YOU *****')
            exit()