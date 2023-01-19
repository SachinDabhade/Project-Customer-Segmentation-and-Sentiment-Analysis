from flask import Flask, render_template, request, Response
from flask import abort, redirect, url_for
from main import Main_App, Main_App2
from main import CSSA

# Loading Data and Standardizing the data
C1 = CSSA('Test.csv')
scaler, sentiment_pipeline, segmentation_model = C1.start()

app = Flask(__name__)

@app.route("/", methods=['POST', 'GET'])
def home():
    if request.method=='POST':
        income = request.form.get('income')
        age = request.form.get('age')
        spend = request.form.get('name')
        global buy_list
        buy_list = Main_App(income, age, spend, scaler, sentiment_pipeline, segmentation_model, C1)
        # return render_template('stage1.html', buy_list=buy_list)
        # # return buy_product_web(buy_list)
        return redirect(url_for('buy_product_web'))
    return render_template('index.html')

@app.route('/buy', methods=['POST', 'GET'])
def buy_product_web():
    if request.method == 'POST':
        product = request.form.get('product')
        review = request.form.get('review')
        C1.buy_record(product)
        sentiment, probability = Main_App2(sentiment_pipeline, C1, review)
        return render_template('stage2.html', sentiment=sentiment, probability=probability)
    return render_template('stage1.html', buy_list=buy_list)

if __name__=='__main__':
   app.run()