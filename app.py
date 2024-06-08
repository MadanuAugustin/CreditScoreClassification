
import sys
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from src.CreditScoreClassification.logger_file.logger_obj import logger
from src.CreditScoreClassification.Exception.custom_exception import CustomException
from src.CreditScoreClassification.pipeline.prediction_pipeline import PredictionPipeline



# initializing the flask app

app = Flask(__name__)


# route to display the home page

@app.route('/predict', methods = ['POST', 'GET'])
def predict_datapoint():

    if request.method == 'GET':
        return render_template('index.html')
    

    else : 
        try:
            Annual_Income = request.form.get('Annual_Income')
            Monthly_Inhand_Salary = request.form.get('Monthly_Inhand_Salary')
            Num_Bank_Accounts = request.form.get('Num_Bank_Accounts')
            Num_Credit_Card = request.form.get('Num_Credit_Card')
            Interes_Rate = request.form.get('Interes_Rate')
            Num_of_Loan = request.form.get('Num_of_Loan')
            Delay_from_due_date = request.form.get('Delay_from_due_date')
            Num_of_Delayed_Payment = request.form.get('Num_of_Delayed_Payment')
            Outstanding_Debt = request.form.get('Outstanding_Debt')
            Crdit_History_Age = request.form.get('Crdit_History_Age')
            Monthly_balance = request.form.get('Monthly_balance')
            Credit_Mix = request.form.get('Credit_Mix')


            data = [Annual_Income, Monthly_Inhand_Salary, Num_Bank_Accounts, Num_Credit_Card, Interes_Rate,
                    Num_of_Loan, Delay_from_due_date, Num_of_Delayed_Payment, Outstanding_Debt,
                    Crdit_History_Age, Monthly_balance, Credit_Mix]
            
            logger.info(f'-----------Feteched data successfully from the user--------------')
            

            data = np.array(data).reshape(1, 12)

            data = pd.DataFrame(data)

            print(data)

            obj = PredictionPipeline()

            results = obj.predictDatapoint(data)

            logger.info(f'-----------Below is the final result {results}------------------')

            print(results)

            return render_template('index.html', results = str(results))


        except Exception as e:
            raise CustomException(e, sys)
        



if __name__ == '__main__':
    app.run(host = "0.0.0.0", debug = True) ## http://127.0.0.1:5000

