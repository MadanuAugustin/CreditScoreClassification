




import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from src.CreditScoreClassification.logger_file.logger_obj import logger
from src.CreditScoreClassification.Exception.custom_exception import CustomException


class PredictionPipeline:
    def __init__(self):
        self.model = joblib.load(Path('artifacts//model_trainer//model.joblib'))
        self.preprocessorObj = joblib.load(Path('artifacts//data_transformation//preprocessor_obj.joblib'))


    # the below method takes the data from the user to predict

    def predictDatapoint(self, data):
        
        try:

            data_df = data.rename(columns = {0 : 'Annual_Income', 1 : 'Monthly_Inhand_Salary', 2 : 'Num_Bank_Accounts',
                                             3 : 'Num_Credit_Card', 4 : 'Interes_Rate', 5 : 'Num_of_Loan',
                                             6 : 'Delay_from_due_date', 7 : 'Num_of_Delayed_Payment',
                                             8 : 'Outstanding_Debt', 9 : 'Crdit_History_Age', 10 : 'Monthly_balance',
                                             11 : 'Credit_Mix'})
            
            print(data_df)

            user_numeric_cols = data_df.drop(columns = ['Credit_Mix'], axis = 1)

            user_categoric_cols = data_df[['Credit_Mix']]

            transformed_numeric_cols = self.preprocessorObj.transform(user_numeric_cols)

            transformed_categoric_cols = user_categoric_cols['Credit_Mix'].map({'Good': 1, 'Standard' : 2, 'Bad' : 0})

            transformed_user_input = pd.DataFrame(np.c_[transformed_numeric_cols, transformed_categoric_cols])

            logger.info(f'---------Below is the transformed user input----------------')

            print(transformed_user_input)


            prediction = self.model.predict(transformed_user_input)

            list_output  = []

            if prediction == [0.]:
                list_output.append('Good')
            elif prediction == [1.]:
                list_output.append('Poor')
            elif prediction == [2.0]:
                list_output.append('Standard')

            logger.info(f'-----------Below output is predicted by the model---------------')

            print(list_output)

            return list_output
        
        
        except Exception as e:
            raise CustomException(e, sys)

