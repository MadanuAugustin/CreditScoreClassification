
import sys
from src.CreditScoreClassification.logger_file.logger_obj import logger
from src.CreditScoreClassification.Exception.custom_exception import CustomException

try:
    x = 9
    logger.info('cannot divide by zero')
    print(x/0)

    
    

except Exception as e:
    raise CustomException(e, sys)

