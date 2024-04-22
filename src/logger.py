import logging #  Imports the logging module, which provides functionalities for logging messages.
import os # Imports the os module, which provides functions for interacting with the operating system.
from datetime import datetime

LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log" # create unique file name with the date & time (month_day_year_hour_minute_second.log)
logs_path=os.path.join(os.getcwd(),"logs",LOG_FILE) # create string variable logs_path with os.getcdw() gives the current working directory + "logs" + log file name
os.makedirs(logs_path,exist_ok=True) # create directory specified by logs_path and append the file even though directory exist

LOG_FILE_PATH=os.path.join(logs_path,LOG_FILE)

# Configures the logging module by setting the following arguments:
logging.basicConfig(
    filename=LOG_FILE_PATH, # set filename for the log file
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s", # %(asctime)s (timestamp), %(lineno)d (line number), %(name)s (logger name), %(levelname)s (log level), and %(message)s (message).
    level=logging.INFO, # This means that only messages with a level of INFO or higher (WARNING, ERROR, CRITICAL) will be logged to the file.


)

# To initiate and test logger.py: python src/logger.py
"""
    if __name__ == "__main__":
    logging.info("Logging has started")    
"""
