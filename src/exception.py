import sys
from src.logger import logging #import logging from src.logger python file

# error message function. when an execption gets raised, push custom message. error_detail is extracted from sys.
def error_message_detail(error,error_detail:sys):
    
    _,_,exc_tb=error_detail.exc_info() #exc_info() returns a tuple with class, object and traceback. traceback provides the exception details. line etc.
    file_name=exc_tb.tb_frame.f_code.co_filename
    
    error_message="Error occured in python script name [{0}] line number [{1}] error message[{2}]".format(
     file_name,
     exc_tb.tb_lineno, #line number from the traceback
     str(error) #string representation of the exception object
     )

    return error_message

    
# class CustomException(Exception):: Defines a custom exception class named CustomException that inherits from the built-in Exception class. 
# This allows you to create custom exceptions that behave similarly to standard Python exceptions.

class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):
        
        super().__init__(error_message) # Calls the constructor of the parent class (Exception) to initialize the base exception with the provided error_message
        self.error_message=error_message_detail(error_message,error_detail=error_detail)
    
    def __str__(self): #convert the exception object to a string (e.g., by printing it). It simply returns the self.error_message, which provides the detailed error information.
        return self.error_message

#