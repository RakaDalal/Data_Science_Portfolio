# Auto-completion-System

The project involves building an auto-completion system from sample conversations between customers and customer service representatives. The main aim is to speed up the customer service representatives' responses by suggesting sentence completions. I have used a trigram Katz’s back-off model for training. I have split the data into training set (80%) and development set (20%) and used the development set to tune the discount parameters for which I have used the minimize function from scipy.optimize. After training, I have saved the model on disk as a pickle file(model.pklz). During testing, the model is simply loaded from disk to run the predictions.

A more detailed step by step description of the project can be viewed [here](https://medium.com/@rakadalal/designing-an-auto-completion-system-from-scratch-3b769c821e05).

# Prerequisites:

All the packages required for this project to run successfully are listed in the requirements.txt file, which can be installed issuing the following command:

pip install –r requirements.txt

# How to run the system:

Download the entire folder and run,
1. python training.py <input_file>
2. python server.py

<input_file> is the json file containing sample conversations that you want the model to train on.

Sample tests:
1. curl http://localhost:5000/autocomplete?q=What+is+y

![Image of sample test 1](https://github.com/RakaDalal/Auto-completion-System/blob/master/test1.png)

2. curl http://localhost:5000/autocomplete?q=how+ca

![Image of sample test 2](https://github.com/RakaDalal/Auto-completion-System/blob/master/test2.png)

3. curl http://localhost:5000/autocomplete?q=when+w

![Image of sample test 3](https://github.com/RakaDalal/Auto-completion-System/blob/master/test3.png)

# Data Description:
I cannot upload the data unfortunately due to lack of permission from the owner of the data. However, I am providing a description of the data so that the data preprocessing part of the code can be manipulated accordingly for any chat dataset. 

There were two most important fields: text and IsFromCustomer(True if the text is written by customer and False if text is written by Agent). There were some other fields which were not important enough to be used in the project.

A sample conversation:

Hi, I have ordered a dress last week but I still haven't received it.

Hello, can you please tell me your order number?

My order number is XXXXXXX.

Thanks. What is your last name?
