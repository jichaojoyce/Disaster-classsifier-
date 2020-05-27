# Disaster Response Pipeline Project

![Intro Pic](intro.png)

## Table of Contents
1. [Description](#description)
2. [Getting Started](#getting_started)
	1. [Libraries](#library)
	2. [Installing](#installing)
	3. [Instruction](#executing)
	4. [Data distribution](#material)
3. [Authors](#authors)
4. [License](#license)
5. [Acknowledgement](#acknowledgement)
6. [Example](#screenshots)

<a name="descripton"></a>
## Description

This Project is part of Data Science Nanodegree Program by Udacity in collaboration with Figure Eight.
The initial dataset contains disaster related categories and messages. 
This project is aim to classify the disaster related messages to different group. Writen in python.

The Project contains three folders:

1. 'data' folder contains the excel data and ETL Pipeline. The ETL pipeline is to extract data from source, clean data and save them in a proper databse structure
2. 'models' folder contains Machine Learning Pipeline. The ML pipeline used GridSearchCV to train the model to category messages.
3. 'app' folder is to show model results in a html web.

<a name="getting_started"></a>
## Getting Started

<a name="dependencies"></a>
### Libraries
* Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn
* Natural Language Process Libraries: NLTK
* SQLlite Database Libraqries: SQLalchemy
* Web App and Data Visualization: Flask, Plotly

<a name="installing"></a>
### Installing
Clone this GIT repository:
```
https://github.com/jichaojoyce/Disaster-classsifier-.git
```
<a name="Instruction"></a>
### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disease.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/disease.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Data distribution:
![dis Pic](over.png)
<a name="authors"></a>
## Authors

* [Chao Ji](https://github.com/jichaojoyce)

<a name="license"></a>
## License
Feel free to use it!
<a name="acknowledgement"></a>
## Acknowledgements

Credicts give to [Udacity](https://www.udacity.com/).

<a name="screenshots"></a>
## Example

1. You can type a sentence for classify. Here I typed 'we are more than 50 people sleeping on the street. Please help us find food, tent.'

![Sample Input](ex1.jpeg)

2. After clicking **Classify Message**, you can see the categories which the message belongs to highlighted in green

![Sample Output](ex2.jpeg)

![Main Page](ex3.jpeg)

