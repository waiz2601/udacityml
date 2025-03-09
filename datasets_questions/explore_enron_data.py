#!/usr/bin/python3

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import joblib
import os

file_path = "../final_project/final_project_dataset_modified.pkl"

print("Starting the script...")

# Check if the file exists
if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
else:
    print(f"Found file: {file_path}")

try:
    # Load the dataset
    print("Attempting to load the dataset...")
    enron_data = joblib.load(open(file_path, "rb"))
    print("Loaded dataset successfully.")
    
    # Print the number of people in the dataset
    print(f"Number of people in the dataset: {len(enron_data)}")
    
    # Print the keys of the dataset to explore its structure
    for key in enron_data.keys():
        print(len(enron_data[key]))
        break
    
    # Count the number of POIs
    cnt = 0
    for key in enron_data.keys():
        if enron_data[key]['poi'] == 1:
            cnt += 1
    print("Number of POIs:", cnt)
    
    # Print the keys (feature names) for the first person
    first_person = list(enron_data.keys())[0]
    print(f"\nFeatures for {first_person}:")
    for key in enron_data.keys():
        print(key)
    quantified_salary = 0
    known_email_address = 0
    for key in enron_data.keys():
        if enron_data[key]['salary']!='NaN':
            quantified_salary+=1
        if(enron_data[key]['email_address']!='NaN'):
            known_email_address+=1
    print("Number of quantified salaries:", quantified_salary)
    print("Number of known email addresses:", known_email_address)
    
    # Print the total value of the stock belonging to James Prentice
    print("Total value of the stock belonging to James Prentice:", enron_data['PRENTICE JAMES']['total_stock_value'])
    
    # Print the number of email messages from Wesley Colwell to POIs
    print("Number of email messages from Wesley Colwell to POIs:", enron_data['COLWELL WESLEY']['from_this_person_to_poi'])
    
    print(f"total_payments for lay kenneth: {enron_data['LAY KENNETH L']['total_payments']}")
    print(f"total_payments for skilling jeffrey: {enron_data['SKILLING JEFFREY K']['total_payments']}")
    print(f"total_payments for fastow andrew: {enron_data['FASTOW ANDREW S']['total_payments']}")
    try:
        print("Value of stock options exercised by Jeffrey K Skilling:", enron_data['JEFFREY K SKILLING']['exercised_stock_options'])
    except KeyError:
        print("Error: 'SKILLING JEFFREY K' not found in the dataset.")
    
except Exception as e:
    print(f"Error loading dataset: {e}")


