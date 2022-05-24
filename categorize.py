from pathlib import Path
import joblib
import os

categories = { }

# Using the (sklearn based) classifiers, returns the category of the given file content
def categorize(id, content):
    global categories
    # Load the trained catgories once
    if len(categories) == 0:
        folder =  os.getcwd() + "/class/"
        files = os.listdir(folder)
        print(folder)
        for file in files:
            print(file)
            f = str(folder + '/' + file)
            classifier = joblib.load(f)
            categories[file.split('class.pkl')[0]] = classifier
        print("Categories:", list(categories.keys()))
    for category, classifier in categories.items():
        result = classifier.predict(content)
        if result == 1:
            #print(id, "-->", category.upper())
            break
        '''elif result == -1:
            print(id, "-->", "Not", category.upper())'''
    if result == -1:
        category = ""
    return category