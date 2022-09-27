from argparse import FileType
import os
import sys
import json
from urllib.request import urlopen
from tkinter import *
# from PIL import ImageTk, Image
import tkinter.filedialog
import tkinter

# In this code we are trying to implemenet some new stuff

#################################################################
###### IMPORT METHOD FROM PYTHON FILE IN ANOTHER DIRECTORY ######
#################################################################

sys.path.append("C:\\Users\\BatuhanTosun\\Projects\\practices\\some_unnecessary_functions")
from stores_functions import say_my_name_func, another_function_to_tell_ready # even though there seems to be a problem there is not :)

def open():
    root = Tk()
    root.filename = tkinter.filedialog.askopenfilename(initialdir="C:\\", title="Select a File Sir, please!", filetypes=(("png files","*.png"),("all files","*.*")))
    my_label = tkinter.Label(root, text=root.filename).pack()
    # my_image = ImageTk.PhotoImage(Image.open(root.filename))
    # my_image_label = tkinter.Label(image=my_image).pack()


if __name__ == "__main__":

    say_my_name_func("name")
    say_my_name_func("surname")
    say_my_name_func("age")
    another_function_to_tell_ready()


############################################
###### CREATING AND READING JSON FILE ######
############################################


# multi-line string and looks like dictionary
# has a key called "people"
# value of people is an array of more objects
# these objects has a key of "name","phone","emails","has_license"

    people_string = '''

    {
        "people":[

            {
                "name": "John Smith",
                "phone": "615-55-7164",
                "emails": ["johnsmith@bogusemail.com", "john.smith@work-place"],
                "has_license": false
            },

            {
                "name": "Jane Doe",
                "phone": "560-55-5153",
                "emails": null,
                "has_license": true
            }

        ]

    }

    '''

    # 1) to load this into Python from a string


    data = json.loads(people_string)
    print(data)
    print(type(data))
    print(type(data['people']))

    for person in data['people']:
        print(person["name"])
        del person["phone"]

    new_string = json.dumps(data, indent=2, sort_keys=True)
    print(new_string)

    # 2) how to load from json file
    # first we need to open it
    with open('C:\\Users\\BatuhanTosun\\Projects\\practices\\some_necessary_py_files\\states.json') as f:
        data = json.load(f)
    
    for state in data['states']:
        # print(state['name'],state['abbreviation'])
        del state['area_codes']

    
    with open('C:\\Users\\BatuhanTosun\\Projects\\practices\\some_necessary_py_files\\new_states.json', 'w') as f:
        json.dump(data, f, indent=2)
    """ 
    # 3) Grabbing JSON data from a public API looks like

    with urlopen("https://finance.yahoo.com/webservice/v1/symbols/allcurrencies/quote?format=json") as response:
        source = response.read()
    
    # load this response into python object

    data = json.loads(source)
    #print(json.dumps(data, indent=2))
    print(len(data['list']['resources']))
    # 
    usd_rates = dict()

    for item in data['list']['resources']:
        name = item['resource']['fields']['name']
        price = item['resource']['fields']['name']
        usd_rates[name] = price

    print(usd_rates['USD/EUR'])

    """
###################################################
###### TKINTER FOLDER/FILE SELECTION REQUEST ######
###################################################


    # Open Files Dialog Box - Python Tkinter GUI
    root = tkinter.Tk()
    # root.title('Batuhan Tosun')
    # root.iconbitmap('icon_adress.ico') # to add icon

    root.filename = tkinter.filedialog.askopenfilename(initialdir="C:\\", title="Select a File Sir, please!", filetypes=(("png files","*.png"),("all files","*.*")))
    
    my_label = tkinter.Label(root, text=root.filename).pack()
    # my_image = ImageTk.PhotoImage(Image.open(root.filename))
    # my_image_label = tkinter.Label(image=my_image).pack()


    my_btn = tkinter.Button(root, text="Open File", command=open).pack()
    root.mainloop()




