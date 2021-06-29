from flask import Flask
from flask import render_template, request, url_for, jsonify
import pickle
import numpy as np
import pandas

app = Flask(__name__,template_folder="templates")

model = pickle.load(open("rf_model.pkl","rb"))
label = pickle.load(open("label_encoder.pkl","rb"))

columns = ["age","workclass","fnlgwt","education","education_num","marital_status","occupation",
           "relationship","race","sex","capital_gain","capital_loss","hours_per_week","native_country"]

encode_columns = ["occupation","workclass","education","marital_status","relationship","race","sex","native_country"]

@app.route("/")
def index():
    return render_template("adult.html")


def encode_data(column: str, data):
    """encodes the column using label encoder"""
    return label.fit_transform(data[column])


@app.route("/",methods=["POST"])
def predict():
    #creates the single function for getting data
    if request.method == "POST":
        req = request.form
        age = int(req.get("age",default=0))
        workclass = req.get("workclass",default=0)
        fnlgwt = int(req.get("fblgwt",default=0))
        education = req.get("education",default=0)
        education_num = int(req.get("education_num",default=0))
        marital_status = req.get("marital_status",default=0)
        occupation = req.get("occupation",default=0)
        relationship = req.get("relationship",default=0)
        race = req.get("race",default=0)
        sex = req.get("sex",default=0)
        capital_gain = int(req.get("capital_gain",default=0))
        captal_loss = int(req.get("capital_loss",default=0))
        hours_per_week = int(req.get("hours_per_week",default=0))
        native_country = req.get("native_country",default=0)

        #storing in array
        array = np.array([age,workclass,fnlgwt,education,education_num,marital_status,
                            occupation,relationship,race,sex,capital_gain,captal_loss,hours_per_week,
                            native_country]).reshape(1,14)

        #creates a dataframe to hold the data and perform transformation
        data = pandas.DataFrame(data=array,columns=["age","workclass","fnlgwt","education","education_num","marital_status","occupation",
           "relationship","race","sex","capital_gain","capital_loss","hours_per_week","native_country"])

        #performing transformation
        for column in encode_columns:
            data[column] = encode_data(column, data)
        
        #predict over the features gotten from the user
        value = model.predict(data)
        if value == [1]:
            text = "This worker will earn over $50k dollars at the end of the year"
        else:
            text = "This worker will earn less than $50k dollars at the end of the year"
        #passing value gotten to template for rendering
        return render_template("adult.html",text=text)

# app.route("/results",methods=["POST"])
# def results():
#     #this gets the result from the user and return the json representation
#     data = request.get_json(force=True)
#     frame = pandas.DataFrame(data.values(),columns=columns)
#     frame["occupation"] = label.fit_transform(frame["occupation"])
#     frame["workclass"] = label.fit_transform(frame["workclass"])
#     frame["education"] = label.fit_transform(frame["education"])
#     frame["marital_status"] = label.fit_transform(frame["marital_status"])
#     frame["relationship"] = label.fit_transform(frame["relationship"])
#     frame["race"] = label.fit_transform(frame["race"])
#     frame["sex"] = label.fit_transform(frame["sex"])
#     frame["native_country"] = label.fit_transform(frame["native_country"])

#     value = model.predict(frame)

#     #this is for determining the position
#     if value == [1]:
#             text = "This person will earn over $50k dollars"
#     else:
#         text="This person will earn less than $50k dollars"


#     return jsonify(text)


if __name__ == "__main__":
    from flask import Flask
    from flask import render_template, request, url_for, jsonify
    import pickle
    import numpy as np
    import pandas

    app = Flask(__name__,template_folder="templates")

    model = pickle.load(open("adult/rf_model.pkl","rb"))
    label = pickle.load(open("adult/label_encoder.pkl","rb"))

    columns = ["age","workclass","fnlgwt","education","education_num","marital_status","occupation",
            "relationship","race","sex","capital_gain","capital_loss","hours_per_week","native_country"]

    @app.route("/")
    def index():
        return render_template("adult.html")


    @app.route("/",methods=["POST"])
    def predict():
        #creates the single function for getting data
        if request.method == "POST":
            req = request.form
            age = int(req.get("age",default=0))
            workclass = req.get("workclass",default=0)
            fnlgwt = int(req.get("fblgwt",default=0))
            education = req.get("education",default=0)
            education_num = int(req.get("education_num",default=0))
            marital_status = req.get("marital_status",default=0)
            occupation = req.get("occupation",default=0)
            relationship = req.get("relationship",default=0)
            race = req.get("race",default=0)
            sex = req.get("sex",default=0)
            capital_gain = int(req.get("capital_gain",default=0))
            captal_loss = int(req.get("capital_loss",default=0))
            hours_per_week = int(req.get("hours_per_week",default=0))
            native_country = req.get("native_country",default=0)

            #storing in array
            array = np.array([age,workclass,fnlgwt,education,education_num,marital_status,
                                occupation,relationship,race,sex,capital_gain,captal_loss,hours_per_week,
                                native_country]).reshape(1,14)

            #creates a dataframe to hold the data and perform transformation
            data = pandas.DataFrame(data=array,columns=["age","workclass","fnlgwt","education","education_num","marital_status","occupation",
            "relationship","race","sex","capital_gain","capital_loss","hours_per_week","native_country"])

            #performing transformation
            data["occupation"] = label.fit_transform(data["occupation"])
            data["workclass"] = label.fit_transform(data["workclass"])
            data["education"] = label.fit_transform(data["education"])
            data["marital_status"] = label.fit_transform(data["marital_status"])
            data["relationship"] = label.fit_transform(data["relationship"])
            data["race"] = label.fit_transform(data["race"])
            data["sex"] = label.fit_transform(data["sex"])
            data["native_country"] = label.fit_transform(data["native_country"])
            
            #predict over the features gotten from the user
            value = model.predict(data)
            if value == [1]:
                text = "This person will earn over $50k dollars"
            else:
                text = "This person will earn less than $50k dollars"
            #passing value gotten to template for rendering
            return render_template("adult.html",text=text)

    app.route("/results",methods=["POST"])
    def results():
        #this gets the result from the user and return the json representation
        data = request.get_json(force=True)
        frame = pandas.DataFrame(data.values(),columns=columns)
        frame["occupation"] = label.fit_transform(frame["occupation"])
        frame["workclass"] = label.fit_transform(frame["workclass"])
        frame["education"] = label.fit_transform(frame["education"])
        frame["marital_status"] = label.fit_transform(frame["marital_status"])
        frame["relationship"] = label.fit_transform(frame["relationship"])
        frame["race"] = label.fit_transform(frame["race"])
        frame["sex"] = label.fit_transform(frame["sex"])
        frame["native_country"] = label.fit_transform(frame["native_country"])

        value = model.predict(frame)

        #this is for determining the position
        if value == [1]:
                text = "This person will earn over $50k dollars"
        else:
            text="This person will earn less than $50k dollars"


        return jsonify(text)


    if __name__ == "__main__":
        app.run(debug=True)