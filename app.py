#import os
#os.chdir(r"C:\Users\DELL\Desktop")
import pickle
import numpy as np
from flask import Flask,render_template, request

model = None
app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))
@app.route('/')
def home_endpoint():
    return render_template('m.html')
@app.route('/predict',methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        #For rendering results on HTML GUI
        int_features = [float(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        prediction = model.predict(final_features)
        output = round(prediction[0],2)     
        if output<4:
            return render_template('m.html', pred='Network Connectivity is :{}G.\nBidding down attack has occured'.format(output))
        else:
            return render_template('m.html', pred='Network Connectivity is :{}G.\nBidding down attack has NOT occured'.format(output))
if __name__ == '__main__':
    app.debug = True
    app.run()
    
