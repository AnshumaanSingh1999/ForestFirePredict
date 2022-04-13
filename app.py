from flask import Flask, redirect, render_template, request, url_for
import pickle
import numpy as np



app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))



@app.route('/')
def hello():
    return render_template('index.html')
    #return 'Hello, World!'
   
@app.route('/report',methods=['POST','GET'])
def ew():
    if request.method =="POST":
        #t=int(request.form['t'])
        #h=int(request.form['h'])
        #w=int(request.form['w'])
        #r=int(request.form['r'])
        int_features = [int(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        a=model.predict_proba(final_features)
        #a=log_reg.predict_proba(inp)
        if(a[0][1]>=0.6):
            s="fire"
        else:
            s="cool"

        #s=final_features[0][0]+final_features[0][1]+final_features[0][2]+final_features[0][3]
        #s=t+h+w+r     
        return render_template('report.html', results=s)
    else:
        return render_template('index.html')




if __name__ =="__main__":
    app.debug=True
    app.run()

