import numpy as np
from flask import Flask,request,jsonify,render_template
import pickle
import keras 
from keras.models import load_model
import pandas as pd
#import seaborn as sns
#import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

app=Flask(__name__)


def update(X_check,Y_new,Y_final):
    e=float(Y_new[0])
    d=X_check
    d[0,0]=0
    d[0,6]=e
    Y_final.append(e)
    X_check=d
    return X_check,(Y_final)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    

    rmse={'Karnataka':140,'Andhra Pradesh':56,'Tamil Nadu':17,'Delhi':86,'Uttar Pradesh':210,'Rajasthan':88,'Haryana':35,'Gujarat':39,
          'Madhya Pradesh':70,'Assam':12,'Jammu and Kashmir':29,'Goa':17,'Odisha':30,'Bihar':80,'Chhattisgarh':140,'Jharkhand':31,'Uttarakhand':74}

    choice=str(request.form['state'])



    days=7
    df=pd.read_csv('State Name +ve Cases.csv')
    X=df[choice]
    X_check=np.array(X[X.shape[0]-7:])
    X_check=X_check.reshape(-1,1)
    scaler=MinMaxScaler(feature_range=(0,1))
    X_check=scaler.fit_transform(X_check)
    X_check=X_check.reshape((1,7,1))
    Y_final=[]


    model=load_model(r"D:/EY/deployment/models/"+choice+".h5")
    
    for i in range(0,days):
        Y_new=model.predict(X_check)
        X_check,Y_final=update(X_check,Y_new,Y_final)

    final_prediction=scaler.inverse_transform(np.array(Y_final).reshape(days,1))
    final_prediction=final_prediction.astype("int") #7 days ka prediction
  
    

    #lst=["Day-"+str(i) for i in range (1,days+1)]
    new=pd.DataFrame()
    new['Days']=["Day-"+str(i) for i in range (1,8)]
    lst=[]
    for i in final_prediction:
        y=i+rmse[choice]
        x=i-rmse[choice]
        lst.append(str(x)+'-'+str(y))
    new[choice]=lst

    qwerty=pd.DataFrame()
    qwerty['Prediction(Approx.)']=['Next Day','Next 3 Days','Next 7 Days']
    cnt=0
    fst=[]
    s=p=y=z=0
    for i in new[choice].values:
        if cnt==0:
          fst.append(i)
        if cnt>0 and cnt<3:
            x=i.split('-')
            s=s+int(x[0])
            p+=int(x[1])

            if cnt==2:
                a=fst[0].split('-')
                s+=int(a[0])
                p+=int(a[1])
                fst.append(str(s)+'-'+str(p))

        if cnt>2:
            y+=int(i.split('-')[0])
            z+=int(i.split('-')[1])

            if cnt==6:
                y+=int(fst[1].split('-')[0])
                z+=int(fst[1].split('-')[0])
                fst.append(str(y)+'-'+str(z))
        cnt+=1
    qwerty['Cases']=fst

    
    return render_template('index.html',  tables=[qwerty.to_html(classes='state',justify='center')], titles=['na','Prediction for {}'.format(choice)])



if __name__=='__main__':
    app.run(debug=True)
    