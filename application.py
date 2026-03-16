
from flask import Flask, render_template , request


from sklearn.preprocessing import StandardScaler
from src.pipelines.predict_pipeline import CustomData,PredictPipeline

application = Flask(__name__)

app = application

@app.route('/',methods=['GET','POST'])
def homepage():
    return render_template('index.html')

@app.route('/predictions',methods=['GET','POST'])
def predict():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('race_ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            test_preparation_course=request.form.get('test_preparation_course'),
            lunch=request.form.get('lunch'),
            reading_score=int(request.form.get('reading_score')),
            writing_score=int(request.form.get('writing_score')),
        )
        
        pred_df = data.get_input_as_dataframe()
        print(pred_df)
        # Predictions 
        pred_pipeline = PredictPipeline()
        predictions=pred_pipeline.predict(pred_df)
        
        return render_template('home.html',results=predictions[0])

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080)