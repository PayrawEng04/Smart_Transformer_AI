from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
# ئەم دێڕە ڕێگە دەدات وێبسایتەکەی گیت‌هەب قسە لەگەڵ ئەم سێرڤەرە بکات
CORS(app) 

model = joblib.load('transformer_model.pkl')

@app.route('/predict', methods=['POST'])
def predict_transformer():
    try:
        data = request.get_json()
        houses = data['houses']
        area = data['area']
        status = data['status']
        season = data['season']
        
        status_map = {'Low': 1, 'Middle': 2, 'High': 3}
        season_map = {'Spring': 1, 'Autumn': 2, 'Winter': 3, 'Summer': 4}
        
        input_data = pd.DataFrame([[houses, area, status_map[status], season_map[season]]], 
                                  columns=['Num_Houses', 'Avg_Area', 'Status_Encoded', 'Season_Encoded'])
        
        predicted_load = model.predict(input_data)[0]
        
        standard_sizes = [50, 100, 250, 400, 630, 1000, 1600]
        rec_size = 1600
        for size in standard_sizes:
            if size >= (predicted_load * 1.1):
                rec_size = size
                break
                
        return jsonify({
            'success': True,
            'predicted_load_kva': round(predicted_load, 2),
            'recommended_transformer_kva': rec_size
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# ئەم بەشە تەنها بۆ تاقیکردنەوەی سەرەتاییە
@app.route('/')
def home():
    return "API is running successfully!"

if __name__ == '__main__':
    app.run()