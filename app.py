from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
# ئەم دێڕە ڕێگە دەدات وێبسایتەکەی گیت‌هەب قسە لەگەڵ ئەم سێرڤەرە بکات
CORS(app) 

# هێنانە ناوەوەی مۆدێلە نوێیەکە
model = joblib.load('transformer_model.pkl')

@app.route('/predict', methods=['POST'])
def predict_transformer():
    try:
        data = request.get_json()
        houses = data['houses']
        area = data['area']
        status = data['status']
        season = data['season']
        
        # ١. گۆڕینی دۆخی ئابووری بۆ ژمارە
        status_map = {'Low': 1, 'Middle': 2, 'High': 3}
        status_encoded = status_map.get(status, 2) # 2 وەک دیفۆڵت ئەگەر هەڵەیەک هەبوو
        
        # ٢. دروستکردنی ستوونەکانی وەرز (One-Hot Encoding)
        # تێبینی: وەرزی پاییز (Autumn) سفرە لە هەموویان چونکە drop_first=True مان بەکارهێنا لە مۆدێلەکەدا
        season_spring = 1 if season == 'Spring' else 0
        season_summer = 1 if season == 'Summer' else 0
        season_winter = 1 if season == 'Winter' else 0
        
        # ٣. خستنە ناو داتا فرەیمێک بە ڕێکخستنی دروستی ستوونەکان (ڕێک وەک ئەوەی مۆدێلەکە فێری بووە)
        input_data = pd.DataFrame(
            [[houses, area, status_encoded, season_spring, season_summer, season_winter]], 
            columns=['Num_Houses', 'Avg_Area', 'Status_Encoded', 'Season_Spring', 'Season_Summer', 'Season_Winter']
        )
        
        # ٤. پێشبینیکردن
        predicted_load = model.predict(input_data)[0]
        
        # ٥. دیاریکردنی قەبارەی محەویلەکە بە زیادکردنی ١٠٪ سەلامەتی
        standard_sizes =[50, 100, 250, 400, 630, 1000, 1600]
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
    return "API is running successfully with the New ML Model!"

if __name__ == '__main__':
    app.run()