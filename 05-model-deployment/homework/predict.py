import pickle


with open('pipeline_v1.bin', 'rb') as f_in:
    model = pickle.load(f_in)


sample = {
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0
}


pred = model.predict_proba([sample])[0, 1]

print("Predicted probability:", pred)
