import joblib
import pandas as pd
 
# Load the saved model
model = joblib.load("model/category_model.pkl")
 
print("Model loaded successfully!")
print("Type 'exit' at any point to stop.\n")
 
while True:
    product_title = input(" Enter Product title: ")
    if product_title.lower() == "exit":
        print("Exiting...")
        break

    # Create a DataFrame from input
    user_input = pd.DataFrame([{
        "Product Title": product_title,
        
    }])
 
    # Predict category
    prediction = model.predict(user_input)[0]
    print(f" Predicted category: {prediction}\n" + "-" * 40)