import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import joblib
 
df = pd.read_csv("data/products.csv")
 
# drop all rows with missing values
df = df.dropna()
 
# Funkcija za standardizaciju kategorija
def standardize_categories(series: pd.Series) -> pd.Series:
    series = series.str.strip().str.lower()

    mapping = {
        "cpu": "cpus",
        "cpus": "cpus",
        "mobile phone": "mobile phones",
        "mobile phones": "mobile phones",
        "fridge": "fridges",
        "fridges": "fridges",
        "freezer": "freezers",
        "freezers": "freezers",
        "tv": "tvs",
        "tvs": "tvs",
        "dishwasher": "dishwashers",
        "dishwashers": "dishwashers",
        "digital camera": "digital cameras",
        "digital cameras": "digital cameras",
        "microwave": "microwaves",
        "microwaves": "microwaves",
        "washing machine": "washing machines",
        "washing machines": "washing machines",
        "fridge freezer": "fridge freezers",
        "fridge freezers": "fridge freezers"
    }

    return series.map(mapping).fillna(series)

# Primena standardizacije
df["Category Label"] = standardize_categories(df["Category Label"])

# Convert column type to 'category'
df['Category Label'] = df['Category Label'].astype('category')
print("\nCategory column converted to type:", df['Category Label'].dtype)

# Drop columns that are not useful for modeling
df = df.drop(columns=['product ID', 'Merchant ID', 'Product Code', 'Number_of_Views','Merchant Rating', ' Listing Date  '])
 
# Features and label
X = df[["Product Title"]]
y = df["Category Label"]
 
# Preprocessor: TF-IDF for text, MinMaxScaler for numeric feature
preprocessor = ColumnTransformer(
    transformers=[
        ("Product Title", TfidfVectorizer(), "Product Title")
        ]
)
 
# Define pipeline with the best model (e.g. LinearSVC)
pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", LinearSVC())
])
 
# Train the model on the entire dataset
pipeline.fit(X, y)
 
# Save the model to a file
joblib.dump(pipeline, "model/category_model.pkl")
 
print(" Model trained and saved as 'model/category_model.pkl'")