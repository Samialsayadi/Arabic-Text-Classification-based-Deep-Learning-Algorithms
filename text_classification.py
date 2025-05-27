import pandas as pd
from sklearn.model_selection import train_test_split

# Load your prepared data
# Assuming you have a DataFrame with 'text' and 'label' columns
data = pd.read_csv('aji-Arabic_corpus.csv')

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    data['text'], data['label'], test_size=0.2, random_state=42
)