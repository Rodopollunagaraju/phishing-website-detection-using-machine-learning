
# Phishing Detection: URL + HTML

import re
import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
# 1. Feature Extraction Functions


# URL Features
def extract_url_features(url):
    features = {}
    features['url_length'] = len(url)
    features['count_dashes'] = url.count('-')
    features['count_at'] = url.count('@')
    features['count_qm'] = url.count('?')
    features['count_eq'] = url.count('=')
    features['count_subdomains'] = url.count('.') - 1  # assuming domain.com
    features['has_ip'] = int(bool(re.match(r"\d+\.\d+\.\d+\.\d+", url)))

    # Top-level domain
    tld = url.split('.')[-1]
    features['tld'] = tld
    return features

# HTML Features
def extract_html_features(html):
    features = {}
    soup = BeautifulSoup(html, 'html.parser')
    features['num_forms'] = len(soup.find_all('form'))
    features['num_iframes'] = len(soup.find_all('iframe'))
    scripts = soup.find_all('script')
    features['num_scripts'] = len(scripts)
    features['contains_eval'] = int(any('eval(' in script.text for script in scripts))
    features['num_external_links'] = len([a for a in soup.find_all('a', href=True) if not a['href'].startswith('#')])
    features['num_hidden_inputs'] = len(soup.find_all('input', type='hidden'))
    return features

# 2. Load / Prepare Dataset
# DataFrame with columns: ['url', 'label']
df = pd.read_csv("/content/drive/MyDrive/dataset/dataset.csv") 

url_features_list = []
html_features_list = []
labels = []

for idx, row in df.iterrows():
    # Updated column names based on the variable explorer output
    url = row['URLURL_Length'] 
    label = row['Result'] # Assuming Result is the label column

    # Extract URL features
    url_feat = extract_url_features(str(url)) # Convert to string in case it's numeric

    # Fetch HTML safely
    try:
        r = requests.get(str(url), timeout=5)
        html = r.text
    except:
        html = ""

    html_feat = extract_html_features(html)

    url_features_list.append(url_feat)
    html_features_list.append(html_feat)
    labels.append(label)

# 3. Combine Features

url_df = pd.DataFrame(url_features_list)
html_df = pd.DataFrame(html_features_list)

# Encode categorical TLD
le = LabelEncoder()
# Handle potential NaNs before fitting and transforming
url_df['tld'] = url_df['tld'].astype(str)
le.fit(url_df['tld'].unique())
url_df['tld'] = le.transform(url_df['tld'])


# Combine URL + HTML features
X = pd.concat([url_df, html_df], axis=1)
y = np.array(labels)

# Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)


# 5. Train Model
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)


# 6. Evaluation

y_pred = clf.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 7. Example Prediction
test_url = "http://example-phishing.com/login"
test_url_feat = extract_url_features(test_url)

try:
    html = requests.get(test_url, timeout=5).text
except:
    html = ""

test_html_feat = extract_html_features(html)
test_df = pd.DataFrame([ {**test_url_feat, **test_html_feat} ])
# Ensure 'tld' in test_df is a string before transforming
test_df['tld'] = test_df['tld'].astype(str)

# Handle unseen 'tld' values during transformation
try:
    test_df['tld'] = le.transform(test_df['tld'])
except ValueError:
    # Assign a default value for unseen TLDs (e.g., -1 or the most frequent TLD)
    # For simplicity, we'll assign -1. You might want to use a more sophisticated approach
    test_df['tld'] = -1

test_scaled = scaler.transform(test_df)
prediction = clf.predict(test_scaled)
print(f"Prediction for {test_url}: {prediction[0]}")

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Visualize Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['-1', '1'], yticklabels=['-1', '1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
# Print Classification Report
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

plt.figure(figsize=(10, 6))
sns.heatmap(report_df.iloc[:-1, :].T, annot=True, cmap='Blues', fmt=".2f")
plt.title('Classification Report')
plt.show()
