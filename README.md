# phishing-website-detection-using-machine-learning
We propose a multi-modal phishing detection model that combines URL features and HTML features extracted from the website.
•	URL Features: URL length, special characters (@, -, ?, =), number of subdomains, presence of IP, TLD.
•	HTML Features: Number of forms, scripts, iframes, hidden inputs, external links, presence of eval() in JavaScript.
•	Model: Random Forest Classifier trained on combined URL + HTML features.
•	Goal: Improve detection accuracy and robustness for phishing websites.
