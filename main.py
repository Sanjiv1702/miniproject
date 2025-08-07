import sqlite3
from flask import Flask, render_template, request, redirect
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Database setup
def init_db():
    conn = sqlite3.connect('finance.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            amount REAL,
            category TEXT,
            anomaly_label TEXT
        )
    ''')
    conn.commit()
    conn.close()

# Anomaly detection
def detect_anomaly(amounts):
    df = pd.DataFrame({'Amount': amounts})
    scaler = StandardScaler()
    df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
    model = IsolationForest(contamination=0.2, random_state=42)
    df['Anomaly'] = model.fit_predict(df[['Amount_scaled']])
    df['Label'] = df['Anomaly'].apply(lambda x: 'Anomaly' if x == -1 else 'Normal')
    return df['Label'].tolist()

@app.route('/')
def index():
    conn = sqlite3.connect('finance.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM transactions")
    transactions = cursor.fetchall()
    conn.close()
    return render_template('index.html', transactions=transactions)

@app.route('/add', methods=['POST'])
def add_transaction():
    amount = float(request.form['amount'])
    category = request.form['category']
    # Get previous amounts + new one
    conn = sqlite3.connect('finance.db')
    cursor = conn.cursor()
    cursor.execute("SELECT amount FROM transactions")
    amounts = [row[0] for row in cursor.fetchall()] + [amount]
    label = detect_anomaly(amounts)[-1]
    cursor.execute("INSERT INTO transactions (amount, category, anomaly_label) VALUES (?, ?, ?)",
                   (amount, category, label))
    conn.commit()
    conn.close()
    return redirect('/')

if __name__ == '__main__':
    init_db()
    app.run(debug=True)