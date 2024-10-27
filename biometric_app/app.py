from flask import Flask, render_template, request
import os
from training_code import run_training
from testing_code import run_testing

app = Flask(__name__)

# Home page with Training and Testing buttons
@app.route('/')
def index():
    return render_template('index.html')

# Route to run the training process
@app.route('/train', methods=['POST'])
def train():
    training_output = run_training()
    return render_template('index.html', training_output=training_output)

# Testing page where you input the folder number
@app.route('/test')
def test_page():
    return render_template('test.html')

# Route to run the testing process based on folder input
@app.route('/run_test', methods=['POST'])
def run_test():
    folder_num = request.form['folder_num']
    testing_output = run_testing(folder_num)
    return render_template('test.html', testing_output=testing_output)

if __name__ == '__main__':
    app.run(debug=True, port=5002)
