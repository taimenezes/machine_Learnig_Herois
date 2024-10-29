from flask import Flask, render_template
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    # Garantir que a pasta para imagens existe
    os.makedirs(os.path.join('static', 'images'), exist_ok=True)
    app.run(debug=True)
