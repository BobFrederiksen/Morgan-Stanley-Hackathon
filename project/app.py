from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Define your routes (endpoints)
@app.route('/')
def home():
    return "Hello, Flask!"

# Additional example route
@app.route('/api/example', methods=['GET'])
def example_api():
    return jsonify({"message": "This is an example API response"})




if __name__ == '__main__':
    app.run(debug=True)
