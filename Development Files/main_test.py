from flask import Flask

app = Flask(__name__)

# @app.route("/")
# def fahrenheit_from(celsius):
#     """Convert Celsius to Fahrenheit degrees."""
#     try:
#         fahrenheit = float(celsius) * 9 / 5 + 32
#         fahrenheit = round(fahrenheit, 3)  # Round to three decimal places
#         return str(fahrenheit)
#     except ValueError:
#         return "invalid input"

def index():
    return "Congratulations, it's a web app!"

if __name__ == "__main__":
    # celsius = input("Celsius: ")
    # print("Fahrenheit:", fahrenheit_from(celsius))
    app.run(host="127.0.0.1", port=8080, debug=True)
