from flask import Flask, request, jsonify
import smtplib
import random

app = Flask(__name__)

# Gmail credentials (use App Password here)
EMAIL_ADDRESS = "your_email@gmail.com"
EMAIL_PASSWORD = "your_app_password"

# Store codes in memory (dict)
verification_codes = {}

def send_verification_email(to_email, code):
    subject = "Your Verification Code"
    body = f"Your verification code is: {code}"
    message = f"Subject: {subject}\n\n{body}"

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.sendmail(EMAIL_ADDRESS, to_email, message)


@app.route("/send_code", methods=["POST"])
def send_code():
    data = request.json
    user_email = data.get("email")

    if not user_email:
        return jsonify({"error": "Email is required"}), 400

    # Generate a 6-digit code
    code = str(random.randint(100000, 999999))
    verification_codes[user_email] = code

    # Send email
    send_verification_email(user_email, code)

    return jsonify({"message": "Verification code sent"}), 200


@app.route("/verify_code", methods=["POST"])
def verify_code():
    data = request.json
    user_email = data.get("email")
    entered_code = data.get("code")

    if not user_email or not entered_code:
        return jsonify({"error": "Email and code are required"}), 400

    # Check if correct
    if verification_codes.get(user_email) == entered_code:
        return jsonify({"message": "✅ Email verified successfully"}), 200
    else:
        return jsonify({"error": "❌ Invalid code"}), 400


if __name__ == "__main__":
    app.run(debug=True)


# To run this Flask app, save it as a Python file and run it using:
# python3 EmailOTPusingFlaskAPI.py
# Ensure Flask is installed via pip3 install flask
# You can then test the API using tools like Postman or curl.
# Ensure to replace EMAIL_ADDRESS and EMAIL_PASSWORD with your actual Gmail credentials
# and generate an App Password for security.
# Sent code - curl -X POST http://127.0.0.1:5000/send_code -H "Content-Type: application/json" -d '{"email":"user@gmail.com"}'
# Verify code - curl -X POST http://127.0.0.1:5000/verify_code -H "Content-Type: application/json" -d '{"email":"user@gmail.com","code":"123456"}'