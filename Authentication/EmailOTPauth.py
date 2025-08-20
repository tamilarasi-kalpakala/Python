# Use google smtp app to sent a verification code to a user's Gmail, and verify like an OTP
import smtplib
import random

# Gmail credentials (use developer App Password, NOT your main gmail account password)
# Google Account → Security → App passwords → Generate 16-character password

EMAIL_ADDRESS = "your_email@gmail.com"
EMAIL_PASSWORD = "your_app_password"

def send_verification_email(to_email):
    # Generate a 6-digit verification code
    code = str(random.randint(100000, 999999))

    subject = "Your Verification Code"
    body = f"Your verification code is: {code}"

    # Email message format
    message = f"Subject: {subject}\n\n{body}"

    # Send email using Gmail's SMTP server
    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.sendmail(EMAIL_ADDRESS, to_email, message)

    return code


# Example usage
user_email = input("Enter your Gmail: ")
verification_code = send_verification_email(user_email)

# Ask user to enter received code
entered_code = input("Enter the code sent to your email: ")

if entered_code == verification_code:
    print("Email verified successfully!")
else:
    print("Verification failed!")