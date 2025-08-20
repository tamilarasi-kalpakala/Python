from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, EmailStr
import smtplib
import random
from datetime import datetime, timedelta
import asyncio

app = FastAPI(title="Gmail Verification API (Async Email)")

# Gmail credentials (use App Password)
EMAIL_ADDRESS = "your_email@gmail.com"
EMAIL_PASSWORD = "your_app_password"

# Store verification info
verification_data = {}  # {email: {"code": str, "expires": datetime, "attempts": int}}

CODE_EXPIRY_MINUTES = 5
MAX_ATTEMPTS = 3

# Pydantic models
class EmailRequest(BaseModel):
    email: EmailStr

class VerifyRequest(BaseModel):
    email: EmailStr
    code: str

def send_email_blocking(to_email: str, code: str):
    subject = "Your Verification Code"
    body = f"Your verification code is: {code} (valid for {CODE_EXPIRY_MINUTES} minutes)"
    message = f"Subject: {subject}\n\n{body}"

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.sendmail(EMAIL_ADDRESS, to_email, message)

async def send_email_async(to_email: str, code: str):
    # Run blocking email sending in a separate thread
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, send_email_blocking, to_email, code)

# Endpoint to send code
@app.post("/send_code")
async def send_code(request: EmailRequest):
    code = str(random.randint(100000, 999999))
    expires = datetime.now() + timedelta(minutes=CODE_EXPIRY_MINUTES)
    verification_data[request.email] = {"code": code, "expires": expires, "attempts": 0}

    # Send email asynchronously
    asyncio.create_task(send_email_async(request.email, code))

    return {"message": f"Verification code sent to {request.email}, valid for {CODE_EXPIRY_MINUTES} minutes."}

# Endpoint to verify code
@app.post("/verify_code")
def verify_code(request: VerifyRequest):
    data = verification_data.get(request.email)
    if not data:
        raise HTTPException(status_code=400, detail="No code sent to this email")

    # Check expiry
    if datetime.now() > data["expires"]:
        del verification_data[request.email]
        raise HTTPException(status_code=400, detail="Verification code expired")

    # Check attempts
    if data["attempts"] >= MAX_ATTEMPTS:
        del verification_data[request.email]
        raise HTTPException(status_code=400, detail="Maximum verification attempts exceeded")

    # Verify code
    if request.code == data["code"]:
        del verification_data[request.email]  # code used, remove entry
        return {"message": "Email verified successfully"}
    else:
        data["attempts"] += 1
        remaining = MAX_ATTEMPTS - data["attempts"]
        raise HTTPException(status_code=400, detail=f"Invalid code. {remaining} attempt(s) left")


      # To run this code, you need to install FastAPI and Uvicorn
      # pip install fastapi uvicorn
      # Then run the server using the command:
      # uvicorn EmailOTPusingFastAPI:app --reload
        # API will run at http://127.0.0.1:8000
        # To know API specs use Swagger UI where FASTAPI automatically provides API Docs: http://127.0.0.1:8000/docs
        # Now you can test the API using Postman or any other API testing tool
        # Send OTP - curl -X POST "http://127.0.0.1:8000/send_code" \-H "Content-Type: application/json" \-d '{"email":"user@gmail.com"}'
        # Verify OTP - curl -X POST "http://127.0.0.1:8000/verify_code" \ -H "Content-Type: application/json" \ -d '{"email":"user@gmail.com", "code":"123456"}'


