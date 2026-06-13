import smtplib
from email.message import EmailMessage
import os

# ⚠️ For production, move these to environment variables
SENDER_EMAIL = "gssvlaakash@gmail.com"
APP_PASSWORD = "bxgs ohzr hfgv lchh"

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465


def send_violence_alert(video_name, violence_prob, frame_path=None):
    msg = EmailMessage()
    msg["From"] = SENDER_EMAIL
    msg["To"] = SENDER_EMAIL   # you can change receiver
    msg["Subject"] = "🚨 Violence Detected Alert"

    msg.set_content(
        f"""
🚨 Violence Detected

Video Name: {video_name}
Violence Probability: {violence_prob:.2f}

Attached is the key frame where violence is most prominent.

Immediate attention required.
        """
    )

    # 🔑 Attach frame image if available
    if frame_path and os.path.exists(frame_path):
        with open(frame_path, "rb") as f:
            image_data = f.read()

        msg.add_attachment(
            image_data,
            maintype="image",
            subtype="jpeg",
            filename=os.path.basename(frame_path)
        )

    try:
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
            server.login(SENDER_EMAIL, APP_PASSWORD)
            server.send_message(msg)

        print("✅ Email sent successfully")

    except Exception as e:
        print("❌ Email failed:", e)
