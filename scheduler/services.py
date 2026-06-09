import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os

logger = logging.getLogger("NotificationService")

class NotificationService:
    def __init__(self):
        self.email = os.getenv('EMAIL_ADDRESS')
        self.password = os.getenv('EMAIL_PASSWORD')
        self.smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')

    def send_interview_invite(self, to_email: str, candidate_name: str, link: str, time_str: str):
        if not self.email or not self.password:
            logger.warning(f"Email credentials not set. Mocking email to {to_email}")
            logger.info(f"Link would be: {link}")
            return

        try:
            msg = MIMEMultipart()
            msg['From'] = self.email
            msg['To'] = to_email
            msg['Subject'] = "Interview Invitation - Action Required"

            body = f"""
            Dear {candidate_name},

            Thank you for your application. We are excited to invite you to an AI-conducted interview.

            Scheduled Time: {time_str}
            
            Please join using the following secure link at the scheduled time:
            {link}

            Note: This interview will be conducted by our AI interviewer, Eve. Please ensure you have a working microphone and camera.

            Best regards,
            Recruitment Team
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            with smtplib.SMTP_SSL(self.smtp_server, 465) as server:
                server.login(self.email, self.password)
                server.send_message(msg)
                
            logger.info(f"Sent invite to {to_email}")

        except Exception as e:
            logger.error(f"Failed to send email: {e}")
