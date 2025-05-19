import os
import re
import logging
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List

from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse, Gather
from email.utils import parsedate_to_datetime
import imaplib
import email
from email.header import decode_header
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pytz

# Import RAG processor
from rag_processor import RAGProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SimpleInterviewManager")

class SimpleInterviewManager:
    def __init__(self):
        # Email configuration
        self.email = os.getenv('EMAIL_ADDRESS')
        self.password = os.getenv('EMAIL_PASSWORD')
        self.imap_server = os.getenv('IMAP_SERVER', 'imap.gmail.com')
        self.smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        
        # Interview settings
        self.timezone = pytz.timezone('Asia/Kolkata')  # Adjust as needed
        self.interview_duration = 30  # minutes
        self.available_slots = self._generate_available_slots()
        
        # Initialize Twilio client
        self.twilio_client = Client(
            os.getenv('TWILIO_ACCOUNT_SID'),
            os.getenv('TWILIO_AUTH_TOKEN')
        )
        self.twilio_phone_number = os.getenv('TWILIO_PHONE_NUMBER')
        
        # Track scheduled interviews
        self.scheduled_interviews = {}
        
        # Initialize RAG processor
        self.rag_processor = RAGProcessor(persist_directory=os.path.join('database', 'chroma_db'))
        
        # Create database directory if it doesn't exist
        os.makedirs('database', exist_ok=True)
        os.makedirs(os.path.join('database', 'resumes'), exist_ok=True)
    
    def _generate_available_slots(self):
        """Generate available time slots for the next 7 days."""
        slots = []
        now = datetime.now(self.timezone)
        
        for day in range(1, 8):  # Next 7 days
            date = now + timedelta(days=day)
            
            # Skip weekends
            if date.weekday() >= 5:  # 5=Saturday, 6=Sunday
                continue
                
            # Add time slots (10 AM to 5 PM)
            for hour in range(10, 17):
                slot = date.replace(hour=hour, minute=0, second=0, microsecond=0)
                slots.append(slot)
                
        return slots
    
    def _connect_to_email_server(self, max_retries=3):
        """Establish connection to email server with retry logic."""
        for attempt in range(max_retries):
            mail = None
            try:
                logger.info(f"Connecting to email server (attempt {attempt + 1}/{max_retries})...")
                mail = imaplib.IMAP4_SSL(self.imap_server, timeout=30)
                
                # Login with timeout
                logger.debug("Logging in to email...")
                mail.login(self.email, self.password)
                return mail
                
            except (imaplib.IMAP4.error, socket.error, socket.timeout) as e:
                error_msg = str(e)
                logger.error(f"Error connecting to email server (attempt {attempt + 1}): {error_msg}")
                if mail:
                    try:
                        mail.logout()
                    except:
                        pass
                
                if attempt == max_retries - 1:  # Last attempt
                    logger.error("Max retries reached. Could not connect to email server.")
                    return None
                
                # Exponential backoff before retry
                wait_time = (2 ** attempt) * 5  # 5, 10, 20 seconds
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
        
        return None

    def check_emails(self):
        """Check for new emails with resumes."""
        logger.info("Checking for new emails...")
        mail = None
        
        try:
            # Connect to email server with retry logic
            mail = self._connect_to_email_server()
            if not mail:
                logger.error("Failed to establish email connection after retries")
                return
            
            try:
                # Select inbox
                logger.debug("Selecting inbox...")
                status, _ = mail.select('inbox')
                if status != 'OK':
                    logger.error("Failed to select inbox")
                    return
                
                # Search for unread emails
                logger.debug("Searching for unread emails...")
                status, messages = mail.search(None, 'UNSEEN')
                
                if status != 'OK':
                    logger.error("Failed to search emails")
                    return
                
                email_ids = messages[0].split()
                logger.info(f"Found {len(email_ids)} new emails")
                
                for num in email_ids:
                    try:
                        logger.debug(f"Processing email ID: {num.decode() if isinstance(num, bytes) else num}")
                        status, data = mail.fetch(num, '(RFC822)')
                        
                        if status != 'OK':
                            logger.error(f"Failed to fetch email {num}")
                            continue
                            
                        # Process email
                        email_message = email.message_from_bytes(data[0][1])
                        
                        # Get sender information
                        from_header = email_message.get('From')
                        name, email_addr = self._parse_sender(from_header)
                        
                        if not email_addr:
                            logger.warning(f"Could not parse sender email from: {from_header}")
                            continue
                        
                        # Check if this is a new candidate
                        if self._is_new_candidate(email_addr):
                            logger.info(f"New candidate found: {name} <{email_addr}>")
                            self._process_candidate_email(email_message, name, email_addr)
                        else:
                            logger.info(f"Skipping email from existing candidate: {email_addr}")
                            
                    except Exception as e:
                        logger.error(f"Error processing email {num.decode() if isinstance(num, bytes) else num}: {str(e)}")
                        continue  # Continue with next email even if one fails
                    
            except Exception as e:
                logger.error(f"Error checking emails: {str(e)}", exc_info=True)
                
        finally:
            if mail:
                try:
                    try:
                        mail.close()
                    except:
                        pass
                    mail.logout()
                except Exception as e:
                    logger.error(f"Error closing email connection: {str(e)}")
    
    def _parse_sender(self, from_header):
        """Extract sender name and email from header."""
        name, addr = email.utils.parseaddr(from_header)
        if not name:
            name = addr.split('@')[0]
        return name, addr
    
    def _is_new_candidate(self, email):
        """Check if this is a new candidate."""
        # In a real app, check database
        # For now, assume all emails are from new candidates
        return True
    
    def _process_candidate_email(self, email_message, name, email_addr):
        """Process email from a new candidate."""
        logger.info(f"Processing email from {name} <{email_addr}>")
        
        # Extract resume text and phone number
        resume_text, phone_number = self._extract_resume_text(email_message)
        
        if not resume_text:
            logger.warning("No resume content found in email")
            return
            
        if not phone_number:
            logger.warning(f"No phone number found in resume for {name}")
            # Continue anyway, as we can still schedule the interview
        
        # Save resume to file for RAG processing
        resume_saved = self._save_resume(email_message, name, email_addr)
        
        # Process resume with RAG if saved successfully
        if resume_saved:
            try:
                # Process the resume with RAG
                metadata = {
                    'candidate_name': name,
                    'email': email_addr,
                    'phone': phone_number or 'Not provided',
                    'source': 'email',
                    'timestamp': datetime.now().isoformat()
                }
                
                # Process the saved resume file
                self.rag_processor.process_document(resume_saved, metadata=metadata)
                logger.info(f"Successfully processed resume for {name} with RAG")
                
            except Exception as e:
                logger.error(f"Error processing resume with RAG: {e}")
        
        # Schedule interview for 2 minutes from now
        interview_time = self.schedule_interview_in_2min(name, email_addr)
        if interview_time:
            logger.info(f"Scheduled interview for {name} at {interview_time}")
            self._send_interview_confirmation(name, email_addr, interview_time)
            if phone_number:
                self._schedule_voice_call(name, email_addr, phone_number, interview_time)
            else:
                logger.warning(f"No phone number available to schedule call for {name}")
    
    def _save_resume(self, email_message, name, email_addr):
        """
        Save resume attachments to disk for RAG processing.
        
        Args:
            email_message: The email message containing the resume
            name (str): Candidate's name
            email_addr (str): Candidate's email address
            
        Returns:
            str: Path to the saved resume file, or None if not saved
        """
        try:
            # Create a safe filename from the candidate's name and email
            safe_name = re.sub(r'[^\w\s-]', '', name).strip().replace(' ', '_')
            safe_email = email_addr.split('@')[0]  # Use part before @ for filename
            
            if email_message.is_multipart():
                for part in email_message.walk():
                    content_disposition = str(part.get("Content-Disposition"))
                    
                    # Look for attachments
                    if "attachment" in content_disposition:
                        filename = part.get_filename()
                        if filename:
                            try:
                                # Get file extension
                                file_ext = os.path.splitext(filename)[1].lower()
                                if file_ext not in ['.pdf', '.doc', '.docx', '.txt']:
                                    continue  # Skip non-resume files
                                
                                # Create a unique filename
                                timestamp = int(time.time())
                                save_filename = f"{safe_name}_{safe_email}_{timestamp}{file_ext}"
                                save_path = os.path.join('database', 'resumes', save_filename)
                                
                                # Save the file
                                file_data = part.get_payload(decode=True)
                                with open(save_path, 'wb') as f:
                                    f.write(file_data)
                                
                                logger.info(f"Saved resume to {save_path}")
                                return save_path
                                
                            except Exception as e:
                                logger.error(f"Error saving attachment {filename}: {e}")
                                continue
            
            # If no attachments, save the email body as text
            try:
                # Create a unique filename
                timestamp = int(time.time())
                save_filename = f"{safe_name}_{safe_email}_{timestamp}.txt"
                save_path = os.path.join('database', 'resumes', save_filename)
                
                # Get email body text
                body = ""
                if email_message.is_multipart():
                    for part in email_message.walk():
                        content_type = part.get_content_type()
                        content_disposition = str(part.get("Content-Disposition"))
                        
                        if content_type == "text/plain" and "attachment" not in content_disposition:
                            try:
                                payload = part.get_payload(decode=True)
                                if payload:
                                    body += payload.decode('utf-8', errors='ignore') + "\n\n"
                            except Exception as e:
                                logger.warning(f"Error decoding email part: {e}")
                else:
                    payload = email_message.get_payload(decode=True)
                    if payload:
                        body = payload.decode('utf-8', errors='ignore')
                
                # Save the email body
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(body)
                
                logger.info(f"Saved email body as resume to {save_path}")
                return save_path
                
            except Exception as e:
                logger.error(f"Error saving email body as resume: {e}")
                return None
                
        except Exception as e:
            logger.error(f"Error in _save_resume: {e}")
            return None
    
    def _extract_resume_text(self, email_message):
        """Extract resume text from email attachments."""
        try:
            resume_text = ""
            phone_number = None
            
            # Get the email body
            if email_message.is_multipart():
                for part in email_message.walk():
                    content_type = part.get_content_type()
                    content_disposition = str(part.get("Content-Disposition"))
                    
                    # Get email body text
                    if content_type == "text/plain" and "attachment" not in content_disposition:
                        try:
                            payload = part.get_payload(decode=True)
                            if payload:
                                resume_text += payload.decode('utf-8', errors='ignore') + "\n\n"
                        except Exception as e:
                            logger.warning(f"Error decoding email part: {e}")
                    
                    # Handle attachments
                    elif "attachment" in content_disposition:
                        filename = part.get_filename()
                        if filename:
                            try:
                                file_data = part.get_payload(decode=True)
                                if filename.lower().endswith('.pdf'):
                                    # For PDF files
                                    import PyPDF2
                                    from io import BytesIO
                                    with BytesIO(file_data) as file_stream:
                                        reader = PyPDF2.PdfReader(file_stream)
                                        for page in reader.pages:
                                            resume_text += page.extract_text() + "\n"
                                            
                                elif filename.lower().endswith(('.doc', '.docx')):
                                    # For Word documents
                                    import docx2txt
                                    resume_text += docx2txt.process(BytesIO(file_data)) + "\n"
                                    
                                elif filename.lower().endswith('.txt'):
                                    # For plain text files
                                    resume_text += file_data.decode('utf-8', errors='ignore') + "\n"
                                    
                            except Exception as e:
                                logger.error(f"Error processing attachment {filename}: {e}")
            
            # If no attachments, use the email body
            if not resume_text:
                payload = email_message.get_payload(decode=True)
                if payload:
                    resume_text = payload.decode('utf-8', errors='ignore')
            
            # Extract phone number from the extracted text
            if resume_text:
                phone_number = self._extract_phone_number(resume_text)
            
            return resume_text, phone_number
            
        except Exception as e:
            logger.error(f"Error extracting resume text: {e}")
            return "", None
    
    def get_candidate_insights(self, query: str) -> Dict[str, Any]:
        """
        Get insights about candidates based on a natural language query.
        
        Args:
            query (str): Natural language query about candidates
            
        Returns:
            Dict[str, Any]: Dictionary containing the response and source documents
        """
        try:
            if not query.strip():
                return {
                    'success': False,
                    'error': 'Query cannot be empty',
                    'response': None,
                    'sources': []
                }
            
            logger.info(f"Getting candidate insights for query: {query}")
            
            # Get response from RAG processor
            response = self.rag_processor.get_candidate_insights(query)
            
            # Get relevant documents for sources
            relevant_docs = self.rag_processor.query(query, k=3)
            sources = []
            
            for doc in relevant_docs:
                source = {
                    'content': doc['content'][:500] + '...' if len(doc['content']) > 500 else doc['content'],
                    'candidate': doc['metadata'].get('candidate_name', 'Unknown'),
                    'email': doc['metadata'].get('email', 'Unknown'),
                    'source_file': doc['metadata'].get('source', 'Unknown'),
                    'score': float(doc['score'])
                }
                sources.append(source)
            
            return {
                'success': True,
                'response': response,
                'sources': sources
            }
            
        except Exception as e:
            error_msg = f"Error getting candidate insights: {str(e)}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'response': None,
                'sources': []
            }
    
    def _is_valid_indian_number(self, number):
        """Validate if the number is a valid Indian mobile number."""
        import re
        # Must be 10 digits starting with 6-9, or 12 digits with +91
        return (re.match(r'^[6-9]\d{9}$', number) or 
                re.match(r'^\+91[6-9]\d{9}$', number) or
                re.match(r'^91[6-9]\d{9}$', number))

    def _extract_phone_number(self, text):
        """Extract Indian phone number from text with optimized pattern matching."""
        if not text:
            return None
            
        # Common patterns for Indian phone numbers
        patterns = [
            r'\+91[\s-]?[6-9]\d{9}\b',  # +91 followed by 10 digits
            r'\b[6-9]\d{9}\b',           # Just 10 digits (Indian mobile numbers start with 6-9)
            r'\+91[\s-]?\d{5}[\s-]?\d{5}\b',  # +91 followed by 5-5 digits with separators
            r'\b0?[6-9]\d{9}\b'          # Optional 0 followed by 10 digits
        ]
        
        # Try each pattern
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                phone = match.group(0)
                # Clean the phone number
                clean_phone = re.sub(r'[^0-9+]', '', phone)
                
                # Handle different formats
                if clean_phone.startswith('+91') and len(clean_phone) == 13:
                    pass  # Already in correct format
                elif clean_phone.startswith('91') and len(clean_phone) == 12:
                    clean_phone = f"+{clean_phone}"
                elif len(clean_phone) == 10 and clean_phone[0] in '6789':
                    clean_phone = f"+91{clean_phone}"
                else:
                    continue  # Skip if not matching any format
                
                # Validate the number
                if clean_phone and self._is_valid_indian_number(clean_phone.replace('+', '')):
                    logger.info(f"Found valid Indian number: {clean_phone}")
                    return clean_phone
        
        logger.warning("No valid Indian phone number found in text")
        return None
    
    def _schedule_interview(self, name, email_addr, interview_time=None):
        """Schedule an interview.
        
        Args:
            name (str): Candidate's name
            email_addr (str): Candidate's email address
            interview_time (datetime, optional): Specific time to schedule the interview. 
                                              If None, will use next available slot.
        """
        if interview_time is None:
            if not self.available_slots:
                logger.warning("No available interview slots")
                return None
                
            # Get the next available slot
            interview_time = self.available_slots.pop(0)
        
        # Store the interview
        self.scheduled_interviews[email_addr] = {
            'name': name,
            'time': interview_time,
            'status': 'scheduled'
        }
        
        logger.info(f"Scheduled interview for {name} at {interview_time}")
        return interview_time
        
    def schedule_interview_in_2min(self, name, email_addr):
        """Schedule an interview for 2 minutes from now.
        
        Args:
            name (str): Candidate's name
            email_addr (str): Candidate's email address
            
        Returns:
            datetime: The scheduled interview time
        """
        interview_time = datetime.now(self.timezone) + timedelta(minutes=2)
        return self._schedule_interview(name, email_addr, interview_time)
    
    def _send_interview_confirmation(self, name, email_addr, interview_time):
        """Send interview confirmation email."""
        try:
            # Format the time
            formatted_time = interview_time.strftime("%A, %B %d, %Y at %I:%M %p %Z")
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.email
            msg['To'] = email_addr
            msg['Subject'] = "Interview Confirmation"
            
            # Email body
            body = f"""
            Hello {name},
            
            Thank you for your application! We would like to schedule an interview with you.
            
            Your interview has been scheduled for:
            {formatted_time}
            
            We will call you at the scheduled time.
            
            Best regards,
            Interview Team
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP_SSL(self.smtp_server, 465) as server:
                server.login(self.email, self.password)
                server.send_message(msg)
                
            logger.info(f"Sent interview confirmation to {email_addr}")
            
        except Exception as e:
            logger.error(f"Error sending confirmation email: {e}")
    
    def _schedule_voice_call(self, name, email_addr, phone_number, call_time):
        """Schedule a voice call with the candidate."""
        if not phone_number:
            logger.warning(f"No valid phone number found for {name}. Cannot schedule voice call.")
            return False
            
        logger.info(f"Scheduled voice call with {name} at {phone_number} for {call_time}")
        
        # In a real implementation, you would integrate with a voice calling API here
        # For example, using Twilio or another telephony service
        try:
            # TODO: Implement actual voice call scheduling
            # This is a placeholder for the actual implementation
            logger.info(f"Would schedule call to {phone_number} at {call_time}")
            
            # Store the call details
            call_id = f"{email_addr}_{int(call_time.timestamp())}"
            self.scheduled_interviews[email_addr] = {
                'name': name,
                'phone': phone_number,
                'time': call_time,
                'status': 'scheduled'
            }
            
            # Send confirmation email
            self._send_interview_confirmation(name, email_addr, call_time)
            
            return True
        except Exception as e:
            logger.error(f"Error scheduling voice call: {e}")
            return False
            
    def _send_interview_confirmation(self, name, email_addr, call_time):
        """Send interview confirmation email."""
        try:
            # Format the time
            formatted_time = call_time.strftime("%A, %B %d, %Y at %I:%M %p %Z")
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.email
            msg['To'] = email_addr
            msg['Subject'] = "Interview Confirmation"
            
            # Email body
            body = f"""
            Hello {name},
            
            Thank you for your application! We would like to schedule an interview with you.
            
            Your interview has been scheduled for:
            {formatted_time}
            
            We will call you at the scheduled time.
            
            Best regards,
            Interview Team
            """
            
            msg.attach(MIMEText(body.strip(), 'plain'))
            
            # Send email
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
                server.login(self.email, os.getenv('EMAIL_PASSWORD'))
                server.send_message(msg)
                
            logger.info(f"Confirmation email sent to {name} at {email_addr}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending confirmation email to {email_addr}: {str(e)}")
            return False

    def _make_voice_call(self, name, phone_number, call_time):
        """Make a voice call to the candidate using Twilio.
        
        Args:
            name (str): Candidate's name
            phone_number (str): Candidate's phone number in E.164 format
            call_time (datetime): Scheduled time for the call
            
        Returns:
            bool: True if call was initiated successfully, False otherwise
        """
        try:
            if not phone_number or phone_number == 'No phone number':
                logger.warning(f"No valid phone number provided for {name}. Cannot make call.")
                return False
                
            logger.info(f"Initiating Twilio call to {name} at {phone_number}")
            
            # Log the call details
            call_details = {
                'name': name,
                'phone': phone_number,
                'call_time': datetime.now(self.timezone).strftime('%Y-%m-%d %H:%M:%S %Z'),
                'status': 'initiated'
            }
            logger.info(f"Call details: {call_details}")
            
            # Create a TwiML response for the call
            response = VoiceResponse()
            
            # Start the interview directly with introduction
            response.say(f"Hello {name}, thank you for joining this interview. My name is Alex, and I'll be conducting your interview today. "
                       "This is an automated interview process. I'll ask you a series of questions, and there will be a pause after each question for your response. "
                       "Please speak clearly and take your time to answer each question thoroughly. Let's begin with the first question.", 
                       voice='alice')
            
            # Add a brief pause
            response.pause(length=2)
            
            # Interview questions with pauses for responses
            questions = [
                # Background and experience
                "Can you walk me through your resume and tell me about your most relevant experience for this position?",
                
                # Technical skills
                "What technical skills and programming languages are you most proficient in? Can you give examples of projects where you've used them?",
                
                # Problem-solving
                "Describe a challenging technical problem you faced and how you solved it. What was your approach and what was the outcome?",
                
                # Teamwork and communication
                "Can you tell me about a time you had to work in a team? What was your role and how did you contribute to the team's success?",
                
                # Company and role fit
                "What interests you about this position and our company? Why do you think you'd be a good fit?",
                
                # Problem-solving scenario
                "How would you approach a situation where you're given a task with unclear requirements?",
                
                # Career goals
                "Where do you see yourself in the next 3-5 years, and how does this role align with your career goals?"
            ]
            
            # Add each question with a pause for response
            for i, question in enumerate(questions, 1):
                response.say(f"Question {i}. {question}", voice='alice')
                response.pause(length=20)  # 45 seconds for response
                
                # Add a brief transition between questions
                if i < len(questions):
                    response.say("Thank you for your response. Let's move on to the next question.", voice='alice')
                    response.pause(length=2)
            
            # Closing message
            response.say("Thank you for your time and thoughtful responses. We've come to the end of our interview. "
                       "We will review your answers and be in touch with next steps. Do you have any questions for us?", 
                       voice='alice')
            response.pause(length=30)  # 30 seconds for candidate questions
            response.say("Thank you again for your time. We appreciate your interest in our company and will be in touch soon. Have a great day!", 
                       voice='alice')
            
            # Convert the TwiML to a string
            twiml = str(response)
            
            # Make the call using Twilio
            call = self.twilio_client.calls.create(
                to=phone_number,
                from_=self.twilio_phone_number,
                twiml=twiml
            )
            
            logger.info(f"Call initiated with SID: {call.sid}")
            logger.info(f"Interview call to {name} at {phone_number} is being connected")
            
            return True
            
        except Exception as e:
            logger.error(f"Error making Twilio voice call to {phone_number}: {e}")
            return False
    
    def _check_upcoming_calls(self):
        """Check for and make upcoming calls."""
        try:
            now = datetime.now(self.timezone)
            
            for email_addr, details in list(self.scheduled_interviews.items()):
                if details['status'] != 'scheduled':
                    continue
                    
                # Check if it's time for the call (within 1 minute)
                time_until_call = (details['time'] - now).total_seconds()
                
                if 0 <= time_until_call <= 60:  # Within next minute
                    success = self._make_voice_call(
                        details['name'], 
                        details.get('phone', 'No phone number'),
                        details['time']
                    )
                    if success:
                        details['status'] = 'completed'
            
            return True
        except Exception as e:
            logger.error(f"Error in _check_upcoming_calls: {e}")
            return False
            
    def run(self):
        """Run the interview manager."""
        logger.info("Starting Simple Interview Manager")
        
        try:
            while True:
                try:
                    start_time = time.time()
                    
                    # Check for new emails
                    logger.debug("Checking for new emails...")
                    self.check_emails()
                    
                    # Check for upcoming calls
                    self._check_upcoming_calls()
                    
                    # Calculate sleep time to maintain 5-second interval
                    elapsed = time.time() - start_time
                    sleep_time = max(0, 5 - elapsed)  # Ensure we don't sleep negative time
                    
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                        
                except KeyboardInterrupt:
                    logger.info("Shutting down...")
                    break
                except Exception as e:
                    logger.error(f"Error in main loop: {e}", exc_info=True)
                    time.sleep(5)  # Wait 5 seconds before retrying on error
                    
        except Exception as e:
            logger.critical(f"Fatal error in run loop: {e}", exc_info=True)
        finally:
            logger.info("Interview manager stopped")
    
    def _check_upcoming_calls(self):
        """Check for and make upcoming calls."""
        try:
            now = datetime.now(self.timezone)
            
            for email_addr, details in list(self.scheduled_interviews.items()):
                if details['status'] != 'scheduled':
                    continue
                    
                # Check if it's time for the call (within 1 minute)
                time_until_call = (details['time'] - now).total_seconds()
                
                if 0 <= time_until_call <= 60:  # Within next minute
                    success = self._make_voice_call(
                        details['name'], 
                        details.get('phone', 'No phone number'),
                        details['time']
                    )
                    if success:
                        details['status'] = 'completed'
            
            return True
        except Exception as e:
            logger.error(f"Error in _check_upcoming_calls: {e}")
            return False
            
    def _send_interview_confirmation(self, name, email_addr, call_time):
        """Send interview confirmation email."""
        try:
            # Format the time
            formatted_time = call_time.strftime("%A, %B %d, %Y at %I:%M %p %Z")
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.email
            msg['To'] = email_addr
            msg['Subject'] = "Interview Confirmation"
            
            # Email body
            body = f"""
            Hello {name},
            
            Thank you for your application! We would like to schedule an interview with you.
            
            Your interview has been scheduled for:
            {formatted_time}
            
            We will call you at the scheduled time.
            
            Best regards,
            Interview Team
            """
            
            msg.attach(MIMEText(body.strip(), 'plain'))
            
            # Send email
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
                server.login(self.email, os.getenv('EMAIL_PASSWORD'))
                server.send_message(msg)
                
            logger.info(f"Confirmation email sent to {name} at {email_addr}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending confirmation email to {email_addr}: {str(e)}")
            return False

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    # Create and run the manager
    manager = SimpleInterviewManager()
    manager.run()
