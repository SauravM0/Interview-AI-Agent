# Recruitment AI System

A comprehensive AI-powered recruitment system that automates the entire recruitment workflow from resume collection to interview scheduling, conducting phone interviews, and generating candidate reports.

## Features

- **Email Monitoring**: Automatically monitors Gmail for incoming job applications with resume attachments
- **Resume Analysis**: Uses local Ollama LLM to extract information and score candidates
- **Interview Scheduling**: Manages interview time slots and handles scheduling via email
- **Automated Interviews**: Conducts phone interviews using Twilio IVR and analyzes responses
- **Reporting**: Generates detailed candidate reports and recommendations

## System Architecture

The system consists of several modules:

- **Email Processor**: Handles email monitoring, parsing, and sending
- **Resume Analyzer**: Extracts information from resumes and scores candidates
- **Scheduler**: Manages interview time slots and scheduling
- **IVR System**: Conducts phone interviews using Twilio
- **Reporting**: Generates candidate reports and recommendations

## Setup Instructions

### Prerequisites

- Python 3.8+
- Ollama running locally (https://ollama.ai)
- Gmail account for email monitoring
- Twilio account for phone interviews

### Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd recruitment-ai
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Configure Ollama:
   - Install Ollama from https://ollama.ai
   - Pull the LLama3 model: `ollama pull llama3`
   - Ensure Ollama is running on the default port (11434)

4. Configure the system:
   - Update `config/settings.py` with your Gmail credentials
   - Update `config/settings.py` with your Twilio credentials
   - Adjust job positions and requirements in `config/settings.py`
   - Modify email templates as needed

5. Configure webhook URL for Twilio:
   - Open `main.py` and update the `webhook_base_url` with your public URL
   - You can use ngrok for local development: `ngrok http 5000`

### Running the System

1. Start the system:
   ```
   python main.py
   ```

2. The system will:
   - Check for new resumes every 15 minutes
   - Process email replies every 30 minutes
   - Send interview reminders at 9:00 AM daily
   - Initiate scheduled interviews automatically
   - Generate reports at 6:00 PM daily

3. Monitor the system through the log file: `recruitment_ai.log`

## Security Notes

- Keep your credentials secure
- Consider using environment variables for sensitive information
- Regularly rotate your API tokens
- Be mindful of data privacy regulations when storing candidate information

## Customization

### Email Templates

Email templates can be customized in `config/settings.py` under the `EMAIL_TEMPLATES` dictionary.

### LLM Prompts

LLM prompts for resume analysis, interview questions, etc. can be customized in `config/prompts.py`.

### Job Positions

Job positions, required skills, and preferred skills can be modified in `config/settings.py` under the `JOB_POSITIONS` dictionary.

## Troubleshooting

- Check the log file `recruitment_ai.log` for detailed error messages
- Ensure Ollama is running locally
- Verify Gmail and Twilio credentials
- Check that your webhook URL is publicly accessible

## License

MIT License

## Acknowledgements

- Ollama for providing a great local LLM
- Twilio for their robust IVR API
- All the open-source libraries used in this project 