# 🤖 Recruitment AI System

An AI-powered recruitment platform that **automates the entire hiring workflow** — from resume collection to interview scheduling, phone interviews, and candidate reporting.

---

## 🚀 Features

- **📩 Email Monitoring**  
  Automatically monitors Gmail for incoming job applications with resume attachments.

- **🧠 Resume Analysis**  
  Extracts key information and scores candidates using the local **Ollama LLM**.

- **🗓️ Interview Scheduling**  
  Manages and auto-schedules interviews via email.

- **📞 Automated Phone Interviews**  
  Uses **Twilio IVR** to conduct voice-based interviews and analyze answers.

- **📊 Candidate Reporting**  
  Generates detailed reports and AI-based recommendations.

---

## 🏗️ System Architecture

The system is built with modular design principles:

```
📬 Email Processor
    ↳ Monitors Gmail, parses resumes, and sends responses

🧾 Resume Analyzer
    ↳ Uses local LLM to extract and evaluate candidate data

📆 Scheduler
    ↳ Books interview slots and sends confirmations

📞 IVR System (Twilio)
    ↳ Conducts phone interviews via automated voice prompts

📑 Reporting Module
    ↳ Summarizes candidate performance and generates reports
```

---

## ⚙️ Setup Instructions

### ✅ Prerequisites

- Python 3.8+
- Local Ollama instance → [https://ollama.ai](https://ollama.ai)
- Gmail account (API access enabled)
- Twilio account

---

### 📦 Installation

```bash
git clone <repository-url>
cd recruitment-ai
pip install -r requirements.txt
```

---

### 🔧 Configuration

1. **Ollama Setup**

```bash
ollama pull llama3
```

Make sure Ollama is running on `localhost:11434`.

2. **Edit Configuration File**

Update `config/settings.py` with:

- Gmail credentials
- Twilio credentials
- Job roles and requirements
- Email templates

3. **Twilio Webhook Setup**

- Update `webhook_base_url` in `main.py`
- Use [ngrok](https://ngrok.com/) for local development:

```bash
ngrok http 5000
```

---

## ▶️ Running the System

Start the system with:

```bash
python main.py
```

**Automation Schedule:**

- Resume check: every **15 minutes**
- Email response check: every **30 minutes**
- Interview reminders: **9:00 AM daily**
- Phone interviews: Based on schedule
- Candidate reports: **6:00 PM daily**

Check logs:

```bash
tail -f recruitment_ai.log
```

---

## 🔐 Security Best Practices

- Use environment variables for secrets
- Avoid committing sensitive information
- Rotate API keys regularly
- Follow data privacy regulations (e.g., GDPR, CCPA)

---

## 🛠️ Customization

### Email Templates

Modify in `config/settings.py` → `EMAIL_TEMPLATES`

### LLM Prompts

Customize interview/resume prompts in `config/prompts.py`

### Job Positions

Edit roles and required skills in `config/settings.py` → `JOB_POSITIONS`

---

## 🧰 Troubleshooting

- Check `recruitment_ai.log` for errors
- Verify Ollama is active on port 11434
- Confirm Gmail and Twilio credentials
- Make sure webhook URL is accessible

---

## 📜 License

MIT License

---

## 🙏 Acknowledgements

- [Ollama](https://ollama.ai) – for local LLM
- [Twilio](https://www.twilio.com/) – for IVR and SMS API
- The open-source Python community
