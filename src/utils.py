import pandas as pd
import os
import logging
import configparser
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
from sklearn.linear_model import LinearRegression
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication

def setup_logging(log_dir, log_file):
    """Configure logging with file and console handlers."""
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file)]
    )
    return logging.getLogger(__name__)

def load_config(config_file):
    """Load configuration from a properties file."""
    config = configparser.ConfigParser()
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")
    config.read(config_file)
    return config

def load_data(logger, bank_file, internal_file):
    """Load CSV files into memory."""
    logger.info("Loading daily data from %s", bank_file)
    bank_data = pd.read_csv(bank_file)
    logger.info("Daily data loaded: %d rows, columns: %s", len(bank_data), bank_data.columns.tolist())
    logger.info("Loading historical data from %s", internal_file)
    internal_data = pd.read_csv(internal_file)
    logger.info("Historical data loaded: %d rows, columns: %s", len(internal_data), internal_data.columns.tolist())
    return bank_data, internal_data

def initialize_model():
    """Initialize DistilBERT tokenizer and model."""
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return tokenizer, model, device

def get_anomaly_score(tokenizer, model, device, prompt):
    """Score anomaly using DistilBERT locally."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        score = torch.softmax(logits, dim=1)[0][1].item()
    return score

def train_prediction_model(logger, historical_data):
    """Train a Linear Regression model per account to predict Balance Difference."""
    models = {}
    historical_data['As of Date'] = pd.to_datetime(historical_data['As of Date'])
    historical_data['Days'] = (historical_data['As of Date'] - historical_data['As of Date'].min()).dt.days
    
    for account, group in historical_data.groupby('Account Number'):
        if len(group) < 2:
            continue
        X = group['Days'].values.reshape(-1, 1)
        y = group['Balance Difference'].values
        reg = LinearRegression()
        reg.fit(X, y)
        models[account] = reg
    logger.info("Trained prediction models for %d accounts", len(models))
    return models

def predict_balance(account, date, models, historical_data):
    """Predict Balance Difference for a given account and date."""
    if account not in models:
        hist_mean = historical_data[historical_data['Account Number'] == account]['Balance Difference'].mean()
        return hist_mean
    
    reg = models[account]
    base_date = pd.to_datetime(historical_data['As of Date'].min())
    days = (pd.to_datetime(date) - base_date).days
    predicted = reg.predict([[days]])[0]
    return max(predicted, 0)

def send_email(logger, smtp_config, subject, body, attachment_path=None):
    """Send an email using the provided SMTP configuration, with optional attachment."""
    try:
        msg = MIMEMultipart()
        msg['Subject'] = subject
        msg['From'] = smtp_config['sender_email']
        msg['To'] = smtp_config['recipient_email']
        msg.attach(MIMEText(body, 'plain'))

        # Attach file if provided
        if attachment_path and os.path.exists(attachment_path):
            with open(attachment_path, 'rb') as f:
                attachment = MIMEApplication(f.read(), _subtype="csv")
                attachment.add_header('Content-Disposition', 'attachment', filename=os.path.basename(attachment_path))
                msg.attach(attachment)
            logger.info("Attached file: %s", attachment_path)

        with smtplib.SMTP(smtp_config['smtp_server'], smtp_config['smtp_port']) as server:
            server.starttls()
            server.login(smtp_config['sender_email'], smtp_config['sender_password'])
            server.send_message(msg)
        logger.info("Email sent successfully: %s", subject)
    except Exception as e:
        logger.error("Failed to send email: %s", str(e))