import pandas as pd
import os
import logging
import configparser
from tqdm import tqdm
import numpy as np
from sklearn.linear_model import LinearRegression
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch

# Load configuration
config = configparser.ConfigParser()
config_file = '/home/bhuv/dev_code/GenAI-FastAPI-K8s/src/config.properties'
if not os.path.exists(config_file):
    raise FileNotFoundError(f"Config file not found: {config_file}")
config.read(config_file)

BANK_FILE = config['Paths']['bank_file']
INTERNAL_FILE = config['Paths']['internal_file']
LOG_DIR = config['Paths']['log_dir']
LOG_FILE = os.path.join(LOG_DIR, config['Paths']['log_file'])
OUTPUT_DIR = config['Paths']['output_dir']
OUTPUT_CSV = os.path.join(OUTPUT_DIR, config['Paths']['output_csv'])
CORRECTED_CSV = os.path.join(OUTPUT_DIR, 'corrected_bank_data.csv')
TRAINED_MODEL_DIR = config['Paths']['train_model']

# Configure logging
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_FILE)]
)
logger = logging.getLogger(__name__)

# Initialize tokenizer and model globally
tokenizer = None
model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data(bank_file, internal_file):
    """Load CSV files into memory."""
    logger.info("Loading daily data from %s", bank_file)
    bank_data = pd.read_csv(bank_file)
    logger.info("Loading historical data from %s", internal_file)
    internal_data = pd.read_csv(internal_file)
    return bank_data, internal_data

def generate_training_data(bank_data, internal_data):
    """Generate pseudo-labeled training data using statistical rules."""
    combined_data = pd.concat([internal_data, bank_data]).sort_values(['Account Number', 'As of Date'])
    combined_data['Balance Difference'] = combined_data['Balance Difference'].astype(float)
    combined_data = combined_data.reset_index(drop=True)
    
    prompts = []
    labels = []
    account_groups = combined_data.groupby('Account Number')
    
    for account, group in account_groups:
        group = group.reset_index(drop=True)
        window = 30
        rolling_mean = group['Balance Difference'].rolling(window=window, min_periods=1).mean()
        rolling_std = group['Balance Difference'].rolling(window=window, min_periods=1).std()
        diffs = group['Balance Difference'].diff()
        
        for idx, row in group.iterrows():
            current_pos = idx
            past_mean = rolling_mean.iloc[max(0, current_pos - window):current_pos + 1].iloc[-1]
            past_std = rolling_std.iloc[max(0, current_pos - window):current_pos + 1].iloc[-1]
            trend_up = (diffs.iloc[max(0, current_pos - 5):current_pos + 1] > 0).all() if current_pos >= 5 else False
            too_high = row['Balance Difference'] > (past_mean + 3 * past_std)
            too_low = row['Balance Difference'] < (past_mean - 3 * past_std)
            
            prompt = (
                f"Account Number {account}, Transaction ID {row['Transaction ID']}, "
                f"Balance Difference {row['Balance Difference']}, As of Date {row['As of Date']}. "
                f"History: Mean {past_mean:.2f}, Std {past_std:.2f}, Last 5 diffs positive: {trend_up}. "
                f"Conditions: Trend Up: {trend_up}, Too High: {too_high}, Too Low: {too_low}. "
                f"Is this anomalous? Provide a score (0-1) and explanation."
            )
            
            label = 1 if (trend_up or too_high or too_low) else 0
            prompts.append(prompt)
            labels.append(label)
    
    train_df = pd.DataFrame({'prompt': prompts, 'label': labels})
    logger.info("Generated %d training samples", len(train_df))
    return train_df

def train_distilbert(train_df):
    """Fine-tune DistilBERT on the generated training data."""
    global tokenizer, model  # Declare global variables at the start of the function
    if tokenizer is None or model is None:
        # Load from local directory if it exists, otherwise use base model and train
        if os.path.exists(TRAINED_MODEL_DIR):
            tokenizer = DistilBertTokenizer.from_pretrained(TRAINED_MODEL_DIR)
            model = DistilBertForSequenceClassification.from_pretrained(TRAINED_MODEL_DIR)
            logger.info("Loaded pre-trained DistilBERT from %s", TRAINED_MODEL_DIR)
        else:
            tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
            logger.info("Initialized DistilBERT from 'distilbert-base-uncased' for training")

    model.to(device)

    dataset = Dataset.from_pandas(train_df)
    def tokenize_function(examples):
        return tokenizer(examples['prompt'], padding='max_length', truncation=True, max_length=512)
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.rename_column('label', 'labels')
    tokenized_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    train_test_split = tokenized_dataset.train_test_split(test_size=0.2)
    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']
    
    training_args = TrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, 'training_output'),
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=os.path.join(OUTPUT_DIR, 'logs'),
        logging_steps=10,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    if not os.path.exists(TRAINED_MODEL_DIR):
        logger.info("Starting DistilBERT training...")
        trainer.train()
        model.save_pretrained(TRAINED_MODEL_DIR)
        tokenizer.save_pretrained(TRAINED_MODEL_DIR)
        logger.info("Trained model saved to %s", TRAINED_MODEL_DIR)
    else:
        logger.info("Skipping training as model already exists at %s", TRAINED_MODEL_DIR)

def smart_reconcile(bank_data, internal_data):
    """Skip traditional reconciliation since Transaction IDs are unique."""
    logger.info("Skipping Transaction ID-based reconciliation as all daily transactions are new.")
    new_transactions = bank_data.copy()
    duplicates = pd.DataFrame(columns=bank_data.columns)
    logger.info("Reconciliation: %d new, %d duplicates", len(new_transactions), len(duplicates))
    return new_transactions, duplicates

def get_anomaly_score(prompt):
    """Score anomaly using the trained DistilBERT model."""
    global tokenizer, model
    if tokenizer is None or model is None:
        raise ValueError("Tokenizer or model not initialized. Run train_distilbert first.")
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        score = torch.softmax(logits, dim=1)[0][1].item()
    return score

def train_prediction_model(historical_data):
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

def detect_trend_anomalies(bank_data, internal_data):
    """Detect trend-specific anomalies with specific explanations."""
    combined_data = pd.concat([internal_data, bank_data]).sort_values(['Account Number', 'As of Date'])
    combined_data['Balance Difference'] = combined_data['Balance Difference'].astype(float)
    combined_data = combined_data.reset_index(drop=True)
    
    anomalies = []
    account_groups = combined_data.groupby('Account Number')
    
    for account, group in tqdm(account_groups, desc="Analyzing trends"):
        group = group.reset_index(drop=True)
        daily_group = bank_data[bank_data['Account Number'] == account]
        if daily_group.empty:
            continue
        
        window = 30
        rolling_mean = group['Balance Difference'].rolling(window=window, min_periods=1).mean()
        rolling_std = group['Balance Difference'].rolling(window=window, min_periods=1).std()
        diffs = group['Balance Difference'].diff()
        
        for idx, row in daily_group.iterrows():
            hist_indices = group.index[group['Transaction ID'] == row['Transaction ID']].tolist()
            if not hist_indices:
                continue
            hist_idx = hist_indices[0]
            
            current_pos = hist_idx
            past_mean = rolling_mean.iloc[max(0, current_pos - window):current_pos + 1].iloc[-1]
            past_std = rolling_std.iloc[max(0, current_pos - window):current_pos + 1].iloc[-1]
            trend_up = (diffs.iloc[max(0, current_pos - 5):current_pos + 1] > 0).all()
            too_high = row['Balance Difference'] > (past_mean + 3 * past_std)
            too_low = row['Balance Difference'] < (past_mean - 3 * past_std)
            
            if trend_up or too_high or too_low:
                prompt = (
                    f"Account Number {account}, Transaction ID {row['Transaction ID']}, "
                    f"Balance Difference {row['Balance Difference']}, As of Date {row['As of Date']}. "
                    f"History: Mean {past_mean:.2f}, Std {past_std:.2f}, Last 5 diffs positive: {trend_up}. "
                    f"Conditions: Trend Up: {trend_up}, Too High: {too_high}, Too Low: {too_low}. "
                    f"Is this anomalous? Provide a score (0-1) and explanation."
                )
                score = get_anomaly_score(prompt)
                if trend_up and not (too_high or too_low):
                    explanation = "Trend up detected"
                elif too_high:
                    explanation = "Too high detected"
                elif too_low:
                    explanation = "Too low detected"
                else:
                    explanation = "Anomaly detected"
                if score >= 0.5:
                    anomalies.append((row, score, explanation))
    
    result = pd.DataFrame([a[0] for a in anomalies], columns=bank_data.columns)
    result['Fraud_Score'] = [a[1] for a in anomalies]
    result['Explanation'] = [a[2] for a in anomalies]
    logger.info("Detected %d trend-based anomalies", len(result))
    return result

def smart_correct_anomalies(anomalies, bank_data, internal_data):
    """Correct anomalies using model-predicted Balance Difference."""
    corrected = bank_data.copy()
    if 'Status' not in corrected.columns:
        corrected['Status'] = ''
    
    prediction_models = train_prediction_model(internal_data)
    
    for idx, row in tqdm(anomalies.iterrows(), total=len(anomalies), desc="Correcting anomalies"):
        prompt = (
            f"Anomaly: Transaction ID {row['Transaction ID']}, Account Number {row['Account Number']}, "
            f"Balance Difference {row['Balance Difference']}, As of Date {row['As of Date']}. "
            f"Explanation: {row['Explanation']}. "
            f"Suggest one of: 'correct balance to [value]', 'correct date to [date]', or 'flag as fraud'."
        )
        score = get_anomaly_score(prompt)
        
        condition = corrected['Transaction ID'] == row['Transaction ID']
        if "trend up" in row['Explanation'].lower():
            predicted_balance = predict_balance(row['Account Number'], row['As of Date'], prediction_models, internal_data)
            response = f"correct balance to {predicted_balance}"
            try:
                new_amount = float(response.lower().split('correct balance to')[-1].strip())
                corrected.loc[condition, 'Balance Difference'] = new_amount
                logger.info("Corrected Balance Difference for %s: %s -> %s (predicted)", 
                            row['Transaction ID'], row['Balance Difference'], new_amount)
            except ValueError:
                logger.warning("Could not parse new amount: %s", response)
        elif "too high" in row['Explanation'].lower() or "too_low" in row['Explanation'].lower():
            corrected.loc[condition, 'Status'] = 'Flagged as Fraud'
            logger.warning("Flagged as fraud: %s", row['Transaction ID'])
    
    return corrected

if __name__ == "__main__":
    try:
        bank_data, internal_data = load_data(BANK_FILE, INTERNAL_FILE)
        
        # Load or train DistilBERT model
        train_df = generate_training_data(bank_data, internal_data)
        train_distilbert(train_df)  # This will load the local model if available or train a new one
        
        new_transactions, duplicates = smart_reconcile(bank_data, internal_data)
        
        anomalies = detect_trend_anomalies(bank_data, internal_data)
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        anomalies.to_csv(OUTPUT_CSV, index=False)
        logger.info("Anomaly results saved to %s", OUTPUT_CSV)
        
        corrected_bank_data = smart_correct_anomalies(anomalies, bank_data, internal_data)
        corrected_bank_data.to_csv(CORRECTED_CSV, index=False)
        logger.info("Corrected bank data saved to %s", CORRECTED_CSV)

    except Exception as e:
        logger.error("Program failed: %s", e)