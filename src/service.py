import pandas as pd
import os
from tqdm import tqdm
import numpy as np
from src.utils import (
    setup_logging, load_config, load_data, initialize_model,
    get_anomaly_score, train_prediction_model, predict_balance, send_email
)

class BankDataService:
    def __init__(self, config_file: str):
        """Initialize the service with configuration and model setup."""
        self.config = load_config(config_file)

        # Paths from config.properties
        self.bank_file = self.config['Paths']['bank_file']
        self.internal_file = self.config['Paths']['internal_file']
        self.log_dir = self.config['Paths']['log_dir']
        self.log_file = os.path.join(self.log_dir, self.config['Paths']['log_file'])
        self.output_dir = self.config['Paths']['output_dir']

        # Email configuration
        self.smtp_config = {
            'smtp_server': self.config['Email']['smtp_server'],
            'smtp_port': int(self.config['Email']['smtp_port']),
            'sender_email': self.config['Email']['sender_email'],
            'sender_password': self.config['Email']['sender_password'],
            'recipient_email': self.config['Email']['recipient_email']
        }

        # Validate paths
        if not os.path.exists(self.bank_file):
            raise FileNotFoundError(f"Bank file not found: {self.bank_file}")
        if not os.path.exists(self.internal_file):
            raise FileNotFoundError(f"Internal file not found: {self.internal_file}")
        if not os.path.exists(self.output_dir):
            raise FileNotFoundError(f"Output directory not found: {self.output_dir}")
        if not os.access(self.output_dir, os.W_OK):
            raise PermissionError(f"Output directory is not writable: {self.output_dir}")

        # Setup logging
        self.logger = setup_logging(self.log_dir, self.log_file)

        # Initialize DistilBERT model
        self.tokenizer, self.model, self.device = initialize_model()

    def smart_reconcile(self, bank_data, internal_data):
        """Skip traditional reconciliation since Transaction IDs are unique."""
        self.logger.info("Skipping Transaction ID-based reconciliation as all daily transactions are new.")
        new_transactions = bank_data.copy()
        duplicates = pd.DataFrame(columns=bank_data.columns)
        self.logger.info("Reconciliation: %d new, %d duplicates", len(new_transactions), len(duplicates))
        return new_transactions, duplicates

    def detect_trend_anomalies(self, bank_data, internal_data):
        """Detect trend-specific anomalies with specific explanations."""
        combined_data = pd.concat([internal_data, bank_data]).sort_values(['Account Number', 'As of Date'])
        combined_data['Balance Difference'] = combined_data['Balance Difference'].astype(float)
        combined_data = combined_data.reset_index(drop=True)
        self.logger.info("Combined data: %d rows", len(combined_data))
        
        anomalies = []
        account_groups = combined_data.groupby('Account Number')
        self.logger.info("Processing %d account groups", len(account_groups))
        
        for account, group in tqdm(account_groups, desc="Analyzing trends"):
            group = group.reset_index(drop=True)
            daily_group = bank_data[bank_data['Account Number'] == account]
            self.logger.info("Account %s: %d historical rows, %d daily rows", account, len(group), len(daily_group))
            if daily_group.empty:
                continue
            
            window = 30
            rolling_mean = group['Balance Difference'].rolling(window=window, min_periods=1).mean()
            rolling_std = group['Balance Difference'].rolling(window=window, min_periods=1).std()
            diffs = group['Balance Difference'].diff()
            
            for idx, row in daily_group.iterrows():
                hist_indices = group.index[group['Transaction ID'] == row['Transaction ID']].tolist()
                if not hist_indices:
                    self.logger.warning("Transaction ID %s not found in combined data for account %s", row['Transaction ID'], account)
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
                    score = get_anomaly_score(self.tokenizer, self.model, self.device, prompt)
                    self.logger.info("Transaction %s: Score %.2f, Conditions - Trend Up: %s, Too High: %s, Too Low: %s", 
                                     row['Transaction ID'], score, trend_up, too_high, too_low)
                    if score >= 0.5:
                        if trend_up and not (too_high or too_low):
                            explanation = "Trend up detected"
                        elif too_high:
                            explanation = "Too high detected"
                        elif too_low:
                            explanation = "Too low detected"
                        else:
                            explanation = "Anomaly detected"
                        anomalies.append((row, score, explanation))
                        self.logger.info("Anomaly detected for Transaction %s: Score %.2f, Explanation: %s", 
                                         row['Transaction ID'], score, explanation)
        
        result = pd.DataFrame([a[0] for a in anomalies], columns=bank_data.columns)
        result['Fraud_Score'] = [a[1] for a in anomalies]
        result['Explanation'] = [a[2] for a in anomalies]
        self.logger.info("Detected %d trend-based anomalies", len(result))
        return result

    def smart_correct_anomalies(self, anomalies, bank_data, internal_data):
        """Correct anomalies using model-predicted Balance Difference."""
        corrected = bank_data.copy()
        if 'Status' not in corrected.columns:
            corrected['Status'] = ''
        
        prediction_models = train_prediction_model(self.logger, internal_data)
        
        for idx, row in tqdm(anomalies.iterrows(), total=len(anomalies), desc="Correcting anomalies"):
            prompt = (
                f"Anomaly: Transaction ID {row['Transaction ID']}, Account Number {row['Account Number']}, "
                f"Balance Difference {row['Balance Difference']}, As of Date {row['As of Date']}. "
                f"Explanation: {row['Explanation']}. "
                f"Suggest one of: 'correct balance to [value]', 'correct date to [date]', or 'flag as fraud'."
            )
            score = get_anomaly_score(self.tokenizer, self.model, self.device, prompt)
            
            condition = corrected['Transaction ID'] == row['Transaction ID']
            if "trend up" in row['Explanation'].lower():
                predicted_balance = predict_balance(row['Account Number'],
                                                     row['As of Date'], 
                                                     prediction_models, 
                                                     internal_data)
                response = f"correct balance to {predicted_balance}"
                try:
                    new_amount = float(response.lower().split('correct balance to')[-1].strip())
                    corrected.loc[condition, 'Balance Difference'] = new_amount
                    self.logger.info("Corrected Balance Difference for %s: %s -> %s (predicted)", 
                                     row['Transaction ID'], row['Balance Difference'], new_amount)
                    corrected.loc[condition, 'Status'] = 'Flagged as Fraud and raised incident and sent email to all stakeholders'
                except ValueError:
                    self.logger.warning("Could not parse new amount: %s", response)
            elif "too high" in row['Explanation'].lower() or "too low" in row['Explanation'].lower():
                corrected.loc[condition, 'Status'] = 'Flagged as Fraud and raised incident and sent email to all stakeholders'
                self.logger.warning("Flagged as fraud: %s", row['Transaction ID'])
        
        return corrected

    def process_bank_data(self, request_id: str) -> dict:
        """Process bank data and save results to output_dir."""
        try:
            bank_data, internal_data = load_data(self.logger, self.bank_file, self.internal_file)
            new_transactions, duplicates = self.smart_reconcile(bank_data, internal_data)
            
            anomalies = self.detect_trend_anomalies(bank_data, internal_data)
            anomalies_path = os.path.join(self.output_dir, f"anomalies_{request_id}.csv")
            anomalies.to_csv(anomalies_path, index=False)
            self.logger.info("Anomaly results saved to %s (rows: %d)", anomalies_path, len(anomalies))
            
            corrected_bank_data = self.smart_correct_anomalies(anomalies, bank_data, internal_data)
            corrected_path = os.path.join(self.output_dir, f"corrected_{request_id}.csv")
            corrected_bank_data.to_csv(corrected_path, index=False)
            self.logger.info("Corrected bank data saved to %s", corrected_path)
            
            # Send email with attachment if anomalies were detected and corrected
            if not anomalies.empty:
                subject = "Bank Anomaly Detection: Anomalies Detected and Corrected"
                body = f"Detected and corrected {len(anomalies)} anomalies.\nAnomaly file: {anomalies_path}\nCorrected file attached."
                send_email(self.logger, self.smtp_config, subject, body, attachment_path=corrected_path)
            
            return {
                "anomalies_file": anomalies_path,
                "corrected_file": corrected_path,
                "request_id": request_id
            }
        except Exception as e:
            # Send email on failure
            subject = "Bank Anomaly Detection: Processing Failed"
            body = f"Bank data processing failed with request ID {request_id}.\nError: {str(e)}"
            send_email(self.logger, self.smtp_config, subject, body)
            raise  # Re-raise the exception to be caught by the API layer