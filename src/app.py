from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from uuid import uuid4
import os

from src.service import BankDataService
from src.utils import send_email


# Pydantic model for optional input data
class ProcessRequest(BaseModel):
    bank_file: str | None = None  # Optional bank file path
    internal_file: str | None = None  # Optional internal file path


# FastAPI app initialization
app = FastAPI(title="Bank Anomaly Detection API")

# Instantiate the service with default config
config_file = os.path.join(os.path.dirname(__file__), 'config.properties')
service = BankDataService(config_file=config_file)


@app.post("/process-bank-data/", response_class=JSONResponse)
def process_bank_data_api(request: ProcessRequest | None = None):
    """Process bank data using file paths from config or request, and save results to output_dir."""
    try:
        request_id = str(uuid4())
        
        # Override service file paths if provided in the request
        if request:
            if request.bank_file:
                service.bank_file = request.bank_file
            if request.internal_file:
                service.internal_file = request.internal_file
        
        result = service.process_bank_data(request_id)
        return JSONResponse(content=result)
    except Exception as e:
        service.logger.error("Processing failed: %s", str(e))
        # Send email on API-level failure
        subject = "Bank Anomaly Detection API: Failure"
        body = f"API request failed with request ID {request_id}.\nError: {str(e)}"
        send_email(service.logger, service.smtp_config, subject, body)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)