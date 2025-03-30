# Create virtual environment 
python3 -m venv wf_echo_anomaly

# Activate environment
source wf_echo_anomaly/bin/activate

# Install packages
poetry install


#test case to run

python -m test.test_app



#start local server

python -m src.app


#client request
curl -X POST http://localhost:9000/process-bank-data/