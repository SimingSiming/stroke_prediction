#!/bin/bash

# Start FastAPI using gunicorn with uvicorn workers
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:8000 &

# Start Streamlit
streamlit run app.py --server.port 8501 --server.address 0.0.0.0 &

# Wait for background processes to finish
wait
