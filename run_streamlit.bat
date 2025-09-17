@echo off
cd /d "C:\Users\swata\skill\projects\Data Analytics\Financial Data Analytics\financial-data-analytics\streamlit_app"
echo Current directory: %CD%
echo Files in directory:
dir /b
echo.
echo Starting Streamlit app...
python -m streamlit run app.py --server.port=8501
pause