# Move the main app to root level for Streamlit Cloud
import sys
import os

# Add the streamlit_app directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
streamlit_app_dir = os.path.join(current_dir, 'streamlit_app')
sys.path.insert(0, streamlit_app_dir)

# Import and run the main app
from app import main

if __name__ == "__main__":
    main()