# Streamlit Deployment Guide

## Financial Data Analytics Dashboard

### Table of Contents

1. [Local Deployment](#local-deployment)
2. [Streamlit Cloud Deployment](#streamlit-cloud-deployment)
3. [Heroku Deployment](#heroku-deployment)
4. [Other Hosting Options](#other-hosting-options)
5. [Troubleshooting](#troubleshooting)

---

## Local Deployment

### Prerequisites

- Python 3.8 or higher
- Git installed
- All required packages in `requirements.txt`

### Steps

1. **Clone Repository**

```bash
git clone https://github.com/SwatantraKasliwal/financial-data-analytics.git
cd financial-data-analytics
```

2. **Create Virtual Environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**

```bash
pip install -r requirements.txt
```

4. **Run Streamlit App**

```bash
cd streamlit_app
streamlit run app.py
```

5. **Access Dashboard**

- Open browser and go to `http://localhost:8501`
- Dashboard will be available for local access

---

## Streamlit Cloud Deployment

### Prerequisites

- GitHub account
- Streamlit Cloud account (free at share.streamlit.io)
- Repository pushed to GitHub

### Steps

1. **Prepare Repository**

   - Ensure all files are in the repository
   - Check that `requirements.txt` is in the root directory
   - Verify `streamlit_app/app.py` is the main file

2. **Deploy on Streamlit Cloud**

   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Connect your GitHub account
   - Select repository: `SwatantraKasliwal/financial-data-analytics`
   - Set main file path: `streamlit_app/app.py`
   - Click "Deploy"

3. **Configuration**

   - Streamlit will automatically detect `requirements.txt`
   - App will be available at: `https://[app-name].streamlit.app`

4. **Custom Domain (Optional)**
   - Configure custom domain in Streamlit Cloud settings
   - Add CNAME record in your DNS settings

### Expected URL Format

```
https://financial-analytics-dashboard.streamlit.app
```

---

## Heroku Deployment

### Prerequisites

- Heroku account
- Heroku CLI installed
- Git repository

### Required Files

1. **Procfile** (create in root directory)

```bash
web: sh setup.sh && streamlit run streamlit_app/app.py --server.port=$PORT --server.address=0.0.0.0
```

2. **setup.sh** (create in root directory)

```bash
mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"your-email@domain.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
```

3. **runtime.txt** (create in root directory)

```
python-3.9.18
```

### Deployment Steps

1. **Login to Heroku**

```bash
heroku login
```

2. **Create Heroku App**

```bash
heroku create financial-analytics-dashboard
```

3. **Configure Git Remote**

```bash
git remote add heroku https://git.heroku.com/financial-analytics-dashboard.git
```

4. **Deploy to Heroku**

```bash
git add .
git commit -m "Deploy to Heroku"
git push heroku main
```

5. **Open App**

```bash
heroku open
```

---

## Alternative Hosting Options

### 1. Railway

1. **Connect Repository**

   - Go to [railway.app](https://railway.app)
   - Connect GitHub repository
   - Select the financial-data-analytics repo

2. **Configure Deployment**

   - Set start command: `streamlit run streamlit_app/app.py --server.port=$PORT`
   - Railway will auto-detect Python and install requirements

3. **Custom Domain**
   - Configure custom domain in Railway dashboard

### 2. Render

1. **Create Web Service**

   - Go to [render.com](https://render.com)
   - Create new "Web Service"
   - Connect GitHub repository

2. **Configuration**
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run streamlit_app/app.py --server.port=$PORT --server.address=0.0.0.0`

### 3. DigitalOcean App Platform

1. **Create App**

   - Go to DigitalOcean App Platform
   - Connect GitHub repository

2. **App Spec Configuration**

```yaml
name: financial-analytics
services:
  - name: web
    source_dir: /
    github:
      repo: SwatantraKasliwal/financial-data-analytics
      branch: main
    run_command: streamlit run streamlit_app/app.py --server.port=$PORT --server.address=0.0.0.0
    environment_slug: python
    instance_count: 1
    instance_size_slug: basic-xxs
    http_port: 8080
```

---

## Environment Variables (if needed)

For production deployments, you might need environment variables:

```python
# In app.py, add at the top:
import os

# Example usage:
DATABASE_URL = os.getenv('DATABASE_URL', 'default_value')
API_KEY = os.getenv('API_KEY', 'default_key')
```

Set environment variables in your hosting platform:

- **Streamlit Cloud**: App settings → Environment variables
- **Heroku**: `heroku config:set VAR_NAME=value`
- **Railway**: Environment tab in dashboard
- **Render**: Environment tab in service settings

---

## Optimization for Production

### 1. Performance Optimizations

```python
# Add to app.py
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_large_dataset():
    # Your data loading logic
    pass

# Reduce memory usage
@st.cache_data(max_entries=3)
def process_data(df):
    # Data processing logic
    pass
```

### 2. Error Handling

```python
# Add comprehensive error handling
try:
    df = load_data()
    if df is None:
        st.error("Failed to load data. Please check data files.")
        st.stop()
except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.info("Please contact support if the issue persists.")
```

### 3. Secrets Management

Create `.streamlit/secrets.toml` for sensitive data:

```toml
[database]
username = "your_username"
password = "your_password"
url = "your_database_url"

[api]
key = "your_api_key"
```

Access in app:

```python
import streamlit as st

# Access secrets
db_username = st.secrets["database"]["username"]
api_key = st.secrets["api"]["key"]
```

---

## Monitoring and Analytics

### 1. Add Usage Analytics

```python
# Add to app.py
import time
from datetime import datetime

# Track page views
if 'page_views' not in st.session_state:
    st.session_state.page_views = 0
st.session_state.page_views += 1

# Log usage (optional)
def log_usage(page_name):
    timestamp = datetime.now().isoformat()
    # Log to file or database
    pass
```

### 2. Performance Monitoring

```python
# Add performance tracking
@st.cache_data
def monitor_performance():
    start_time = time.time()
    # Your code here
    end_time = time.time()
    execution_time = end_time - start_time
    return execution_time
```

---

## Custom Domain Setup

### 1. For Streamlit Cloud

- Go to app settings
- Add custom domain
- Update DNS CNAME record: `CNAME @ [app-name].streamlit.app`

### 2. For Heroku

```bash
heroku domains:add www.yourdomain.com
heroku domains:add yourdomain.com
```

Update DNS records:

```
CNAME www [app-name].herokuapp.com
ALIAS @ [app-name].herokuapp.com
```

---

## Security Considerations

### 1. Data Privacy

- Ensure no sensitive data in public repository
- Use environment variables for API keys
- Implement data masking for demo datasets

### 2. Access Control

```python
# Add simple password protection
def check_password():
    def password_entered():
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.error("Password incorrect")
        return False
    else:
        return True

if check_password():
    # Your app content here
    pass
```

---

## Troubleshooting

### Common Issues

1. **Import Errors**

   - Check all dependencies in `requirements.txt`
   - Verify Python version compatibility

2. **Memory Issues**

   - Reduce dataset size for demo
   - Use data sampling for large files
   - Implement efficient caching

3. **Deployment Failures**

   - Check logs in hosting platform
   - Verify file paths are correct
   - Ensure all required files are included

4. **Performance Issues**
   - Use `@st.cache_data` for expensive operations
   - Optimize data loading and processing
   - Reduce number of visualizations per page

### Debug Commands

```bash
# Local debugging
streamlit run app.py --logger.level=debug

# Check requirements
pip list

# Test app locally before deployment
python -m streamlit run streamlit_app/app.py
```

---

## Sharing Your Dashboard

Once deployed, you can share your dashboard:

1. **Direct Link**: Share the deployed URL
2. **QR Code**: Generate QR code for mobile access
3. **Embed**: Use iframe to embed in websites
4. **Social Media**: Share screenshots and link

### Example Sharing Template

```markdown
🚀 Check out my Financial Data Analytics Dashboard!

📊 Interactive analysis of transaction and credit data
🔍 SQL query interface for custom analysis  
📈 Advanced risk scoring and fraud detection
🎯 Built with Python, Streamlit, and Power BI

🔗 Live Demo: https://your-app-url.streamlit.app
📱 Mobile-friendly responsive design
💻 Source Code: https://github.com/SwatantraKasliwal/financial-data-analytics

#DataAnalytics #Python #Streamlit #PowerBI #FinTech
```

---

This deployment guide provides multiple options for hosting your Streamlit dashboard, from free cloud platforms to more advanced hosting solutions. Choose the option that best fits your needs and technical requirements.
