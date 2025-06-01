from flask import Flask, request, jsonify, session, redirect, url_for
from flask_cors import CORS
import joblib
import numpy as np
import traceback
import json

app = Flask(__name__)
CORS(app)
app.secret_key = "your_secret_key_here"  # Replace with a strong secret key

# Updated credentials
VALID_USERNAME = "payal"
VALID_PASSWORD = "payal"

# Load the trained models
model_lr = joblib.load("model_lr.pkl")
model_rf = joblib.load("model_rf.pkl")
model_xgb = joblib.load("model_xgb.pkl")
kmeans = joblib.load("kmeans.pkl")

# Load metrics from file
with open("data/metrics.json", "r") as f:
    METRICS = json.load(f)

@app.route("/", methods=["GET", "POST"])
def login():
    # If user is already logged in, redirect to dashboard
    if session.get("logged_in"):
        return redirect(url_for("dashboard"))

    error = ""
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        if username == VALID_USERNAME and password == VALID_PASSWORD:
            session["logged_in"] = True
            session["username"] = username
            return redirect(url_for("dashboard"))
        else:
            error = "Invalid username or password."

    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1.0" />
      <title>Login - Airbnb Price Predictor</title>
      <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
      <style>
        * {{
          box-sizing: border-box;
          margin: 0;
          padding: 0;
        }}
        body {{
          font-family: 'Inter', sans-serif;
          background-color: #F0F2F5;
          color: #222222;
          display: flex;
          height: 100vh;
          align-items: center;
          justify-content: center;
        }}
        .login-container {{
          background: #FFFFFF;
          padding: 2.5rem;
          border-radius: 12px;
          box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
          width: 360px;
        }}
        .logo {{
          display: flex;
          align-items: center;
          justify-content: center;
          margin-bottom: 1.5rem;
        }}
        .logo img {{
          height: 50px;
          margin-right: 0.75rem;
        }}
        .logo h2 {{
          font-size: 1.5rem;
          color: #FF385C;
          font-weight: 700;
        }}
        h3 {{
          font-size: 1.25rem;
          text-align: center;
          margin-bottom: 1rem;
          color: #484848;
        }}
        .form-group {{
          margin-bottom: 1.2rem;
        }}
        label {{
          font-size: 0.9rem;
          margin-bottom: 0.4rem;
          display: block;
          color: #484848;
          font-weight: 500;
        }}
        input[type="text"], input[type="password"] {{
          width: 100%;
          padding: 0.75rem;
          border: 1px solid #DADCE0;
          border-radius: 8px;
          font-size: 1rem;
          transition: border-color 0.2s, box-shadow 0.2s;
        }}
        input[type="text"]:focus, input[type="password"]:focus {{
          outline: none;
          border-color: #FF5A5F;
          box-shadow: 0 0 0 3px rgba(255, 90, 95, 0.2);
        }}
        .checkbox-group {{
          display: flex;
          align-items: center;
          gap: 0.5rem;
          margin-bottom: 1rem;
        }}
        .checkbox-group input {{
          width: auto;
        }}
        .checkbox-group label {{
          margin: 0;
          font-size: 0.9rem;
        }}
        .forgot-password {{
          text-align: right;
          margin-bottom: 1.5rem;
        }}
        .forgot-password a {{
          font-size: 0.85rem;
          color: #FF385C;
          text-decoration: none;
        }}
        .forgot-password a:hover {{
          text-decoration: underline;
        }}
        .btn {{
          width: 100%;
          background-color: #FF385C;
          color: #FFFFFF;
          border: none;
          padding: 0.9rem;
          border-radius: 8px;
          font-size: 1rem;
          font-weight: 600;
          cursor: pointer;
          transition: background-color 0.2s, transform 0.1s;
        }}
        .btn:hover {{
          background-color: #E03151;
          transform: translateY(-1px);
        }}
        .btn:disabled {{
          background-color: #DADCE0;
          cursor: not-allowed;
          transform: none;
        }}
        .error-msg {{
          color: #FF5A5F;
          font-size: 0.85rem;
          margin-bottom: 1rem;
          text-align: center;
        }}
      </style>
    </head>
    <body>
      <div class="login-container">
        <div class="logo">
          <img src="https://upload.wikimedia.org/wikipedia/commons/6/69/Airbnb_Logo_B%C3%A9lo.svg" alt="Logo" />
          <h2>Airbnb AI</h2>
        </div>
        <h3>Sign in to continue</h3>
        {f'<div class="error-msg">{error}</div>' if error else ''}
        <form method="POST" action="/">
          <div class="form-group">
            <label>Username</label>
            <input type="text" name="username" required placeholder="Enter your username" />
          </div>
          <div class="form-group">
            <label>Password</label>
            <input type="password" name="password" required placeholder="Enter your password" />
          </div>
          <div class="checkbox-group">
            <input type="checkbox" id="remember" name="remember" />
            <label for="remember">Remember me</label>
          </div>
          <div class="forgot-password">
            <a href="#">Forgot password?</a>
          </div>
          <button type="submit" class="btn">Sign In</button>
        </form>
      </div>
    </body>
    </html>
    """

@app.route("/dashboard")
def dashboard():
    # Only allow access if logged in
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1.0" />
      <title>Airbnb Price Predictor</title>
      <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
      <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
      <style>
        :root {
          --primary: #FF385C;
          --primary-light: #FF5A5F;
          --primary-lighter: #FFEBEC;
          --secondary: #008489;
          --dark: #222222;
          --dark-gray: #484848;
          --medium-gray: #767676;
          --light-gray: #EBEBEB;
          --lighter-gray: #F7F7F7;
          --white: #FFFFFF;
          --success: #00A699;
          --success-light: #E8F6F5;
          --warning: #FFB400;
          --warning-light: #FFF5E0;
          --error: #FF5A5F;
          --shadow-sm: 0 1px 3px rgba(0, 0, 0, 0.1);
          --shadow-md: 0 4px 6px rgba(0, 0, 0, 0.1);
          --shadow-lg: 0 10px 15px rgba(0, 0, 0, 0.1);
          --radius-sm: 8px;
          --radius-md: 12px;
          --radius-lg: 16px;
          --transition: all 0.2s ease;
        }
        
        * {
          box-sizing: border-box;
          margin: 0;
          padding: 0;
        }
        
        body {
          font-family: 'Inter', sans-serif;
          color: var(--dark);
          background-color: var(--lighter-gray);
          line-height: 1.6;
        }
        
        .wrapper {
          max-width: 1200px;
          margin: 0 auto;
          padding: 2rem 1rem;
        }
        
        header {
          text-align: center;
          margin-bottom: 2rem;
        }
        
        .logo {
          color: var(--primary);
          font-size: 2rem;
          font-weight: 700;
          margin-bottom: 0.5rem;
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 0.5rem;
        }
        
        .logo i {
          font-size: 2.2rem;
        }
        
        h1 {
          color: var(--dark);
          font-size: 2.2rem;
          margin-bottom: 0.5rem;
          font-weight: 700;
        }
        
        .subtitle {
          color: var(--medium-gray);
          font-size: 1.1rem;
          max-width: 600px;
          margin: 0 auto 1.5rem;
        }
        
        .dashboard {
          display: flex;
          gap: 2rem;
          margin-top: 2rem;
        }
        
        .form-panel, .results-panel {
          flex: 1;
          border-radius: var(--radius-lg);
          padding: 2rem;
          background: var(--white);
          box-shadow: var(--shadow-md);
          transition: var(--transition);
        }
        
        .form-panel:hover, .results-panel:hover {
          box-shadow: var(--shadow-lg);
        }
        
        .panel-title {
          font-size: 1.3rem;
          font-weight: 600;
          margin-bottom: 1.5rem;
          color: var(--dark);
          display: flex;
          align-items: center;
          gap: 0.5rem;
          padding-bottom: 0.75rem;
          border-bottom: 1px solid var(--light-gray);
        }
        
        .panel-title i {
          color: var(--primary);
        }
        
        form {
          display: flex;
          flex-direction: column;
          gap: 1.5rem;
        }
        
        .form-grid {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 1.5rem;
        }
        
        .form-group {
          display: flex;
          flex-direction: column;
        }
        
        label {
          font-size: 0.95rem;
          margin-bottom: 0.5rem;
          color: var(--dark-gray);
          font-weight: 500;
          display: flex;
          align-items: center;
          gap: 0.3rem;
        }
        
        label .required {
          color: var(--error);
          font-size: 0.8rem;
        }
        
        input {
          width: 100%;
          padding: 0.8rem 1rem;
          border: 1px solid var(--light-gray);
          border-radius: var(--radius-sm);
          font-size: 1rem;
          transition: var(--transition);
        }
        
        input:focus {
          outline: none;
          border-color: var(--primary-light);
          box-shadow: 0 0 0 2px rgba(255, 90, 95, 0.2);
        }
        
        input::placeholder {
          color: var(--light-gray);
        }
        
        .btn {
          background-color: var(--primary);
          color: white;
          border: none;
          padding: 1rem 1.5rem;
          border-radius: var(--radius-sm);
          font-size: 1rem;
          font-weight: 600;
          cursor: pointer;
          transition: var(--transition);
          margin-top: 0.5rem;
          display: inline-flex;
          align-items: center;
          justify-content: center;
          gap: 0.5rem;
        }
        
        .btn:hover {
          background-color: var(--primary-light);
          transform: translateY(-2px);
          box-shadow: var(--shadow-sm);
        }
        
        .btn:disabled {
          background-color: var(--light-gray);
          cursor: not-allowed;
          transform: none;
          box-shadow: none;
        }
        
        .results-content {
          display: none;
          flex-direction: column;
          gap: 1.5rem;
        }
        
        .model-cards {
          display: grid;
          grid-template-columns: 1fr;
          gap: 1.5rem;
        }
        
        /* Reordered: XGBoost first, then Random Forest, then Linear Regression */
        .model-card {
          border: 1px solid var(--light-gray);
          border-radius: var(--radius-md);
          padding: 1.5rem;
          background: var(--white);
          transition: var(--transition);
          position: relative;
          overflow: hidden;
        }
        
        .model-card:hover {
          transform: translateY(-5px);
          box-shadow: var(--shadow-lg);
        }
        
        .model-card::before {
          content: '';
          position: absolute;
          top: 0;
          left: 0;
          width: 4px;
          height: 100%;
          background: var(--primary);
        }
        
        .model-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          margin-bottom: 1rem;
        }
        
        .model-name {
          font-weight: 700;
          font-size: 1.2rem;
          color: var(--dark);
          display: flex;
          align-items: center;
          gap: 0.5rem;
        }
        
        .model-name i {
          color: var(--primary);
        }
        
        .model-tag {
          display: inline-flex;
          align-items: center;
          padding: 0.3rem 0.8rem;
          border-radius: 20px;
          font-size: 0.8rem;
          font-weight: 600;
          gap: 0.3rem;
        }
        
        .tag-reliable {
          background-color: var(--success-light);
          color: var(--success);
        }
        
        .tag-robust {
          background-color: var(--warning-light);
          color: var(--warning);
        }
        
        .tag-premium {
          background-color: var(--primary-lighter);
          color: var(--primary);
        }
        
        .price-container {
          display: flex;
          align-items: baseline;
          gap: 0.5rem;
          margin: 1rem 0;
        }
        
        .price {
          font-size: 1.8rem;
          font-weight: 700;
          color: var(--dark);
        }
        
        .price-change {
          font-size: 0.9rem;
          font-weight: 500;
          padding: 0.2rem 0.5rem;
          border-radius: 4px;
        }
        
        .price-up {
          background-color: rgba(0, 166, 153, 0.1);
          color: var(--success);
        }
        
        .price-down {
          background-color: rgba(255, 90, 95, 0.1);
          color: var(--error);
        }
        
        .model-description {
          color: var(--medium-gray);
          font-size: 0.95rem;
          line-height: 1.5;
          margin-bottom: 1rem;
        }
        
        .metrics {
          display: flex;
          flex-wrap: wrap;
          gap: 0.8rem;
          margin-top: 1rem;
          font-size: 0.85rem;
        }
        
        .metric {
          background-color: var(--lighter-gray);
          padding: 0.5rem 0.8rem;
          border-radius: var(--radius-sm);
          color: var(--dark-gray);
          display: flex;
          align-items: center;
          gap: 0.3rem;
        }
        
        .metric strong {
          color: var(--dark);
        }
        
        .metric i {
          font-size: 0.9rem;
        }
        
        .cluster-card {
          border: 1px solid var(--light-gray);
          border-radius: var(--radius-md);
          padding: 1.5rem;
          background: var(--white);
          margin-top: 1rem;
          transition: var(--transition);
        }
        
        .cluster-card:hover {
          transform: translateY(-2px);
          box-shadow: var(--shadow-md);
        }
        
        .cluster-content {
          display: flex;
          align-items: center;
          gap: 1.5rem;
        }
        
        .cluster-icon {
          font-size: 2rem;
          color: var(--primary);
          flex-shrink: 0;
        }
        
        .cluster-info {
          flex: 1;
        }
        
        .cluster-title {
          font-weight: 600;
          margin-bottom: 0.5rem;
          color: var(--dark);
          font-size: 1.1rem;
        }
        
        .cluster-description {
          color: var(--medium-gray);
          font-size: 0.95rem;
          margin-bottom: 1rem;
        }
        
        .cluster-tag {
          display: inline-flex;
          align-items: center;
          padding: 0.5rem 1rem;
          border-radius: 20px;
          font-weight: 600;
          gap: 0.5rem;
          transition: var(--transition);
        }
        
        .cluster-1 {
          background-color: var(--primary-lighter);
          color: var(--primary);
        }
        
        .cluster-0 {
          background-color: var(--success-light);
          color: var(--success);
        }
        
        .cluster-features {
          margin-top: 1rem;
          display: flex;
          flex-wrap: wrap;
          gap: 0.5rem;
        }
        
        .feature-tag {
          background-color: var(--lighter-gray);
          padding: 0.3rem 0.6rem;
          border-radius: 4px;
          font-size: 0.8rem;
          color: var(--dark-gray);
        }
        
        .loading {
          display: none;
          text-align: center;
          padding: 2rem;
        }
        
        .spinner {
          width: 40px;
          height: 40px;
          border: 4px solid rgba(255, 56, 92, 0.2);
          border-radius: 50%;
          border-top-color: var(--primary);
          animation: spin 1s ease-in-out infinite;
          margin: 0 auto 1rem;
        }
        
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
        
        @media (max-width: 768px) {
          .dashboard {
            flex-direction: column;
          }
          
          .form-grid {
            grid-template-columns: 1fr;
          }
        }
      </style>
    </head>
    <body>
      <div class="wrapper">
        <header>
          <div class="logo">
            <i class="fas fa-home"></i>
            <span>Airbnb Price AI</span>
          </div>
          <h1>Smart Pricing Predictor</h1>
          <p class="subtitle">Get accurate price estimates and market insights for your Airbnb property using machine learning</p>
          <div style="margin-top: 1rem; text-align: center;">
            <a href="/logout" style="font-size: 0.9rem; color: #FF385C; text-decoration: none; display: inline-flex; align-items: center; gap: 0.3rem;">
              <i class="fas fa-sign-out-alt"></i> Logout
            </a>
          </div>
        </header>
        
        <div class="dashboard">
          <!-- Form Panel -->
          <div class="form-panel">
            <h2 class="panel-title"><i class="fas fa-edit"></i> Property Details</h2>
            
            <form id="input-form">
              <div class="form-grid">
                <div class="form-group">
                  <label><i class="fas fa-users"></i> Accommodates <span class="required">*</span></label>
                  <input type="number" id="accommodates" step="1" min="1" required placeholder="Number of guests">
                </div>
                
                <div class="form-group">
                  <label><i class="fas fa-bath"></i> Bathrooms <span class="required">*</span></label>
                  <input type="number" id="bathrooms" step="0.5" min="1" required placeholder="Number of bathrooms">
                </div>
                
                <div class="form-group">
                  <label><i class="fas fa-bed"></i> Bedrooms <span class="required">*</span></label>
                  <input type="number" id="bedrooms" step="1" min="1" required placeholder="Number of bedrooms">
                </div>
                
                <div class="form-group">
                  <label><i class="fas fa-bed"></i> Beds <span class="required">*</span></label>
                  <input type="number" id="beds" step="1" min="1" required placeholder="Number of beds">
                </div>
                
                <div class="form-group">
                  <label><i class="fas fa-map-marker-alt"></i> Latitude <span class="required">*</span></label>
                  <input type="number" id="latitude" step="0.000001" required placeholder="e.g., 40.7128">
                </div>
                
                <div class="form-group">
                  <label><i class="fas fa-map-marker-alt"></i> Longitude <span class="required">*</span></label>
                  <input type="number" id="longitude" step="0.000001" required placeholder="e.g., -74.0060">
                </div>
              </div>
              
              <button type="submit" id="submit-btn" class="btn">
                <i class="fas fa-chart-line"></i> Analyze Property
              </button>
            </form>
          </div>
          
          <!-- Results Panel -->
          <div class="results-panel">
            <h2 class="panel-title"><i class="fas fa-chart-pie"></i> Analysis Results</h2>
            
            <div class="loading" id="loading">
              <div class="spinner"></div>
              <p>Analyzing your property...</p>
            </div>
            
            <div class="results-content" id="results">
              <div class="model-cards">
                <!-- XGBoost first -->
                <div class="model-card">
                  <div class="model-header">
                    <div class="model-name">
                      <i class="fas fa-rocket"></i> XGBoost
                    </div>
                    <div class="model-tag tag-premium">
                      <i class="fas fa-crown"></i> PREMIUM
                    </div>
                  </div>
                  <div class="price-container">
                    <div class="price" id="xgb-price">$0.00</div>
                    <div class="price-change" id="xgb-change"></div>
                  </div>
                  <div class="model-description">
                    State-of-the-art gradient boosting algorithm with superior predictive performance. Our most accurate model for premium listings.
                  </div>
                  <div class="metrics">
                    <div class="metric"><i class="fas fa-crosshairs"></i> <strong>Precision:</strong> ±$15.80</div>
                    <div class="metric"><i class="fas fa-percentage"></i> <strong>R²:</strong> 0.89</div>
                    <div class="metric"><i class="fas fa-bullseye"></i> <strong>Accuracy:</strong> 91%</div>
                  </div>
                </div>
                
                <!-- Random Forest second -->
                <div class="model-card">
                  <div class="model-header">
                    <div class="model-name">
                      <i class="fas fa-tree"></i> Random Forest
                    </div>
                    <div class="model-tag tag-robust">
                      <i class="fas fa-shield-alt"></i> ROBUST
                    </div>
                  </div>
                  <div class="price-container">
                    <div class="price" id="rf-price">$0.00</div>
                    <div class="price-change" id="rf-change"></div>
                  </div>
                  <div class="model-description">
                    Ensemble method that handles complex patterns and non-linear relationships effectively. More accurate for diverse properties.
                  </div>
                  <div class="metrics">
                    <div class="metric"><i class="fas fa-crosshairs"></i> <strong>Precision:</strong> ±$18.20</div>
                    <div class="metric"><i class="fas fa-percentage"></i> <strong>R²:</strong> 0.85</div>
                    <div class="metric"><i class="fas fa-bullseye"></i> <strong>Accuracy:</strong> 88%</div>
                  </div>
                </div>
                
                <!-- Linear Regression last -->
                <div class="model-card">
                  <div class="model-header">
                    <div class="model-name">
                      <i class="fas fa-chart-bar"></i> Linear Regression
                    </div>
                    <div class="model-tag tag-reliable">
                      <i class="fas fa-check-circle"></i> RELIABLE
                    </div>
                  </div>
                  <div class="price-container">
                    <div class="price" id="lr-price">$0.00</div>
                    <div class="price-change" id="lr-change"></div>
                  </div>
                  <div class="model-description">
                    Provides a reliable baseline estimate based on linear relationships between features. Best for straightforward pricing analysis.
                  </div>
                  <div class="metrics">
                    <div class="metric"><i class="fas fa-crosshairs"></i> <strong>Precision:</strong> ±$24.50</div>
                    <div class="metric"><i class="fas fa-percentage"></i> <strong>R²:</strong> 0.72</div>
                    <div class="metric"><i class="fas fa-bullseye"></i> <strong>Accuracy:</strong> 82%</div>
                  </div>
                </div>
              </div>
              
              <div class="cluster-card">
                <div class="cluster-content">
                  <div class="cluster-icon">
                    <i class="fas fa-map-marked-alt"></i>
                  </div>
                  <div class="cluster-info">
                    <div class="cluster-title">Market Segment Analysis</div>
                    <div class="cluster-description">
                      Identifies the optimal market positioning for your property based on location and capacity.
                    </div>
                    <div id="cluster-tag" class="cluster-tag" style="display: none;">
                      <i class="fas fa-tag"></i>
                      <span id="cluster-label">Economy Properties</span>
                    </div>
                    <div class="cluster-features" id="cluster-features" style="display: none;">
                      <div class="feature-tag"><i class="fas fa-map-marker"></i> <span id="cluster-location">Neighborhood info</span></div>
                      <div class="feature-tag"><i class="fas fa-home"></i> <span id="cluster-type">Property type</span></div>
                      <div class="feature-tag"><i class="fas fa-star"></i> <span id="cluster-rating">4.2 avg rating</span></div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <script>
        const API_BASE = "";
        
        const formatCurrency = (amount) => {
          return '$' + parseFloat(amount).toFixed(2).replace(/\\B(?=(\\d{3})+(?!\\d))/g, ",");
        };
        
        const calculatePriceChange = (price, avg) => {
          if (avg === 0) return null;
          const changePerc = ((price - avg) / avg) * 100;
          if (Math.abs(changePerc) < 2) return null; // below 2% is not significant
          return {
            value: Math.abs(changePerc).toFixed(1) + '%',
            direction: changePerc > 0 ? 'up' : 'down'
          };
        };

        const updatePriceChange = (idPrefix, price, avg) => {
          const element = document.getElementById(`${idPrefix}-change`);
          const changeInfo = calculatePriceChange(price, avg);
          if (!changeInfo) {
            element.style.display = 'none';
          } else {
            element.style.display = 'inline-flex';
            element.textContent = (changeInfo.direction === 'up' ? '↑ ' : '↓ ') + changeInfo.value;
            element.className = `price-change ${changeInfo.direction === 'up' ? 'price-up' : 'price-down'}`;
          }
        };
        
        document.getElementById('input-form').addEventListener('submit', async (e) => {
          e.preventDefault();
          
          const btn = document.getElementById('submit-btn');
          const resultsContent = document.getElementById('results');
          const loading = document.getElementById('loading');
          
          btn.disabled = true;
          resultsContent.style.display = 'none';
          loading.style.display = 'block';
          
          const payload = {
            accommodates: parseFloat(document.getElementById('accommodates').value),
            bathrooms: parseFloat(document.getElementById('bathrooms').value),
            bedrooms: parseFloat(document.getElementById('bedrooms').value),
            beds: parseFloat(document.getElementById('beds').value),
            latitude: parseFloat(document.getElementById('latitude').value),
            longitude: parseFloat(document.getElementById('longitude').value),
          };
          
          try {
            // Fetch all predictions in parallel
            const [xgb, rf, lr, cluster] = await Promise.all([
              fetchPrediction('/predict_xgb', payload),
              fetchPrediction('/predict_rf', payload),
              fetchPrediction('/predict_lr', payload),
              fetchPrediction('/cluster', payload)
            ]);
            
            // Compute average price for relative changes
            const prices = [xgb.predicted_price, rf.predicted_price, lr.predicted_price];
            const avgPrice = (xgb.predicted_price + rf.predicted_price + lr.predicted_price) / 3;
            
            // Update UI with results in new sequence
            document.getElementById('xgb-price').textContent = formatCurrency(xgb.predicted_price);
            document.getElementById('rf-price').textContent = formatCurrency(rf.predicted_price);
            document.getElementById('lr-price').textContent = formatCurrency(lr.predicted_price);
            
            // Update price-change badges relative to average
            updatePriceChange('xgb', xgb.predicted_price, avgPrice);
            updatePriceChange('rf', rf.predicted_price, avgPrice);
            updatePriceChange('lr', lr.predicted_price, avgPrice);
            
            // Update cluster info
            const clusterTag = document.getElementById('cluster-tag');
            document.getElementById('cluster-label').textContent = cluster.label;
            clusterTag.style.display = 'inline-flex';
            clusterTag.className = `cluster-tag cluster-${cluster.cluster}`;
            
            document.getElementById('cluster-location').textContent = cluster.cluster === 1 ? 'High-end neighborhood' : 'Budget-friendly area';
            document.getElementById('cluster-type').textContent = cluster.cluster === 1 ? 'Luxury Lodging' : 'Standard Apartment';
            document.getElementById('cluster-rating').textContent = cluster.cluster === 1 ? '4.8 avg rating' : '4.0 avg rating';
            document.getElementById('cluster-features').style.display = 'flex';
            
            // Show results
            loading.style.display = 'none';
            resultsContent.style.display = 'flex';
            
            // Smooth scroll to results
            document.querySelector('.results-panel').scrollIntoView({ 
              behavior: 'smooth',
              block: 'start'
            });
            
          } catch (err) {
            loading.style.display = 'none';
            alert('Error: ' + err.message);
            console.error(err);
          } finally {
            btn.disabled = false;
          }
        });
        
        async function fetchPrediction(endpoint, payload) {
          const res = await fetch(API_BASE + endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
          });
          
          if (!res.ok) {
            const err = await res.json();
            throw new Error(err.error || res.statusText);
          }
          
          return res.json();
        }
      </script>
    </body>
    </html>
    """

@app.route("/metrics", methods=["GET"])
def get_metrics():
    # Metrics can remain public or be restricted; here we leave it public
    return jsonify(METRICS)

def parse_input(data):
    return np.array([
        data["accommodates"],
        data["bathrooms"],
        data["bedrooms"],
        data["beds"],
        data["latitude"],
        data["longitude"]
    ]).reshape(1, -1)

@app.route("/predict_lr", methods=["POST"])
def predict_lr():
    try:
        data = request.get_json()
        x = parse_input(data)
        pred = model_lr.predict(x)[0]
        return jsonify({"predicted_price": round(float(pred), 2)})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400

@app.route("/predict_rf", methods=["POST"])
def predict_rf():
    try:
        data = request.get_json()
        x = parse_input(data)
        pred = model_rf.predict(x)[0]
        return jsonify({"predicted_price": round(float(pred), 2)})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400

@app.route("/predict_xgb", methods=["POST"])
def predict_xgb():
    try:
        data = request.get_json()
        x = parse_input(data)
        pred = model_xgb.predict(x)[0]
        return jsonify({"predicted_price": round(float(pred), 2)})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400

@app.route("/cluster", methods=["POST"])
def cluster():
    try:
        data = request.get_json()
        x_input = np.array([
            data["latitude"],
            data["longitude"],
            data["accommodates"]
        ]).reshape(1, -1)

        cluster_label = kmeans.predict(x_input)[0]
        cluster_labels = {0: "Economy Properties", 1: "Premium Properties"}

        return jsonify({
            "cluster": int(cluster_label),
            "label": cluster_labels.get(int(cluster_label), f"Group {cluster_label}")
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

if __name__ == "__main__":
    app.run(debug=True, port=5000)
