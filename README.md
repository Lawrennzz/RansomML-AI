# 🚨 Ransomware Detection AI/ML System

## 🌟 **Complete AI/ML System with Multiple Interfaces**

A comprehensive ransomware detection system that combines machine learning, web development, and interactive learning tools. Features multiple interfaces including web applications, Jupyter notebooks, and step-by-step tutorials.

## 🎯 **What's New - Enhanced Features**

### 🔐 **Role-Based Access Control (NEW!)**
- **4 User Roles** based on UML Use Case Diagram:
  - **Cybersecurity Professional**: Detect ransomware, monitor behavior, view reports, train models, configure rules
  - **IT Administrator**: Train models, configure rules, monitor performance, manage settings
  - **System User**: View security status, receive protection, make predictions
  - **Academic Researcher**: Conduct research, view reports, train models, view visualizations
- **Authentication System** with session management and password hashing
- **Permission-based UI** that shows/hides features based on user role
- **Protected API routes** with role-based access control

### 🤖 **Advanced ML Features**
- **Feature Importance for ALL Models**: Now supports Random Forest (native), SVM, Neural Networks, and CNN-LSTM using permutation importance
- **Training Time Tracking**: Real-time tracking and display of training duration for each model
- **Multiple ML Models**: Random Forest, SVM, Neural Networks (MLP), CNN-LSTM
- **Training Time Estimates**: 
  - Random Forest: ~30-60 seconds (fastest)
  - SVM: ~2-5 minutes
  - Neural Networks: ~1-3 minutes
  - CNN-LSTM: ~3-8 minutes

### 🌐 **Enhanced Web Interface**
- **Separated Navigation**: Each feature has its own isolated view section
- **Training Visualizations**: Confusion matrix and feature importance directly in Train Model section
- **Real-time Metrics**: Training performance metrics displayed immediately after training
- **Responsive Design**: Mobile-friendly interface with smooth navigation

### 📓 **Jupyter Notebook Support (NEW!)**
- **Comprehensive Analysis Notebook**: `ransomware_detection_analysis.ipynb`
  - Complete data loading and exploration
  - Data preprocessing and cleaning
  - Training multiple ML models (Random Forest, SVM, Neural Network)
  - Model evaluation and comparison
  - Feature importance analysis
  - Advanced visualizations (confusion matrices, ROC curves, feature importance charts)
  - Making predictions and testing
  - Model saving and persistence
- **Interactive Data Science Environment** for experimentation and research

### 🚀 **Production-Ready Features**
- **RESTful API endpoints** with authentication
- **Auto-trained ML models** with comprehensive performance metrics
- **Detection history** tracking and logging
- **Secure authentication** with password hashing

---

## 🚀 **Quick Start Guide**

### **🌐 For Web Interface (Recommended):**
```bash
# Start the web application
python app.py

# Access: http://localhost:5000
# You will be redirected to login page
```

### **🔐 Login with Test Accounts:**
The system includes 4 default test accounts for each role:

| Role | Email | Password |
|------|-------|----------|
| **Cybersecurity Professional** | `cyber_pro@example.com` | `cyber123` |
| **IT Administrator** | `admin@example.com` | `admin123` |
| **System User** | `user@example.com` | `user123` |
| **Academic Researcher** | `researcher@example.com` | `research123` |

**Note**: Each role has different permissions and access to different features!

### **📓 For Jupyter Notebook Analysis:**
```bash
# Start Jupyter Notebook
jupyter notebook

# Or use JupyterLab
jupyter lab

# Then open: ransomware_detection_analysis.ipynb
```

The notebook provides a comprehensive analysis environment with:
- Data exploration and visualization
- Model training and evaluation
- Feature importance analysis
- Interactive visualizations
- Model comparison and benchmarking

---

## 📁 **Complete File Structure**

### **🌐 Web Applications**
- **`app.py`** - Main web application with role-based authentication
- **`simple_test.py`** - Simple test script

### **📓 Jupyter Notebooks**
- **`ransomware_detection_analysis.ipynb`** - Comprehensive ML analysis notebook with data exploration, model training, evaluation, and visualizations

### **🌐 Web Templates**
- **`templates/index.html`** - Main dashboard interface with separated views
- **`templates/login.html`** - Login page with role descriptions and test accounts

### **📊 Generated Files**
- **`best_ransomware_model.pkl`** - Trained Random Forest model
- **`ransomware_scaler.pkl`** - Feature scaler
- **`ransomware_svm_model.pkl`** - SVM model
- **`detection_logs.csv`** - Detection history

---

## 🎯 **Key Features**

### **🤖 Machine Learning Models**
- **Random Forest Classifier** - Fast training (~30-60s), built-in feature importance
- **Support Vector Machine (SVM)** - High accuracy, permutation-based feature importance
- **Neural Networks (MLP)** - Deep learning approach, permutation-based feature importance
- **CNN-LSTM** - Advanced deep learning model for sequential patterns
- **Feature Importance Analysis** - Available for ALL models (native for RF, permutation for others)
- **Training Time Tracking** - Real-time training duration display

### **📊 Behavioral Features Analyzed**
- File access patterns and frequency
- Entropy changes in file modifications
- System call analysis and monitoring
- Network connection patterns
- CPU and memory usage patterns
- Disk I/O operations
- Process count monitoring
- Windows registry changes

### **🌐 Multiple Interfaces**
- **Web Dashboard** - User-friendly interface with role-based access
- **Jupyter Notebook** - Interactive data science environment with comprehensive ML analysis (`ransomware_detection_analysis.ipynb`)
- **API Endpoints** - Programmatic access with authentication
- **Tutorial System** - Step-by-step learning

### **📈 Real-Time Detection**
- **Instant predictions** with confidence scores
- **Risk level assessment** (High/Medium/Low)
- **Detection history** tracking
- **Performance metrics** visualization

---

## 🛠️ **Technologies Used**

- **Backend**: Flask (Python web framework) with session management
- **Authentication**: Werkzeug (password hashing, session security)
- **Machine Learning**: scikit-learn, pandas, numpy, TensorFlow (optional)
- **Feature Importance**: scikit-learn permutation_importance for universal support
- **Visualization**: Chart.js, matplotlib, seaborn
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5
- **Notebook**: Jupyter, ipywidgets
- **Model Persistence**: joblib
- **Data Storage**: JSON (user database), CSV (datasets)

---

## 📊 **Performance Metrics**

- **Accuracy**: >95% target (varies by model)
- **Detection Latency**: <2 seconds
- **Cross-validation**: 5-fold CV for robust evaluation
- **Confidence Scoring**: Probability-based predictions
- **Real-time Processing**: Instant analysis

---

## 🚀 **Installation & Setup**

### **Prerequisites**
```bash
# Python 3.7+ required
python --version
```

### **Install Dependencies**
```bash
# Install all required packages
pip install -r requirements.txt

# Or install individually
pip install flask pandas numpy scikit-learn matplotlib seaborn jupyter ipywidgets joblib
```

### **Start Jupyter Notebook**
```bash
# Start Jupyter Notebook server
jupyter notebook

# Or use JupyterLab for enhanced features
jupyter lab

# Open ransomware_detection_analysis.ipynb for comprehensive ML analysis
```

---

## 🎮 **Usage Examples**


### **Web Application**
```bash
# Run main application
python app.py

# Access: http://localhost:5000
# You will be redirected to login page
```

### **📓 Jupyter Notebook Analysis**
```bash
# Start Jupyter Notebook
jupyter notebook

# Open ransomware_detection_analysis.ipynb
# Run all cells to perform complete ML analysis
```

**Notebook Features:**
- Load and explore dataset
- Preprocess and clean data
- Train multiple ML models (Random Forest, SVM, Neural Network)
- Compare model performance
- Visualize confusion matrices, ROC curves, and feature importance
- Make predictions on test samples
- Save trained models

**Notebook Sections:**
1. Data Loading & Exploration
2. Data Preprocessing
3. Model Training (3 different models)
4. Model Comparison & Evaluation
5. Feature Importance Analysis
6. Visualizations
7. Predictions & Testing
8. Model Saving

### **API Usage with Authentication**
```python
import requests

# Create session for authentication
session = requests.Session()

# Login first
login_data = {
    'email': 'admin@example.com',
    'password': 'admin123'
}
response = session.post('http://localhost:5000/login', json=login_data)
print(f"Login: {response.json()}")

# Now you can make authenticated requests
# Train model (requires train_ml_model permission)
response = session.post('http://localhost:5000/api/train', json={'model_type': 'rf'})
print(response.json())

# Make prediction (available to all authenticated users)
data = {
    'Machine': 332,
    'DebugSize': 28,
    'DebugRVA': 65536,
    # ... other PE features
}
response = session.post('http://localhost:5000/api/predict', json=data)
print(response.json())
```

### **🧪 Testing Real-Time Behavior Data**

#### **Via Web UI:**
1. **Login** with a role that has `monitor_system_behavior` permission (Cybersecurity Professional)
2. Navigate to **"Real-Time"** section in the navbar
3. Enter behavioral data as JSON in the textarea
4. Click **"Classify Real-Time"** button

#### **Behavioral Data Format (JSON):**
The system accepts behavioral indicators that map to PE features:

```json
{
  "file_access_count": 100,
  "file_modifications": 50,
  "system_calls": 25,
  "directory_access": 15,
  "crypto_operations": 2,
  "process_count": 8,
  "registry_changes": 5,
  "memory_usage": 2048000,
  "dll_characteristics": 512,
  "debug_info": 1024
}
```

#### **Example Test Cases:**

**Test Case 1: Benign Behavior (Normal Activity)**
```json
{
  "file_access_count": 50,
  "file_modifications": 10,
  "system_calls": 15,
  "directory_access": 8,
  "crypto_operations": 0,
  "process_count": 5,
  "registry_changes": 2,
  "memory_usage": 1048576,
  "dll_characteristics": 256,
  "debug_info": 512
}
```

**Test Case 2: Suspicious Behavior (Potential Ransomware)**
```json
{
  "file_access_count": 500,
  "file_modifications": 200,
  "system_calls": 100,
  "directory_access": 50,
  "crypto_operations": 5,
  "process_count": 20,
  "registry_changes": 15,
  "memory_usage": 4194304,
  "dll_characteristics": 4096,
  "debug_info": 2048
}
```

**Test Case 3: High-Risk Behavior (Likely Ransomware)**
```json
{
  "file_access_count": 1000,
  "file_modifications": 500,
  "system_calls": 200,
  "directory_access": 100,
  "crypto_operations": 10,
  "process_count": 30,
  "registry_changes": 25,
  "memory_usage": 8388608,
  "dll_characteristics": 8192,
  "debug_info": 4096
}
```

#### **Via API:**
```python
import requests
import json

# Create session and login
session = requests.Session()
login_data = {
    'email': 'cyber_pro@example.com',
    'password': 'cyber123'
}
session.post('http://localhost:5000/login', json=login_data)

# Test real-time behavior data
behavioral_data = {
    "file_access_count": 500,
    "file_modifications": 200,
    "crypto_operations": 5,
    "system_calls": 100,
    "directory_access": 50,
    "process_count": 20,
    "registry_changes": 15,
    "memory_usage": 4194304,
    "dll_characteristics": 4096,
    "debug_info": 2048
}

response = session.post(
    'http://localhost:5000/api/classify-realtime',
    json=behavioral_data
)
result = response.json()

print("Prediction:", result['result']['prediction'])
print("Confidence:", result['result']['confidence'])
print("Threat Classification:", result['threat_classification'])
print("Recommendation:", result['recommendation'])
print("Behavioral Indicators:", result['behavioral_indicators'])
```

#### **Expected Response:**
```json
{
  "success": true,
  "result": {
    "prediction": 1,
    "confidence": 0.85,
    "benign_probability": 0.15,
    "ransomware_probability": 0.85,
    "risk_level": "HIGH",
    "model_type": "rf"
  },
  "behavioral_indicators": {
    "file_modifications": 700,
    "system_calls": 100,
    "crypto_operations": 5,
    "suspicious_activity_score": 0.8
  },
  "threat_classification": "High-Risk Crypto Ransomware",
  "recommendation": "IMMEDIATE_ACTION"
}
```

#### **Behavioral Data Mapping:**
The system automatically maps behavioral indicators to PE features:
- `file_access_count` → `ExportSize`
- `file_modifications` → `ResourceSize`
- `system_calls` → `NumberOfSections`
- `directory_access` → `DebugRVA`
- `crypto_operations` → `BitcoinAddresses`
- `process_count` → `Machine`
- `registry_changes` → `IatVRA`
- `memory_usage` → `SizeOfStackReserve`
- `dll_characteristics` → `DllCharacteristics`
- `debug_info` → `DebugSize`

---

## 📚 **Learning Path**

### **For Beginners**
1. **Start**: Run `python app.py`
2. **Login**: Use test accounts to explore different roles
3. **Practice**: Use the Predict feature to make predictions
4. **Explore**: Dashboard with statistics and visualizations

### **For Developers**
1. **Study**: This README and code structure
2. **Run**: `python app.py`
3. **Login**: Use test accounts to explore different roles
4. **Explore**: API endpoints with authentication
5. **Customize**: Add new features and roles

### **For Data Scientists**
1. **Open Notebook**: Launch `ransomware_detection_analysis.ipynb` in Jupyter
2. **Explore Data**: Run data loading and exploration cells
3. **Train Models**: Train Random Forest, SVM, and Neural Network models
4. **Analyze Results**: Compare model performance and feature importance
5. **Visualize**: View confusion matrices, ROC curves, and feature importance charts
6. **Experiment**: Modify hyperparameters and try different configurations
7. **Web Interface**: Use the web dashboard for real-time predictions and monitoring

---

## 🔧 **API Endpoints**

### **Authentication**
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/login` | GET/POST | Login page and authentication |
| `/logout` | GET | Logout user |
| `/api/current-user` | GET | Get current logged-in user info |

### **Core Features** (Authentication Required)
| Endpoint | Method | Description | Required Permission |
|----------|--------|-------------|-------------------|
| `/` | GET | Main dashboard | All authenticated users |
| `/api/predict` | POST | Make prediction | All authenticated users |
| `/api/train` | POST | Train ML models | `train_ml_model` |
| `/api/dataset-stats` | GET | Get dataset statistics | All authenticated users |
| `/api/model-performance` | GET | Get model metrics | All authenticated users |
| `/api/feature-columns` | GET | Get feature columns | All authenticated users |
| `/api/feature-importance` | GET | Get feature importance | All authenticated users |
| `/api/upload-csv` | POST | Upload CSV dataset | `train_ml_model` |

### **Advanced Features** (Role-Specific)
| Endpoint | Method | Description | Required Permission |
|----------|--------|-------------|-------------------|
| `/api/detection-history` | GET | Get detection history | `view_detection_reports` |
| `/api/ingest-logs` | POST | Ingest system behavior logs | `monitor_system_behavior` |
| `/api/classify-realtime` | POST | Real-time classification | `monitor_system_behavior` |
| `/api/detection-logs` | GET | Get detailed detection logs | `view_detection_reports` |
| `/api/system-logs` | GET | Get system behavior logs | `monitor_system_performance` |
| `/api/rules` | GET | List detection rules | `configure_detection_rules` |
| `/api/rules` | POST | Create/Update rule (id optional) | `configure_detection_rules` |
| `/api/rules/<id>` | DELETE | Delete rule | `configure_detection_rules` |
| `/api/settings` | GET | Get system settings | `manage_system_settings` |
| `/api/settings` | POST | Update system settings | `manage_system_settings` |

---

## 🎯 **Project Highlights**

- ✅ **Complete ML Pipeline** - From data to deployment
- ✅ **Role-Based Access Control** - 4 user roles with permission-based access
- ✅ **Multiple ML Models** - Random Forest, SVM, Neural Networks, CNN-LSTM
- ✅ **Universal Feature Importance** - Available for all model types
- ✅ **Training Time Tracking** - Real-time duration display
- ✅ **Production Ready** - Secure authentication and scalable architecture
- ✅ **Separated UI Views** - Clean interface with isolated feature sections
- ✅ **Jupyter Notebook** - Comprehensive ML analysis notebook included
- ✅ **Comprehensive Documentation** - Complete guides and test accounts

---

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

---

## 📄 **License**

This project is open source and available under the MIT License.

---

## 👨‍💻 **Author**

**Lawrence** - [GitHub Profile](https://github.com/Lawrennzz)

---

## 🙏 **Acknowledgments**

- Kaggle for the ransomware detection dataset
- Scikit-learn and TensorFlow communities
- Jupyter project for the notebook environment
- Flask community for web framework

---

## 📞 **Support**

For questions or issues:
- Create an issue in this repository
- Check this README for setup and usage instructions
- Use the test accounts to explore different roles and features

---

## 📝 **Key Features Summary**

### **🔐 Security & Access Control**
- Session-based authentication
- Password hashing with Werkzeug
- Role-based permission system
- Protected API routes

### **🤖 Machine Learning**
- 4 model types with different training speeds
- Feature importance for all models (native + permutation)
- Training time tracking and estimates
- Performance metrics visualization

### **🎨 User Interface**
- Separated navigation sections
- Role-based feature visibility
- Real-time training visualizations
- Responsive design

**🎉 This is a complete, production-ready AI/ML system with role-based access control!**

Perfect for:
- **Cybersecurity research** - Detect and analyze ransomware
- **Academic projects** - Research role with full access
- **Production deployment** - Secure multi-user system
- **Portfolio demonstration** - Showcase ML and web development skills
- **Learning AI/ML** - Multiple models and visualization tools