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

### **📓 For Jupyter Notebook:**
```bash
# Interactive notebook experience
python -m jupyterlab
# Open: ransomware_detection.ipynb
```

---

## 📁 **Complete File Structure**

### **🌐 Web Applications**
- **`hybrid_app.py`** - Main hybrid web + Jupyter application
- **`app.py`** - Original web application
- **`simple_test_app.py`** - Simple test app for debugging
- **`start_web_app.py`** - Startup script

### **📓 Jupyter Notebooks**
- **`ransomware_detection.ipynb`** - Main interactive notebook (1728 lines)

### **🎓 Educational Resources**
- **`learning_tutorial.py`** - Interactive tutorial with 7 lessons
- **`LEARNING_GUIDE.md`** - Comprehensive learning guide (686 lines)
- **`COMPLETE_FILE_DOCUMENTATION.md`** - Complete file documentation

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
- **Web Dashboard** - User-friendly interface
- **Jupyter Notebook** - Interactive data science environment
- **API Endpoints** - Programmatic access
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

### **Enable Jupyter Extensions**
```bash
# Enable ipywidgets for interactive features
jupyter nbextension enable --py widgetsnbextension
```

---

## 🎮 **Usage Examples**

### **Interactive Learning**
```bash
# Start the tutorial
python learning_tutorial.py

# Choose your learning path:
# 1. Complete tutorial (recommended)
# 2. Individual lessons
# 3. Exit
```

### **Web Application**
```bash
# Run hybrid app
python hybrid_app.py

# Access interfaces:
# - Dashboard: http://localhost:5000
# - Notebook View: http://localhost:5000/notebook
```

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

---

## 📚 **Learning Path**

### **For Beginners**
1. **Start**: `python learning_tutorial.py`
2. **Follow**: 7 interactive lessons
3. **Practice**: With the web interface
4. **Explore**: Jupyter notebook

### **For Developers**
1. **Study**: This README and code structure
2. **Run**: `python app.py`
3. **Login**: Use test accounts to explore different roles
4. **Explore**: API endpoints with authentication
5. **Customize**: Add new features and roles

### **For Data Scientists**
1. **Open**: `ransomware_detection.ipynb`
2. **Analyze**: Data and models
3. **Experiment**: With different algorithms
4. **Visualize**: Results and insights

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

---

## 🎯 **Project Highlights**

- ✅ **Complete ML Pipeline** - From data to deployment
- ✅ **Role-Based Access Control** - 4 user roles with permission-based access
- ✅ **Multiple ML Models** - Random Forest, SVM, Neural Networks, CNN-LSTM
- ✅ **Universal Feature Importance** - Available for all model types
- ✅ **Training Time Tracking** - Real-time duration display
- ✅ **Production Ready** - Secure authentication and scalable architecture
- ✅ **Separated UI Views** - Clean interface with isolated feature sections
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
- Check the `LEARNING_GUIDE.md` for troubleshooting
- Run `python learning_tutorial.py` for guided help

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