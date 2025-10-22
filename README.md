# 🚨 Ransomware Detection AI/ML System

## 🌟 **Complete AI/ML System with Multiple Interfaces**

A comprehensive ransomware detection system that combines machine learning, web development, and interactive learning tools. Features multiple interfaces including web applications, Jupyter notebooks, and step-by-step tutorials.

## 🎯 **What's New - Enhanced Features**

### 🌐 **Hybrid Web + Jupyter Application**
- **Interactive web dashboard** with Jupyter-style cells
- **Real-time ransomware detection** with confidence scoring
- **Beautiful visualizations** and analytics charts
- **Multiple interfaces**: Web dashboard + Pure notebook view

### 🎓 **Complete Learning System**
- **Interactive tutorial** with 7 step-by-step lessons
- **Comprehensive learning guide** (686 lines of documentation)
- **Hands-on coding examples** and explanations
- **Progressive learning** from basics to advanced concepts

### 🚀 **Production-Ready Features**
- **RESTful API endpoints** for all functionality
- **Auto-trained ML models** (Random Forest, SVM)
- **Detection history** tracking and logging
- **Multiple deployment options**

---

## 🚀 **Quick Start Guide**

### **🎓 For Complete Beginners:**
```bash
# Interactive tutorial with 7 lessons
python learning_tutorial.py
# Choose "Run complete tutorial"
```

### **🌐 For Web Interface:**
```bash
# Hybrid web + Jupyter app (recommended)
python hybrid_app.py
# Access: http://localhost:5000

# Simple test app (guaranteed to work)
python simple_test_app.py
# Access: http://localhost:5001
```

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
- **`templates/hybrid_index.html`** - Hybrid dashboard interface
- **`templates/notebook_view.html`** - Pure notebook view
- **`templates/simple_test.html`** - Test interface

### **📊 Generated Files**
- **`best_ransomware_model.pkl`** - Trained Random Forest model
- **`ransomware_scaler.pkl`** - Feature scaler
- **`ransomware_svm_model.pkl`** - SVM model
- **`detection_logs.csv`** - Detection history

---

## 🎯 **Key Features**

### **🤖 Machine Learning Models**
- **Random Forest Classifier** - Primary detection model
- **Support Vector Machine (SVM)** - Alternative model
- **Feature Importance Analysis** - Identifies key behavioral patterns
- **Cross-validation** - Robust model evaluation

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

- **Backend**: Flask (Python web framework)
- **Machine Learning**: scikit-learn, pandas, numpy
- **Visualization**: matplotlib, seaborn, Chart.js
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5
- **Interactive**: Jupyter Notebooks, ipywidgets
- **Model Persistence**: joblib

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

### **API Usage**
```python
import requests

# Train model
response = requests.post('http://localhost:5000/api/train')
print(response.json())

# Make prediction
data = {
    'file_access_count': 150,
    'cpu_usage': 85.0,
    'memory_usage': 90.0,
    # ... other features
}
response = requests.post('http://localhost:5000/api/predict', json=data)
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
1. **Study**: `LEARNING_GUIDE.md`
2. **Run**: `python hybrid_app.py`
3. **Explore**: API endpoints
4. **Customize**: Add new features

### **For Data Scientists**
1. **Open**: `ransomware_detection.ipynb`
2. **Analyze**: Data and models
3. **Experiment**: With different algorithms
4. **Visualize**: Results and insights

---

## 🔧 **API Endpoints**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main dashboard |
| `/notebook` | GET | Notebook view |
| `/api/train` | POST | Train ML models |
| `/api/predict` | POST | Make prediction |
| `/api/dataset-stats` | GET | Get dataset statistics |
| `/api/model-performance` | GET | Get model metrics |
| `/api/feature-chart` | GET | Get feature importance chart |
| `/api/detection-history` | GET | Get detection history |

---

## 🎯 **Project Highlights**

- ✅ **Complete ML Pipeline** - From data to deployment
- ✅ **Multiple Interfaces** - Web, notebook, tutorial
- ✅ **Educational Focus** - Learn by doing
- ✅ **Production Ready** - Scalable architecture
- ✅ **Interactive Tools** - Real-time experimentation
- ✅ **Comprehensive Documentation** - Complete guides

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

**🎉 This is a complete, production-ready AI/ML system with comprehensive educational materials!**

Perfect for:
- **Learning AI/ML concepts**
- **Cybersecurity research**
- **Academic projects**
- **Production deployment**
- **Portfolio demonstration**