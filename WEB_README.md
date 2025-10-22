# Ransomware Detection AI/ML Web Application

## 🌐 Web-Based GUI for Ransomware Detection System

This web application provides a beautiful, interactive dashboard for your ransomware detection AI/ML system. It features real-time detection, data visualization, and comprehensive analytics.

## ✨ Features

### 🎯 Real-Time Detection Interface
- Interactive form for inputting system behavior metrics
- Instant ransomware detection with confidence scoring
- Risk level assessment (High/Medium/Low)
- Detailed probability breakdown

### 📊 Interactive Dashboard
- **System Status**: Model status and total detections
- **Dataset Statistics**: Training data overview
- **Model Performance**: Accuracy, Precision, Recall, F1-Score metrics
- **Feature Importance**: Visual ranking of most important features
- **Confusion Matrix**: Model performance visualization

### 📈 Advanced Analytics
- Feature importance charts
- Model performance metrics
- Detection history tracking
- Real-time data visualization

### 🎨 Modern UI/UX
- Responsive Bootstrap 5 design
- Beautiful gradient backgrounds
- Interactive charts with Chart.js
- Font Awesome icons
- Mobile-friendly interface

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the Web Application
```bash
python start_web_app.py
```

Or directly:
```bash
python app.py
```

### 3. Access the Dashboard
Open your browser and go to: **http://localhost:5000**

## 🖥️ Web Interface Overview

### Dashboard Tab
- **System Status**: Shows if the model is trained and ready
- **Dataset Statistics**: Overview of training data
- **Model Performance**: Key metrics and accuracy scores

### Real-Time Detection Tab
- Input system behavior metrics:
  - File Access Count
  - Entropy Change
  - System Calls
  - Network Connections
  - File Modifications
  - CPU Usage (%)
  - Memory Usage (%)
  - Disk I/O
  - Process Count
  - Registry Changes
- Get instant detection results with confidence scores

### Analytics Tab
- **Feature Importance Chart**: Bar chart showing which features are most important for detection
- **Confusion Matrix**: Visual representation of model performance

### Detection History Tab
- View recent detection results
- See confidence scores and risk levels
- Track detection patterns over time

## 🔧 Technical Details

### Backend (Flask)
- **RansomwareDetector Class**: Core ML functionality
- **RESTful API**: Clean API endpoints for frontend communication
- **Model Training**: Automatic model training with synthetic data
- **Real-time Prediction**: Fast inference with confidence scoring

### Frontend (HTML/CSS/JavaScript)
- **Bootstrap 5**: Modern, responsive UI framework
- **Chart.js**: Interactive data visualizations
- **Vanilla JavaScript**: Clean, efficient frontend logic
- **AJAX**: Seamless API communication

### Machine Learning Models
- **Random Forest**: Primary detection model
- **Feature Scaling**: StandardScaler for normalization
- **Cross-validation**: Robust model evaluation
- **Synthetic Data**: Self-contained dataset generation

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main dashboard page |
| `/api/train` | POST | Train ML models |
| `/api/predict` | POST | Make detection prediction |
| `/api/dataset-stats` | GET | Get dataset statistics |
| `/api/model-performance` | GET | Get model performance metrics |
| `/api/detection-history` | GET | Get detection history |

## 🎮 Usage Examples

### Training the Model
1. Go to the Dashboard tab
2. Click "Train Model" button
3. Wait for training to complete
4. View updated performance metrics

### Real-Time Detection
1. Go to the Real-Time Detection tab
2. Enter system behavior metrics
3. Click "Analyze System Behavior"
4. View detection results with confidence scores

### Viewing Analytics
1. Go to the Analytics tab
2. View feature importance chart
3. Examine confusion matrix
4. Analyze model performance

## 🔍 Detection Results Interpretation

### Prediction Types
- **🚨 RANSOMWARE DETECTED**: System shows ransomware behavior patterns
- **✅ System Appears Benign**: Normal system behavior detected

### Risk Levels
- **🔴 HIGH RISK**: Confidence > 80%
- **🟡 MEDIUM RISK**: Confidence 60-80%
- **🟢 LOW RISK**: Confidence < 60%

### Confidence Scoring
- Shows probability of correct classification
- Higher confidence = more reliable prediction
- Based on model's certainty in prediction

## 🛠️ Customization

### Adding New Features
1. Update the `RansomwareDetector` class in `app.py`
2. Add new form fields in `templates/index.html`
3. Update JavaScript validation in `static/js/app.js`

### Modifying Visualizations
1. Edit chart configurations in `static/js/app.js`
2. Customize Chart.js options for different chart types
3. Add new chart types as needed

### Styling Changes
1. Modify CSS in `templates/index.html`
2. Update Bootstrap classes for different layouts
3. Customize color schemes and gradients

## 📁 File Structure

```
RansomML-AI/
├── app.py                          # Main Flask application
├── start_web_app.py               # Startup script
├── templates/
│   └── index.html                 # Main dashboard template
├── static/
│   ├── css/                       # CSS files (if any)
│   └── js/
│       └── app.js                # Frontend JavaScript
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── [existing ML files...]       # Your existing ML system files
```

## 🔒 Security Considerations

- **Input Validation**: All user inputs are validated
- **Error Handling**: Comprehensive error handling and user feedback
- **CORS Support**: Cross-origin resource sharing enabled
- **Secret Key**: Flask secret key for session security

## 🚨 Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Change port in app.py
   app.run(debug=True, host='0.0.0.0', port=5001)
   ```

2. **Missing Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Charts Not Loading**
   - Check browser console for JavaScript errors
   - Ensure Chart.js is loading properly
   - Verify API responses are valid JSON

4. **Model Training Fails**
   - Check Python console for error messages
   - Ensure all ML libraries are properly installed
   - Verify sufficient system memory

## 🎯 Performance Optimization

- **Caching**: Model and scaler are cached after training
- **Efficient API**: Minimal data transfer between frontend and backend
- **Responsive Design**: Optimized for various screen sizes
- **Fast Inference**: Optimized ML model for quick predictions

## 🔮 Future Enhancements

- **Real-time Monitoring**: Live system monitoring integration
- **Alert System**: Email/SMS notifications for threats
- **Model Comparison**: Side-by-side model performance comparison
- **Export Features**: Download detection reports and data
- **User Authentication**: Multi-user support with login system
- **API Documentation**: Interactive API documentation with Swagger

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review browser console for JavaScript errors
3. Check Python console for backend errors
4. Ensure all dependencies are properly installed

## 🎉 Enjoy Your Web-Based Ransomware Detection System!

Your AI/ML ransomware detection system now has a beautiful, interactive web interface with comprehensive visualizations and real-time detection capabilities. The system is production-ready and can be easily deployed or extended with additional features.

---

**Note**: This web application integrates seamlessly with your existing ML models and provides a user-friendly interface for both technical and non-technical users to interact with your ransomware detection system.
