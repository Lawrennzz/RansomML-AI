#!/usr/bin/env python3
"""
Interactive Learning Tutorial: Ransomware Detection System
Step-by-step guide to understand and build the system
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class RansomwareDetectionTutorial:
    """Interactive tutorial for learning ransomware detection"""
    
    def __init__(self):
        self.dataset = None
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.training_history = []
        
    def lesson_1_data_understanding(self):
        """Lesson 1: Understanding the Data"""
        print("🎓 LESSON 1: Understanding Ransomware Detection Data")
        print("=" * 60)
        
        print("\n📊 What is ransomware detection data?")
        print("Ransomware detection uses behavioral features to identify malicious software.")
        print("These features capture how a program behaves on a system.")
        
        print("\n🔍 Key Features We'll Use:")
        features_explanation = {
            'file_access_count': 'Number of files accessed (ransomware accesses many files)',
            'entropy_change': 'Randomness in file changes (ransomware encrypts files randomly)',
            'system_calls': 'Number of system calls (ransomware makes many calls)',
            'network_connections': 'Network activity (ransomware may communicate)',
            'file_modifications': 'Files modified (ransomware modifies many files)',
            'cpu_usage': 'CPU utilization (ransomware uses high CPU)',
            'memory_usage': 'Memory usage (ransomware uses high memory)',
            'disk_io': 'Disk input/output (ransomware reads/writes heavily)',
            'process_count': 'Number of processes (ransomware may spawn processes)',
            'registry_changes': 'Windows registry changes (ransomware modifies registry)'
        }
        
        for feature, explanation in features_explanation.items():
            print(f"  • {feature}: {explanation}")
        
        print("\n💡 Why These Features?")
        print("Ransomware has distinct behavioral patterns:")
        print("  - Accesses many files quickly")
        print("  - Uses high CPU/memory resources")
        print("  - Makes many system calls")
        print("  - Modifies files (encryption)")
        print("  - Changes system settings")
        
        return True
    
    def lesson_2_data_creation(self):
        """Lesson 2: Creating Synthetic Data"""
        print("\n🎓 LESSON 2: Creating Synthetic Ransomware Data")
        print("=" * 60)
        
        print("\n📝 Why Synthetic Data?")
        print("Real ransomware datasets are:")
        print("  - Hard to obtain (security reasons)")
        print("  - May contain actual malware")
        print("  - Limited in scope")
        print("\nSynthetic data allows us to:")
        print("  - Learn safely")
        print("  - Control the dataset size")
        print("  - Understand the patterns")
        
        print("\n🔧 Creating the Dataset...")
        
        # Set random seed for reproducibility
        np.random.seed(42)
        n_samples = 5000
        
        print(f"Generating {n_samples} samples...")
        
        # Generate synthetic features
        data = {
            'file_access_count': np.random.poisson(50, n_samples),
            'entropy_change': np.random.normal(0, 2, n_samples),
            'system_calls': np.random.poisson(100, n_samples),
            'network_connections': np.random.poisson(20, n_samples),
            'file_modifications': np.random.poisson(30, n_samples),
            'cpu_usage': np.random.beta(2, 5, n_samples) * 100,
            'memory_usage': np.random.beta(2, 5, n_samples) * 100,
            'disk_io': np.random.poisson(200, n_samples),
            'process_count': np.random.poisson(150, n_samples),
            'registry_changes': np.random.poisson(10, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        print("✅ Features generated!")
        
        # Create labels based on feature combinations
        print("\n🏷️ Creating Labels...")
        print("We'll label samples as ransomware if they show suspicious patterns:")
        
        ransomware_indicators = (
            (df['file_access_count'] > 80) |
            (df['entropy_change'] > 2) |
            (df['system_calls'] > 150) |
            (df['file_modifications'] > 50) |
            (df['cpu_usage'] > 80)
        )
        
        df['label'] = ransomware_indicators.astype(int)
        
        # Add some noise to make it realistic
        noise_mask = np.random.random(n_samples) < 0.1
        df.loc[noise_mask, 'label'] = 1 - df.loc[noise_mask, 'label']
        
        self.dataset = df
        self.feature_columns = [col for col in df.columns if col != 'label']
        
        print("✅ Labels created!")
        
        # Display dataset info
        print(f"\n📊 Dataset Information:")
        print(f"  Shape: {df.shape}")
        print(f"  Features: {len(self.feature_columns)}")
        print(f"  Ransomware samples: {df['label'].sum()} ({df['label'].mean()*100:.1f}%)")
        print(f"  Benign samples: {len(df) - df['label'].sum()} ({(1-df['label'].mean())*100:.1f}%)")
        
        print(f"\n📋 First 5 rows:")
        print(df.head())
        
        return df
    
    def lesson_3_data_analysis(self):
        """Lesson 3: Data Analysis and Visualization"""
        print("\n🎓 LESSON 3: Data Analysis and Visualization")
        print("=" * 60)
        
        if self.dataset is None:
            print("❌ Please run Lesson 2 first!")
            return False
        
        df = self.dataset
        
        print("\n📈 Understanding Data Distribution")
        
        # Basic statistics
        print(f"\n📊 Dataset Statistics:")
        print(df.describe())
        
        # Label distribution
        print(f"\n🏷️ Label Distribution:")
        label_counts = df['label'].value_counts()
        print(f"  Benign (0): {label_counts[0]} samples")
        print(f"  Ransomware (1): {label_counts[1]} samples")
        
        # Feature analysis
        print(f"\n🔍 Feature Analysis:")
        print("Let's look at how features differ between benign and ransomware:")
        
        benign_data = df[df['label'] == 0]
        ransomware_data = df[df['label'] == 1]
        
        print(f"\n📊 Average Values by Class:")
        comparison = pd.DataFrame({
            'Benign': benign_data[self.feature_columns].mean(),
            'Ransomware': ransomware_data[self.feature_columns].mean()
        })
        print(comparison.round(2))
        
        # Create visualizations
        print(f"\n📊 Creating Visualizations...")
        
        # Feature distribution plot
        fig, axes = plt.subplots(2, 5, figsize=(20, 10))
        axes = axes.ravel()
        
        for i, feature in enumerate(self.feature_columns):
            axes[i].hist(benign_data[feature], alpha=0.7, label='Benign', bins=30)
            axes[i].hist(ransomware_data[feature], alpha=0.7, label='Ransomware', bins=30)
            axes[i].set_title(f'{feature}')
            axes[i].legend()
        
        plt.tight_layout()
        plt.show()
        
        print("✅ Visualizations created!")
        
        return True
    
    def lesson_4_model_training(self):
        """Lesson 4: Machine Learning Model Training"""
        print("\n🎓 LESSON 4: Training Machine Learning Models")
        print("=" * 60)
        
        if self.dataset is None:
            print("❌ Please run Lesson 2 first!")
            return False
        
        df = self.dataset
        
        print("\n🤖 What is Machine Learning?")
        print("Machine Learning learns patterns from data to make predictions.")
        print("We'll use supervised learning - we have features (X) and labels (y).")
        
        print("\n📊 Preparing Data for Training")
        
        # Prepare features and labels
        X = df[self.feature_columns]
        y = df['label']
        
        print(f"  Features shape: {X.shape}")
        print(f"  Labels shape: {y.shape}")
        
        # Split data
        print("\n✂️ Splitting Data")
        print("We split data into training (80%) and testing (20%) sets.")
        print("Training set: Used to teach the model")
        print("Testing set: Used to evaluate the model")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"  Training set: {X_train.shape[0]} samples")
        print(f"  Testing set: {X_test.shape[0]} samples")
        
        # Feature scaling
        print("\n⚖️ Feature Scaling")
        print("Different features have different scales:")
        print("  file_access_count: 0-200")
        print("  cpu_usage: 0-100")
        print("We normalize them to have mean=0, std=1")
        
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("✅ Features scaled!")
        
        # Train Random Forest
        print("\n🌲 Training Random Forest Model")
        print("Random Forest is an ensemble method that uses multiple decision trees.")
        print("It's good for:")
        print("  - Handling mixed data types")
        print("  - Feature importance")
        print("  - Robust to overfitting")
        
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        print("Training model...")
        self.model.fit(X_train_scaled, y_train)
        print("✅ Model trained!")
        
        # Cross-validation
        print("\n🔄 Cross-Validation")
        print("Cross-validation tests the model's reliability by training on different subsets.")
        
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5, scoring='f1')
        print(f"  Cross-validation F1 scores: {cv_scores.round(3)}")
        print(f"  Mean CV F1 score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Evaluate on test set
        print("\n📊 Model Evaluation")
        y_pred = self.model.predict(X_test_scaled)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        
        # Confusion matrix
        print(f"\n📊 Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print("                Predicted")
        print("Actual     Benign  Ransomware")
        print(f"Benign        {cm[0,0]:4d}       {cm[0,1]:4d}")
        print(f"Ransomware    {cm[1,0]:4d}       {cm[1,1]:4d}")
        
        # Feature importance
        print(f"\n🎯 Feature Importance:")
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(feature_importance)
        
        # Store training history
        self.training_history.append({
            'model': 'Random Forest',
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        })
        
        return True
    
    def lesson_5_prediction(self):
        """Lesson 5: Making Predictions"""
        print("\n🎓 LESSON 5: Making Real-Time Predictions")
        print("=" * 60)
        
        if self.model is None or self.scaler is None:
            print("❌ Please run Lesson 4 first!")
            return False
        
        print("\n🔮 How Predictions Work")
        print("1. Input: System behavior metrics")
        print("2. Preprocess: Scale features")
        print("3. Predict: Use trained model")
        print("4. Output: Ransomware probability")
        
        def predict_ransomware(features_dict):
            """Make prediction on new data"""
            print(f"\n🔍 Analyzing system behavior...")
            
            # Convert input to array
            feature_values = [features_dict[col] for col in self.feature_columns]
            feature_array = np.array(feature_values).reshape(1, -1)
            
            # Scale features
            feature_array_scaled = self.scaler.transform(feature_array)
            
            # Make prediction
            prediction = self.model.predict(feature_array_scaled)[0]
            probability = self.model.predict_proba(feature_array_scaled)[0]
            confidence = max(probability)
            
            result = {
                'prediction': int(prediction),
                'confidence': float(confidence),
                'benign_probability': float(probability[0]),
                'ransomware_probability': float(probability[1])
            }
            
            return result
        
        # Example predictions
        print("\n🧪 Testing with Example Scenarios")
        
        # Benign example
        print("\n📝 Example 1: Normal System Behavior")
        benign_example = {
            'file_access_count': 30,
            'entropy_change': 0.5,
            'system_calls': 80,
            'network_connections': 10,
            'file_modifications': 15,
            'cpu_usage': 25.0,
            'memory_usage': 40.0,
            'disk_io': 100,
            'process_count': 120,
            'registry_changes': 2
        }
        
        print("Input features:")
        for feature, value in benign_example.items():
            print(f"  {feature}: {value}")
        
        result = predict_ransomware(benign_example)
        
        if result['prediction'] == 1:
            print(f"🚨 PREDICTION: RANSOMWARE DETECTED!")
        else:
            print(f"✅ PREDICTION: System appears BENIGN")
        
        print(f"Confidence: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
        print(f"Benign probability: {result['benign_probability']:.4f}")
        print(f"Ransomware probability: {result['ransomware_probability']:.4f}")
        
        # Ransomware example
        print("\n📝 Example 2: Suspicious System Behavior")
        ransomware_example = {
            'file_access_count': 150,
            'entropy_change': 3.5,
            'system_calls': 300,
            'network_connections': 50,
            'file_modifications': 100,
            'cpu_usage': 85.0,
            'memory_usage': 90.0,
            'disk_io': 500,
            'process_count': 200,
            'registry_changes': 25
        }
        
        print("Input features:")
        for feature, value in ransomware_example.items():
            print(f"  {feature}: {value}")
        
        result = predict_ransomware(ransomware_example)
        
        if result['prediction'] == 1:
            print(f"🚨 PREDICTION: RANSOMWARE DETECTED!")
        else:
            print(f"✅ PREDICTION: System appears BENIGN")
        
        print(f"Confidence: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
        print(f"Benign probability: {result['benign_probability']:.4f}")
        print(f"Ransomware probability: {result['ransomware_probability']:.4f}")
        
        return True
    
    def lesson_6_web_integration(self):
        """Lesson 6: Web Application Integration"""
        print("\n🎓 LESSON 6: Web Application Integration")
        print("=" * 60)
        
        print("\n🌐 Why Web Integration?")
        print("Web applications make ML models accessible to:")
        print("  - Non-technical users")
        print("  - Multiple users simultaneously")
        print("  - Different devices (desktop, mobile)")
        print("  - Remote access")
        
        print("\n🏗️ Web Architecture")
        print("Frontend (HTML/CSS/JavaScript) ←→ Backend (Flask/Python) ←→ ML Model")
        print("  User Interface                    API Endpoints           Predictions")
        
        print("\n📡 API Endpoints")
        print("RESTful API for communication:")
        print("  GET /api/dataset-stats      - Get dataset information")
        print("  GET /api/model-performance  - Get model metrics")
        print("  POST /api/train            - Train the model")
        print("  POST /api/predict         - Make prediction")
        
        print("\n💻 Frontend Components")
        print("  • Input forms for system metrics")
        print("  • Real-time prediction results")
        print("  • Data visualizations")
        print("  • Detection history")
        
        print("\n🔧 Backend Components")
        print("  • Flask web framework")
        print("  • Model loading and prediction")
        print("  • Data preprocessing")
        print("  • API endpoints")
        
        print("\n📊 Data Flow")
        print("1. User enters system metrics in web form")
        print("2. JavaScript sends data to Flask API")
        print("3. Flask preprocesses data and calls ML model")
        print("4. Model returns prediction with confidence")
        print("5. Flask sends result back to frontend")
        print("6. JavaScript displays result to user")
        
        print("\n✅ Web Integration Benefits")
        print("  • User-friendly interface")
        print("  • Real-time predictions")
        print("  • Scalable architecture")
        print("  • Easy deployment")
        
        return True
    
    def lesson_7_jupyter_notebooks(self):
        """Lesson 7: Jupyter Notebooks for Interactive Learning"""
        print("\n🎓 LESSON 7: Jupyter Notebooks for Interactive Learning")
        print("=" * 60)
        
        print("\n📓 What are Jupyter Notebooks?")
        print("Jupyter notebooks combine:")
        print("  • Code cells (executable Python)")
        print("  • Markdown cells (text, equations, images)")
        print("  • Output cells (results, plots, tables)")
        print("  • Interactive widgets")
        
        print("\n🎯 Why Use Jupyter for ML?")
        print("  • Interactive experimentation")
        print("  • Step-by-step analysis")
        print("  • Visual data exploration")
        print("  • Shareable results")
        print("  • Educational tool")
        
        print("\n🔧 Jupyter Cell Types")
        print("Code Cells:")
        print("  ```python")
        print("  import pandas as pd")
        print("  df = pd.read_csv('data.csv')")
        print("  print(df.head())")
        print("  ```")
        
        print("\nMarkdown Cells:")
        print("  ```markdown")
        print("  # This is a heading")
        print("  - Bullet point 1")
        print("  - Bullet point 2")
        print("  **Bold text** and *italic text*")
        print("  ```")
        
        print("\n📊 Interactive Widgets")
        print("ipywidgets for interactive elements:")
        print("  • Sliders for parameter tuning")
        print("  • Buttons for actions")
        print("  • Dropdowns for selections")
        print("  • Text inputs for data entry")
        
        print("\n🎨 Visualization Integration")
        print("  • matplotlib for plots")
        print("  • seaborn for statistical plots")
        print("  • plotly for interactive charts")
        print("  • Inline display of results")
        
        print("\n📚 Educational Benefits")
        print("  • Learn by doing")
        print("  • Immediate feedback")
        print("  • Experimentation")
        print("  • Documentation")
        print("  • Collaboration")
        
        return True
    
    def run_complete_tutorial(self):
        """Run the complete tutorial"""
        print("🎓 COMPLETE RANSOMWARE DETECTION TUTORIAL")
        print("=" * 80)
        print("This tutorial will teach you everything about building a ransomware detection system!")
        print("=" * 80)
        
        lessons = [
            self.lesson_1_data_understanding,
            self.lesson_2_data_creation,
            self.lesson_3_data_analysis,
            self.lesson_4_model_training,
            self.lesson_5_prediction,
            self.lesson_6_web_integration,
            self.lesson_7_jupyter_notebooks
        ]
        
        for i, lesson in enumerate(lessons, 1):
            print(f"\n{'='*20} LESSON {i} {'='*20}")
            try:
                lesson()
                print(f"\n✅ Lesson {i} completed successfully!")
            except Exception as e:
                print(f"\n❌ Lesson {i} failed: {str(e)}")
                break
            
            if i < len(lessons):
                input("\nPress Enter to continue to the next lesson...")
        
        print(f"\n🎉 TUTORIAL COMPLETED!")
        print("=" * 80)
        print("You've learned:")
        print("  ✅ Data understanding and creation")
        print("  ✅ Data analysis and visualization")
        print("  ✅ Machine learning model training")
        print("  ✅ Making predictions")
        print("  ✅ Web application integration")
        print("  ✅ Jupyter notebook usage")
        print("\n🚀 Next steps:")
        print("  • Try modifying the code")
        print("  • Experiment with different features")
        print("  • Build your own web interface")
        print("  • Deploy your model")
        print("=" * 80)

# Interactive tutorial runner
def start_tutorial():
    """Start the interactive tutorial"""
    tutorial = RansomwareDetectionTutorial()
    
    print("🎓 Welcome to the Ransomware Detection Learning Tutorial!")
    print("\nChoose an option:")
    print("1. Run complete tutorial")
    print("2. Run individual lessons")
    print("3. Exit")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        tutorial.run_complete_tutorial()
    elif choice == "2":
        print("\nAvailable lessons:")
        print("1. Data Understanding")
        print("2. Data Creation")
        print("3. Data Analysis")
        print("4. Model Training")
        print("5. Making Predictions")
        print("6. Web Integration")
        print("7. Jupyter Notebooks")
        
        lesson_choice = input("\nEnter lesson number (1-7): ").strip()
        
        lessons = {
            "1": tutorial.lesson_1_data_understanding,
            "2": tutorial.lesson_2_data_creation,
            "3": tutorial.lesson_3_data_analysis,
            "4": tutorial.lesson_4_model_training,
            "5": tutorial.lesson_5_prediction,
            "6": tutorial.lesson_6_web_integration,
            "7": tutorial.lesson_7_jupyter_notebooks
        }
        
        if lesson_choice in lessons:
            lessons[lesson_choice]()
        else:
            print("Invalid lesson choice!")
    elif choice == "3":
        print("Goodbye! 👋")
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    start_tutorial()
