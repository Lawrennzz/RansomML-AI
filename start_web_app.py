#!/usr/bin/env python3
"""
Ransomware Detection Web Application Startup Script
Run this script to start the web-based GUI for the ransomware detection system
"""

import subprocess
import sys
import os

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'flask',
        'pandas',
        'numpy',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'joblib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nInstall missing packages with:")
        print("   pip install -r requirements.txt")
        return False
    
    return True

def main():
    """Main startup function"""
    print("Ransomware Detection Web Application")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists('app.py'):
        print("Error: app.py not found!")
        print("Make sure you're running this script from the project directory")
        return
    
    # Check dependencies
    print("Checking dependencies...")
    if not check_dependencies():
        return
    
    print("All dependencies found!")
    print("\nStarting web application...")
    print("Dashboard will be available at: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    print("\n" + "=" * 50)
    
    try:
        # Start the Flask application
        subprocess.run([sys.executable, 'app.py'])
    except KeyboardInterrupt:
        print("\n\nWeb application stopped!")
    except Exception as e:
        print(f"\nError starting application: {e}")

if __name__ == "__main__":
    main()
