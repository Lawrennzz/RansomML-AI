#!/usr/bin/env python3
"""
Simple test for Enhanced Ransomware Detection System
"""

import requests
import json

def test_system():
    """Test the enhanced ransomware detection system"""
    
    base_url = "http://localhost:5000"
    
    print("Testing Enhanced Ransomware Detection System")
    print("=" * 50)
    
    # Test 1: Train models
    print("\n1. Testing model training...")
    try:
        response = requests.post(f"{base_url}/api/train")
        data = response.json()
        
        if data['success']:
            print("SUCCESS: Models trained successfully!")
            performance = data['performance']
            print(f"   Accuracy: {performance['accuracy']:.4f}")
            print(f"   F1-Score: {performance['f1_score']:.4f}")
        else:
            print(f"FAILED: Training failed: {data['message']}")
            return False
    except Exception as e:
        print(f"ERROR: Error training models: {e}")
        return False
    
    # Test 2: Test benign sample
    print("\n2. Testing benign sample detection...")
    benign_sample = {
        'file_access_count': 30,
        'file_modifications': 20,
        'file_encryptions': 0,
        'system_calls': 80,
        'process_count': 120,
        'network_connections': 10,
        'cpu_usage': 30.0,
        'memory_usage': 40.0,
        'disk_io_read': 150,
        'disk_io_write': 100,
        'registry_changes': 2,
        'entropy_change': 0.5,
        'crypto_operations': 1
    }
    
    try:
        response = requests.post(f"{base_url}/api/predict", json=benign_sample)
        data = response.json()
        
        if data['success']:
            result = data['result']
            print("SUCCESS: Benign sample analyzed!")
            print(f"   Prediction: {'Ransomware' if result['prediction'] else 'Benign'}")
            print(f"   Confidence: {result['confidence']:.4f}")
            print(f"   Risk Level: {result['risk_level']}")
        else:
            print(f"FAILED: Prediction failed: {data['message']}")
    except Exception as e:
        print(f"ERROR: Error predicting benign sample: {e}")
    
    # Test 3: Test ransomware sample
    print("\n3. Testing ransomware sample detection...")
    ransomware_sample = {
        'file_access_count': 200,
        'file_modifications': 150,
        'file_encryptions': 10,
        'system_calls': 300,
        'process_count': 250,
        'network_connections': 50,
        'cpu_usage': 90.0,
        'memory_usage': 95.0,
        'disk_io_read': 800,
        'disk_io_write': 1000,
        'registry_changes': 25,
        'entropy_change': 5.0,
        'crypto_operations': 20
    }
    
    try:
        response = requests.post(f"{base_url}/api/predict", json=ransomware_sample)
        data = response.json()
        
        if data['success']:
            result = data['result']
            print("SUCCESS: Ransomware sample analyzed!")
            print(f"   Prediction: {'Ransomware' if result['prediction'] else 'Benign'}")
            print(f"   Confidence: {result['confidence']:.4f}")
            print(f"   Risk Level: {result['risk_level']}")
        else:
            print(f"FAILED: Prediction failed: {data['message']}")
    except Exception as e:
        print(f"ERROR: Error predicting ransomware sample: {e}")
    
    print("\n" + "=" * 50)
    print("Enhanced System Testing Completed!")
    print("=" * 50)

if __name__ == "__main__":
    print("Make sure the enhanced app is running on http://localhost:5000")
    print("Run: python enhanced_app.py")
    print()
    
    try:
        test_system()
    except KeyboardInterrupt:
        print("\nTesting interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
