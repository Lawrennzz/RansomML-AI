// Ransomware Detection Dashboard JavaScript
// Main application logic and API interactions

class RansomwareDashboard {
    constructor() {
        this.charts = {};
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.loadInitialData();
    }

    setupEventListeners() {
        // Detection form submission
        document.getElementById('detectionForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.performDetection();
        });

        // Tab change events
        document.querySelectorAll('[data-bs-toggle="tab"]').forEach(tab => {
            tab.addEventListener('shown.bs.tab', (e) => {
                const target = e.target.getAttribute('data-bs-target');
                this.onTabChange(target);
            });
        });
    }

    async loadInitialData() {
        try {
            await this.loadDatasetStats();
            await this.loadModelPerformance();
            await this.loadDetectionHistory();
        } catch (error) {
            console.error('Error loading initial data:', error);
            this.showAlert('Error loading initial data', 'danger');
        }
    }

    async loadDatasetStats() {
        try {
            const response = await fetch('/api/dataset-stats');
            const data = await response.json();
            
            if (data.success) {
                const stats = data.stats;
                document.getElementById('totalSamples').textContent = stats.total_samples;
                document.getElementById('features').textContent = stats.features;
                document.getElementById('ransomwareSamples').textContent = stats.ransomware_samples;
                document.getElementById('benignSamples').textContent = stats.benign_samples;
            }
        } catch (error) {
            console.error('Error loading dataset stats:', error);
        }
    }

    async loadModelPerformance() {
        try {
            const response = await fetch('/api/model-performance');
            const data = await response.json();
            
            if (data.success) {
                const perf = data.performance;
                document.getElementById('accuracy').textContent = (perf.accuracy * 100).toFixed(1) + '%';
                document.getElementById('precision').textContent = (perf.precision * 100).toFixed(1) + '%';
                document.getElementById('recall').textContent = (perf.recall * 100).toFixed(1) + '%';
                document.getElementById('f1Score').textContent = (perf.f1_score * 100).toFixed(1) + '%';
                
                // Update model status
                document.getElementById('modelStatus').innerHTML = '<i class="fas fa-check-circle"></i>';
                
                // Create charts if not already created
                this.createFeatureImportanceChart(perf.feature_importance);
                this.createConfusionMatrixChart(perf.confusion_matrix);
            }
        } catch (error) {
            console.error('Error loading model performance:', error);
        }
    }

    async loadDetectionHistory() {
        try {
            const response = await fetch('/api/detection-history');
            const data = await response.json();
            
            if (data.success) {
                this.displayDetectionHistory(data.history);
                document.getElementById('totalDetections').textContent = data.history.length;
            }
        } catch (error) {
            console.error('Error loading detection history:', error);
        }
    }

    async trainModel() {
        this.showLoading(true);
        
        try {
            const response = await fetch('/api/train', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.showAlert('Model trained successfully!', 'success');
                await this.loadModelPerformance();
            } else {
                this.showAlert('Training failed: ' + data.message, 'danger');
            }
        } catch (error) {
            console.error('Error training model:', error);
            this.showAlert('Error training model', 'danger');
        } finally {
            this.showLoading(false);
        }
    }

    async performDetection() {
        this.showLoading(true);
        
        try {
            // Collect form data
            const features = {
                file_access_count: parseInt(document.getElementById('file_access_count').value),
                entropy_change: parseFloat(document.getElementById('entropy_change').value),
                system_calls: parseInt(document.getElementById('system_calls').value),
                network_connections: parseInt(document.getElementById('network_connections').value),
                file_modifications: parseInt(document.getElementById('file_modifications').value),
                cpu_usage: parseFloat(document.getElementById('cpu_usage').value),
                memory_usage: parseFloat(document.getElementById('memory_usage').value),
                disk_io: parseInt(document.getElementById('disk_io').value),
                process_count: parseInt(document.getElementById('process_count').value),
                registry_changes: parseInt(document.getElementById('registry_changes').value)
            };

            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(features)
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.displayDetectionResult(data.result);
                await this.loadDetectionHistory(); // Refresh history
            } else {
                this.showAlert('Detection failed: ' + data.message, 'danger');
            }
        } catch (error) {
            console.error('Error performing detection:', error);
            this.showAlert('Error performing detection', 'danger');
        } finally {
            this.showLoading(false);
        }
    }

    displayDetectionResult(result) {
        const resultDiv = document.getElementById('detectionResult');
        const noDetectionDiv = document.getElementById('noDetection');
        const resultTitle = document.getElementById('resultTitle');
        const resultConfidence = document.getElementById('resultConfidence');
        const resultRisk = document.getElementById('resultRisk');

        // Hide no detection message
        noDetectionDiv.style.display = 'none';
        
        // Show result
        resultDiv.style.display = 'block';
        
        if (result.prediction === 1) {
            resultDiv.className = 'detection-result ransomware';
            resultTitle.innerHTML = '<i class="fas fa-exclamation-triangle"></i> RANSOMWARE DETECTED!';
        } else {
            resultDiv.className = 'detection-result benign';
            resultTitle.innerHTML = '<i class="fas fa-check-circle"></i> System Appears Benign';
        }
        
        resultConfidence.innerHTML = `<strong>Confidence:</strong> ${(result.confidence * 100).toFixed(2)}%`;
        resultRisk.innerHTML = `<strong>Risk Level:</strong> ${result.risk_level}`;
        
        // Show additional details
        const details = `
            <div class="mt-3">
                <small>
                    <strong>Benign Probability:</strong> ${(result.benign_probability * 100).toFixed(2)}%<br>
                    <strong>Ransomware Probability:</strong> ${(result.ransomware_probability * 100).toFixed(2)}%
                </small>
            </div>
        `;
        resultDiv.innerHTML += details;
    }

    displayDetectionHistory(history) {
        const historyDiv = document.getElementById('detectionHistory');
        
        if (history.length === 0) {
            historyDiv.innerHTML = `
                <div class="text-center text-muted">
                    <i class="fas fa-clock fa-3x mb-3"></i>
                    <p>No detection history available yet</p>
                </div>
            `;
            return;
        }
        
        let historyHTML = '';
        history.reverse().forEach(item => {
            const result = item.result;
            const features = item.features;
            const timestamp = new Date(result.timestamp).toLocaleString();
            
            const isRansomware = result.prediction === 1;
            const itemClass = isRansomware ? 'ransomware' : 'benign';
            const icon = isRansomware ? 'fa-exclamation-triangle' : 'fa-check-circle';
            const status = isRansomware ? 'RANSOMWARE' : 'BENIGN';
            
            historyHTML += `
                <div class="history-item ${itemClass}">
                    <div class="d-flex justify-content-between align-items-center">
                        <div>
                            <h6><i class="fas ${icon}"></i> ${status}</h6>
                            <small class="text-muted">${timestamp}</small>
                        </div>
                        <div class="text-end">
                            <div><strong>Confidence:</strong> ${(result.confidence * 100).toFixed(1)}%</div>
                            <div><strong>Risk:</strong> ${result.risk_level}</div>
                        </div>
                    </div>
                    <div class="mt-2">
                        <small>
                            <strong>Key Metrics:</strong>
                            CPU: ${features.cpu_usage}% | 
                            Memory: ${features.memory_usage}% | 
                            File Access: ${features.file_access_count} | 
                            System Calls: ${features.system_calls}
                        </small>
                    </div>
                </div>
            `;
        });
        
        historyDiv.innerHTML = historyHTML;
    }

    createFeatureImportanceChart(featureImportance) {
        const ctx = document.getElementById('featureImportanceChart');
        if (!ctx) return;
        
        // Destroy existing chart if it exists
        if (this.charts.featureImportance) {
            this.charts.featureImportance.destroy();
        }
        
        const features = Object.keys(featureImportance);
        const importance = Object.values(featureImportance);
        
        this.charts.featureImportance = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: features.map(f => f.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())),
                datasets: [{
                    label: 'Importance',
                    data: importance,
                    backgroundColor: 'rgba(102, 126, 234, 0.8)',
                    borderColor: 'rgba(102, 126, 234, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Importance Score'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Features'
                        }
                    }
                }
            }
        });
    }

    createConfusionMatrixChart(confusionMatrix) {
        const ctx = document.getElementById('confusionMatrixChart');
        if (!ctx) return;
        
        // Destroy existing chart if it exists
        if (this.charts.confusionMatrix) {
            this.charts.confusionMatrix.destroy();
        }
        
        this.charts.confusionMatrix = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['True Negative', 'False Positive', 'False Negative', 'True Positive'],
                datasets: [{
                    label: 'Count',
                    data: [confusionMatrix[0][0], confusionMatrix[0][1], confusionMatrix[1][0], confusionMatrix[1][1]],
                    backgroundColor: [
                        'rgba(81, 207, 102, 0.8)',  // True Negative - Green
                        'rgba(255, 193, 7, 0.8)',   // False Positive - Yellow
                        'rgba(255, 107, 107, 0.8)', // False Negative - Red
                        'rgba(102, 126, 234, 0.8)'  // True Positive - Blue
                    ],
                    borderColor: [
                        'rgba(81, 207, 102, 1)',
                        'rgba(255, 193, 7, 1)',
                        'rgba(255, 107, 107, 1)',
                        'rgba(102, 126, 234, 1)'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Count'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Confusion Matrix Elements'
                        }
                    }
                }
            }
        });
    }

    onTabChange(targetTab) {
        switch (targetTab) {
            case '#analytics':
                // Refresh charts when analytics tab is opened
                setTimeout(() => {
                    if (this.charts.featureImportance) {
                        this.charts.featureImportance.resize();
                    }
                    if (this.charts.confusionMatrix) {
                        this.charts.confusionMatrix.resize();
                    }
                }, 100);
                break;
            case '#history':
                this.loadDetectionHistory();
                break;
        }
    }

    showLoading(show) {
        const loadingModal = document.getElementById('loadingModal');
        loadingModal.style.display = show ? 'block' : 'none';
    }

    showAlert(message, type) {
        // Remove existing alerts
        const existingAlerts = document.querySelectorAll('.alert');
        existingAlerts.forEach(alert => alert.remove());
        
        // Create new alert
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
        alertDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        // Insert at the top of the main container
        const mainContainer = document.querySelector('.main-container');
        mainContainer.insertBefore(alertDiv, mainContainer.firstChild);
        
        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.remove();
            }
        }, 5000);
    }
}

// Initialize the dashboard when the page loads
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new RansomwareDashboard();
});

// Global function for training model (called from HTML)
function trainModel() {
    if (window.dashboard) {
        window.dashboard.trainModel();
    }
}
