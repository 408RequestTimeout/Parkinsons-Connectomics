# parkinsons_connectomics.py
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import datasets, connectome, plotting
from nilearn.connectome import ConnectivityMeasure
from nilearn.input_data import NiftiLabelsMasker
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class ParkinsonsConnectomeAnalyzer:
    """Advanced Parkinson's Disease Functional Connectomics Analysis"""
    
    def __init__(self):
        self.atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
        self.regions = self.atlas.labels
        self.pd_subtypes = ['Tremor-Dominant', 'Akinetic-Rigid', 'Mixed']
        
    def compute_functional_connectivity(self, time_series_data):
        """Compute functional connectivity matrices"""
        connectivity_measure = ConnectivityMeasure(
            kind='correlation', 
            vectorize=False, 
            discard_diagonal=True
        )
        connectivity_matrices = connectivity_measure.fit_transform(time_series_data)
        return connectivity_matrices
    
    def extract_dynamic_connectivity_features(self, time_series, window_size=30, step_size=10):
        """Extract dynamic functional connectivity features"""
        n_timepoints, n_regions = time_series.shape
        dynamic_features = []
        
        for start in range(0, n_timepoints - window_size, step_size):
            end = start + window_size
            window_data = time_series[start:end, :]
            
            # Compute connectivity for this window
            window_connectivity = np.corrcoef(window_data.T)
            dynamic_features.append(window_connectivity)
        
        return np.array(dynamic_features)
    
    def calculate_network_metrics(self, connectivity_matrix):
        """Calculate comprehensive network metrics"""
        metrics = {}
        
        # Global network metrics
        metrics['global_efficiency'] = self._compute_global_efficiency(connectivity_matrix)
        metrics['local_efficiency'] = self._compute_local_efficiency(connectivity_matrix)
        metrics['modularity'] = self._compute_modularity(connectivity_matrix)
        metrics['small_worldness'] = self._compute_small_worldness(connectivity_matrix)
        
        # Nodal metrics
        metrics['degree_centrality'] = self._compute_degree_centrality(connectivity_matrix)
        metrics['betweenness_centrality'] = self._compute_betweenness_centrality(connectivity_matrix)
        metrics['eigenvector_centrality'] = self._compute_eigenvector_centrality(connectivity_matrix)
        
        return metrics
    
    def predict_parkinsons_subtype(self, connectivity_features, clinical_features):
        """Predict Parkinson's disease subtypes using multimodal features"""
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.preprocessing import StandardScaler
        
        # Combine imaging and clinical features
        X = np.column_stack([connectivity_features.flatten(), clinical_features])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train classifier (in practice, this would use labeled training data)
        clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
        
        return clf, scaler

# Advanced visualization and reporting
class ConnectomicsVisualizer:
    """Advanced visualization for connectomics results"""
    
    def plot_connectivity_matrix(self, matrix, regions, title="Functional Connectivity"):
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(matrix, cmap='RdBu_r', aspect='auto')
        ax.set_xticks(range(len(regions)))
        ax.set_yticks(range(len(regions)))
        ax.set_xticklabels(regions, rotation=90, fontsize=8)
        ax.set_yticklabels(regions, fontsize=8)
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.colorbar(im)
        plt.tight_layout()
        return fig