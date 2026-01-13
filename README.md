# BVIX Predictor

**Date:** 6/20/2025  

## Problem Statement
Predict Bitcoin price volatility over the next 3 hours.

---

## Dataset
- **Source:** [Btc_15min_data](https://www.cryptodatadownload.com/data/)  
- **Size:** 260,974  
- **Domain:** Tabular  

---

## Objectives
Classify future volatility into three categories based on current data:
- **0**: Low volatility  
- **1**: Medium volatility  
- **2**: High volatility  

---

## Core Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# ML libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Display settings
pd.set_option('display.max_columns', None)
np.random.seed(42)

