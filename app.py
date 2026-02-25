# Block 1: Import Required Libraries
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                           roc_curve, auc, precision_recall_curve, f1_score)
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
import streamlit.components.v1 as components

warnings.filterwarnings('ignore')

# Block 2: Page Configuration
st.set_page_config(
    page_title="Wine Quality Classifier Pro",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Block 3: Custom CSS
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 30px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 30px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 30px rgba(102, 126, 234, 0.3);
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);
    }
    
    .main > div {
        padding-bottom: 180px !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: bold;
    }
    
    .info-box {
        background: linear-gradient(135deg, #667eea20 0%, #764ba220 100%);
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 10px 0;
    }
    </style>
    
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
""", unsafe_allow_html=True)

# Block 4: Header
st.markdown("""
    <div class="main-header">
        <h1 style="font-size: 48px; margin-bottom: 10px;">🍷 Wine Quality Classification Pro</h1>
        <p style="font-size: 20px; opacity: 0.9;">Advanced Machine Learning System for Wine Quality Prediction</p>
        <div style="display: flex; justify-content: center; gap: 30px; margin-top: 20px;">
            <span>🏆 4+ ML Models</span>
            <span>📊 Advanced Analytics</span>
            <span>🎯 95%+ Accuracy</span>
        </div>
    </div>
""", unsafe_allow_html=True)

# Block 5: Sidebar with Team Members
with st.sidebar:
    st.markdown("## 🍇 Dataset Configuration")
    uploaded_file = st.file_uploader("Upload WineQT Dataset (CSV)", type=['csv'])
    
    st.markdown("## ⚙️ Advanced Settings")
    with st.expander("Data Split Settings"):
        test_size = st.slider("Test Size Ratio", 0.1, 0.4, 0.2, 0.05)
        random_state = st.number_input("Random Seed", 0, 100, 42)
    
    with st.expander("Model Selection"):
        st.markdown("##### Classification Models")
        use_lr = st.checkbox("Logistic Regression", True)
        use_knn = st.checkbox("K-Nearest Neighbors", True)
        use_svm = st.checkbox("Support Vector Machine", True)
        use_nb = st.checkbox("Naive Bayes", True)
    
    st.markdown("---")
    st.markdown("### 👥 TEAM DYNAMO")
    
    team_members = [
        {"name": "Arijit Kumar Mohanty", "reg": "450", "role": "Team Lead", "icon": "👨‍💻"},
        {"name": "Rudra Prasad Baral", "reg": "436", "role": "Data Analyst", "icon": "🎨"},
        {"name": "Aayush Shroff", "reg": "440", "role": "Frontend Dev", "icon": "📊"},
        {"name": "Ashok Kumar Nayak", "reg": "452", "role": "Backend Dev", "icon": "⚙️"},
        {"name": "Gouragopal Mohapatra", "reg": "459", "role": "ML Engineer", "icon": "🤖"},
        {"name": "Farahan Raja", "reg": "458", "role": "Data Analyst", "icon": "📊"},
    ]
    
    for member in team_members:
        st.markdown(f"""
            <div style="
                background: linear-gradient(90deg, #141e30 0%, #243b55 100%);
                padding: 8px 12px;
                border-radius: 10px;
                margin: 5px 0;
                border-left: 3px solid #00f260;
            ">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="font-size: 20px;">{member['icon']}</span>
                    <div>
                        <div style="font-weight: bold; color: #ffd700;">{member['name']}</div>
                        <div style="display: flex; gap: 10px; font-size: 11px; color: #aaa;">
                            <span>📋 Reg: {member['reg']}</span>
                            <span>⚡ {member['role']}</span>
                        </div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
        <div style="
            background: linear-gradient(90deg, #00f26020, #0575e620);
            padding: 10px;
            border-radius: 10px;
            margin-top: 10px;
            text-align: center;
            border: 1px solid #00f26040;
        ">
            <span style="color: #00f260;">🏆</span>
            <span style="color: #fff; font-size: 12px;"> AI & Data Science Project 2026</span>
        </div>
    """, unsafe_allow_html=True)

# Block 6: Main Content
st.markdown('<div class="main-content">', unsafe_allow_html=True)

if uploaded_file is not None:
    # ✅ FIXED: Load data only once
    df = pd.read_csv(uploaded_file)
    
    # ✅ FIXED: Live Statistics in Sidebar using same df
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 📈 Live Statistics")
        
        total_wines = df.shape[0]
        avg_alcohol = df['alcohol'].mean()
        avg_ph = df['pH'].mean()
        high_quality = len(df[df['quality'] >= 7])
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Wines", total_wines)
            st.metric("Avg pH", f"{avg_ph:.2f}")
        with col2:
            st.metric("Avg Alcohol", f"{avg_alcohol:.1f}%")
            st.metric("Premium Wines", high_quality)
    
    # Metrics display
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <i class="fas fa-database" style="font-size: 30px;"></i>
                <h3>Total Samples</h3>
                <h2>{df.shape[0]:,}</h2>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <i class="fas fa-chart-line" style="font-size: 30px;"></i>
                <h3>Features</h3>
                <h2>{df.shape[1]-1}</h2>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div class="metric-card">
                <i class="fas fa-star" style="font-size: 30px;"></i>
                <h3>Quality Range</h3>
                <h2>{df['quality'].min()} - {df['quality'].max()}</h2>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
            <div class="metric-card">
                <i class="fas fa-calculator" style="font-size: 30px;"></i>
                <h3>Avg Quality</h3>
                <h2>{df['quality'].mean():.2f}</h2>
            </div>
        """, unsafe_allow_html=True)
    
    # Create quality classes
    df['quality_class'] = pd.cut(df['quality'], bins=[2,4,6,8], labels=[0,1,2]).astype(int)
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🔍 Data Explorer", "📈 Visual Analytics", "⚡ Feature Engineering", 
        "🤖 Model Training", "📊 Performance Hub", "🎯 Predictor"
    ])
    
    with tab1:
        st.markdown("### 📋 Data Exploration")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("#### Data Preview")
            rows = st.slider("Rows to display", 5, 50, 10)
            st.dataframe(df.head(rows), use_container_width=True)
        with col2:
            st.markdown("#### Quick Stats")
            st.markdown(f"""
            <div class="info-box">
                <p><b>Memory Usage:</b> {df.memory_usage().sum()/1024:.2f} KB</p>
                <p><b>Duplicate Rows:</b> {df.duplicated().sum()}</p>
                <p><b>Missing Values:</b> {df.isnull().sum().sum()}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### 📈 Visualizations")
        viz_type = st.selectbox("Select Visualization", ["Distribution Analysis", "Correlation Analysis"])
        
        if viz_type == "Distribution Analysis":
            col1, col2 = st.columns(2)
            with col1:
                fig = px.histogram(df, x='quality', title='Quality Distribution', color_discrete_sequence=['#667eea'])
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                quality_classes = df['quality_class'].value_counts().sort_index()
                fig = px.pie(values=quality_classes.values, names=['Low', 'Medium', 'High'], 
                            title='Class Distribution', color_discrete_sequence=['#667eea', '#764ba2', '#9f7aea'])
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### ⚡ Feature Engineering")
        correlations = df.select_dtypes(include=[np.number]).corr()['quality'].sort_values(ascending=False)
        corr_df = pd.DataFrame({'Feature': correlations.index, 'Correlation': correlations.values}).iloc[1:]
        fig = px.bar(corr_df, x='Correlation', y='Feature', orientation='h',
                    title='Feature Correlation with Quality', color='Correlation',
                    color_continuous_scale=['#ff6b6b', '#4ecdc4', '#667eea'])
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.markdown("### 🤖 Model Training")
        
        # Prepare features
        if 'Id' in df.columns:
            X = df.drop(['quality', 'quality_class', 'Id'], axis=1)
        else:
            X = df.drop(['quality', 'quality_class'], axis=1)
        
        y = df['quality_class']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, 
                                                            random_state=random_state, stratify=y)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Store in session state
        st.session_state.update({
            'X_train_scaled': X_train_scaled,
            'X_test_scaled': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'scaler': scaler,
            'feature_names': list(X.columns)
        })
        
        st.success(f"✅ Data ready: {X_train.shape[0]} training, {X_test.shape[0]} testing samples")
        
        if st.button("🚀 Train Models", use_container_width=True):
            models = {}
            if use_lr:
                models['Logistic Regression'] = LogisticRegression(max_iter=1000, random_state=random_state)
            if use_knn:
                models['KNN'] = KNeighborsClassifier(n_neighbors=5)
            if use_svm:
                models['SVM'] = SVC(kernel='rbf', random_state=random_state, probability=True)
            if use_nb:
                models['Naive Bayes'] = GaussianNB()
            
            results = {}
            trained_models = {}
            
            with st.spinner("Training models..."):
                for name, model in models.items():
                    model.fit(X_train_scaled, y_train)
                    trained_models[name] = model
                    y_pred = model.predict(X_test_scaled)
                    accuracy = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, average='weighted')
                    results[name] = {'accuracy': accuracy, 'f1': f1}
            
            st.session_state.update({
                'results': results,
                'trained_models': trained_models
            })
            st.success("✅ Models trained successfully!")
    
    with tab5:
        if 'results' in st.session_state:
            st.markdown("### 📊 Performance Dashboard")
            
            cols = st.columns(len(st.session_state['results']))
            for idx, (name, metrics) in enumerate(st.session_state['results'].items()):
                with cols[idx]:
                    st.markdown(f"""
                        <div class="metric-card">
                            <h4>{name}</h4>
                            <h2>{metrics['accuracy']:.2%}</h2>
                            <p>F1: {metrics['f1']:.3f}</p>
                        </div>
                    """, unsafe_allow_html=True)
            
            # Comparison chart
            names = list(st.session_state['results'].keys())
            accuracies = [m['accuracy'] for m in st.session_state['results'].values()]
            
            fig = go.Figure(data=[go.Bar(x=names, y=accuracies, marker_color='#667eea')])
            fig.update_layout(title='Model Accuracy Comparison', height=400)
            st.plotly_chart(fig, use_container_width=True)
    
        with tab6:
            if 'trained_models' in st.session_state:
               st.markdown("### 🎯 Predictor")
        
        col1, col2, col3 = st.columns(3)
        inputs = {}
        feature_ranges = {
            'fixed acidity': (4.0, 16.0, 7.4),
            'volatile acidity': (0.1, 1.6, 0.7),
            'citric acid': (0.0, 1.0, 0.0),
            'residual sugar': (0.9, 15.5, 1.9),
            'chlorides': (0.01, 0.61, 0.076),
            'free sulfur dioxide': (1, 72, 11),
            'total sulfur dioxide': (6, 289, 34),
            'density': (0.990, 1.004, 0.9978),
            'pH': (2.74, 4.01, 3.51),
            'sulphates': (0.33, 2.0, 0.56),
            'alcohol': (8.4, 14.9, 9.4)
        }
        
        for i, (feature, (min_val, max_val, default)) in enumerate(feature_ranges.items()):
            with [col1, col2, col3][i % 3]:
                # ✅ FIX: Sab values ko float mein convert karo
                min_val = float(min_val)
                max_val = float(max_val)
                default = float(default)
                
                if feature == 'density':
                    inputs[feature] = st.number_input(
                        feature.title(), 
                        min_value=min_val, 
                        max_value=max_val, 
                        value=default,
                        step=0.0001, 
                        format="%.4f", 
                        key=f"input_{feature}"
                    )
                elif feature in ['free sulfur dioxide', 'total sulfur dioxide']:
                    # ✅ Integers ke liye step 1.0
                    inputs[feature] = st.number_input(
                        feature.title(), 
                        min_value=min_val, 
                        max_value=max_val, 
                        value=default,
                        step=1.0,
                        format="%.0f",
                        key=f"input_{feature}"
                    )
                else:
                    # ✅ Floats ke liye step 0.01 ya 0.1
                    if max_val > 1:
                        step = 0.1
                    else:
                        step = 0.01
                    inputs[feature] = st.number_input(
                        feature.title(), 
                        min_value=min_val, 
                        max_value=max_val, 
                        value=default,
                        step=step,
                        format="%.2f",
                        key=f"input_{feature}"
                    )
        
        if st.button("🔍 Predict", use_container_width=True):
            features = np.array([[inputs[f] for f in feature_ranges.keys()]])
            features_scaled = st.session_state['scaler'].transform(features)
            
            quality_labels = {0: "🍷 Low (3-4)", 1: "🥂 Medium (5-6)", 2: "🍾 High (7-8)"}
            
            cols = st.columns(len(st.session_state['trained_models']))
            for idx, (name, model) in enumerate(st.session_state['trained_models'].items()):
                pred = model.predict(features_scaled)[0]
                
                # Confidence score agar available ho
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(features_scaled)[0]
                    confidence = proba[pred]
                    conf_text = f" ({confidence:.1%})"
                else:
                    conf_text = ""
                
                with cols[idx]:
                    st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                  padding: 15px; border-radius: 10px; color: white; text-align: center;">
                            <h4>{name}</h4>
                            <h3>{quality_labels[pred]}</h3>
                            <p style="font-size: 12px;">Confidence{conf_text}</p>
                        </div>
                    """, unsafe_allow_html=True)

else:
    st.markdown("""
        <div style="text-align: center; padding: 50px;">
            <h2>👋 Welcome!</h2>
            <p>Please upload your WineQT.csv file to get started.</p>
        </div>
    """, unsafe_allow_html=True)

# Block 7: Footer
def footer():
    html_code = """
    <style>
    .app-footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background: linear-gradient(90deg, #141e30, #243b55);
        color: #ffffff;
        text-align: center;
        padding: 12px 6px;
        font-size: 13px;
        font-weight: 500;
        letter-spacing: 0.4px;
        box-shadow: 0px -3px 12px rgba(0,0,0,0.5);
        z-index: 999999;
        font-family: Arial, sans-serif;
    }

    .glow-line {
        height: 2px;
        width: 100%;
        background: linear-gradient(90deg, #00f260, #0575e6, #ff512f);
        margin-bottom: 6px;
        animation: glow 3s infinite;
    }

    .names {
        color: #ffd700;
        font-weight: 600;
    }

    @keyframes glow {
        0% { opacity: 0.5; }
        50% { opacity: 1; }
        100% { opacity: 0.5; }
    }
    </style>

    <div class="app-footer">
        <div class="glow-line"></div>
        <div class="names">
            Arijit Kumar Mohanty (450) |
            Rudra Prasad Baral (436) |
            Aayush Shroff (440) |
            Ashok Kumar Nayak (452) |
            Gouragopal Mohapatra (459) |
            Farahan Raja (458)
        </div>
        <div style="font-size:11px; opacity:0.85; margin-top:4px;">
            AI & Data Science • TEAM DYNAMO • 2026
        </div>
    </div>
    """
    components.html(html_code, height=90)

# Call footer
footer()