import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json

# Page configuration
st.set_page_config(
    page_title="Factory Sensor Anomaly Detection",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 2rem 0;
    }
    .faulty-box {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
        color: white;
        box-shadow: 0 10px 30px rgba(238, 90, 111, 0.3);
    }
    .normal-box {
        background: linear-gradient(135deg, #51cf66 0%, #37b24d 100%);
        color: white;
        box-shadow: 0 10px 30px rgba(55, 178, 77, 0.3);
    }
    .info-box {
        background-color: #95bfde;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1c7ed6;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem;
        font-size: 1.1rem;
        border-radius: 8px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    </style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    try:
        with open('factorysensors.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("❌ Model file 'factorysensor.pkl' not found. Please ensure it's in the same directory.")
        return None

# Feature information
FEATURE_INFO = {
    'temperature': {
        'name': 'Temperature (°C)',
        'min': 20, 'max': 100, 'default': 60,
        'help': 'Operating temperature of the equipment'
    },
    'vibration': {
        'name': 'Vibration Level',
        'min': 0, 'max': 10, 'default': 5,
        'help': 'Vibration intensity measurement'
    },
    'pressure': {
        'name': 'Pressure (PSI)',
        'min': 0, 'max': 200, 'default': 100,
        'help': 'Operating pressure level'
    },
    'humidity': {
        'name': 'Humidity (%)',
        'min': 0, 'max': 100, 'default': 50,
        'help': 'Relative humidity percentage'
    },
    'power_consumption': {
        'name': 'Power Consumption (kW)',
        'min': 0, 'max': 500, 'default': 250,
        'help': 'Power usage in kilowatts'
    },
    'equipment': {
        'name': 'Equipment Type',
        'options': ['Compressor', 'Turbine', 'Pump'],
        'help': 'Type of manufacturing equipment'
    }
}

def create_gauge_chart(probability, threshold=0.35):
    """Create a gauge chart for probability visualization"""
    
    color = "red" if probability >= threshold else "green"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Fault Probability (%)", 'font': {'size': 24}},
        delta = {'reference': threshold * 100, 'suffix': '%'},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, threshold * 100], 'color': '#d4edda'},
                {'range': [threshold * 100, 100], 'color': '#f8d7da'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': threshold * 100
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig

def create_feature_importance_chart(features_dict):
    """Create a horizontal bar chart for feature contributions"""
    
    df = pd.DataFrame(list(features_dict.items()), columns=['Feature', 'Value'])
    df = df.sort_values('Value', ascending=True)
    
    fig = px.bar(df, x='Value', y='Feature', orientation='h',
                 title='Current Sensor Values',
                 color='Value',
                 color_continuous_scale='RdYlGn_r',
                 text='Value')
    
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(
        height=400,
        showlegend=False,
        xaxis_title="Value",
        yaxis_title="Sensor",
        font=dict(size=12)
    )
    
    return fig

def main():
    # Header
    st.markdown('<div class="main-header">🏭 Factory Sensor Anomaly Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Equipment Fault Prediction System</div>', unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    
    if model is None:
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/factory.png", width=80)
        st.title("⚙️ Input Parameters")
        st.markdown("---")
        
        input_method = st.radio(
            "Select Input Method:",
            ["📝 Manual Entry", "📁 Upload CSV"],
            help="Choose how to provide sensor data"
        )
        st.markdown("---")
        threshold = 0.35
        # threshold = st.slider(
        #     "Detection Threshold",
        #     min_value=0.1,
        #     max_value=0.9,
        #     value=0.35,
        #     step=0.05,
        #     help="Probability threshold for fault classification"
        # )
        
        # st.markdown("---")
        # st.info("💡 **Tip:** Lower threshold = More sensitive detection")
    
    # Main content
    if input_method == "📝 Manual Entry":
        st.subheader("Enter Sensor Readings")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            temperature = st.number_input(
                FEATURE_INFO['temperature']['name'],
                min_value=float(FEATURE_INFO['temperature']['min']),
                max_value=float(FEATURE_INFO['temperature']['max']),
                value=float(FEATURE_INFO['temperature']['default']),
                help=FEATURE_INFO['temperature']['help']
            )
            
            vibration = st.number_input(
                FEATURE_INFO['vibration']['name'],
                min_value=float(FEATURE_INFO['vibration']['min']),
                max_value=float(FEATURE_INFO['vibration']['max']),
                value=float(FEATURE_INFO['vibration']['default']),
                help=FEATURE_INFO['vibration']['help']
            )
        
        with col2:
            pressure = st.number_input(
                FEATURE_INFO['pressure']['name'],
                min_value=float(FEATURE_INFO['pressure']['min']),
                max_value=float(FEATURE_INFO['pressure']['max']),
                value=float(FEATURE_INFO['pressure']['default']),
                help=FEATURE_INFO['pressure']['help']
            )
            
            humidity = st.number_input(
                FEATURE_INFO['humidity']['name'],
                min_value=float(FEATURE_INFO['humidity']['min']),
                max_value=float(FEATURE_INFO['humidity']['max']),
                value=float(FEATURE_INFO['humidity']['default']),
                help=FEATURE_INFO['humidity']['help']
            )
        
        with col3:
            # power_consumption = st.number_input(
            #     FEATURE_INFO['power_consumption']['name'],
            #     min_value=float(FEATURE_INFO['power_consumption']['min']),
            #     max_value=float(FEATURE_INFO['power_consumption']['max']),
            #     value=float(FEATURE_INFO['power_consumption']['default']),
            #     help=FEATURE_INFO['power_consumption']['help']
            # )
            
            equipment = st.selectbox(
                FEATURE_INFO['equipment']['name'],
                options=FEATURE_INFO['equipment']['options'],
                help=FEATURE_INFO['equipment']['help']
            )
        
        if st.button("🔍 Predict Equipment Status", key="predict_manual"):
            input_data = pd.DataFrame({
                'temperature': [temperature],
                'vibration': [vibration],
                'pressure': [pressure],
                'humidity': [humidity],
                # 'power_consumption': [power_consumption],
                'equipment': [equipment]
            })
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]
            
            # Display results
            st.markdown("---")
            st.subheader("📊 Prediction Results")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                if probability >= threshold:
                    st.markdown(f'<div class="prediction-box faulty-box">⚠️ EQUIPMENT FAULTY<br><span style="font-size: 1rem;">Immediate Attention Required</span></div>', 
                               unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="prediction-box normal-box">✅ EQUIPMENT NORMAL<br><span style="font-size: 1rem;">Operating Within Parameters</span></div>', 
                               unsafe_allow_html=True)
                
                # Metrics
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Fault Probability", f"{probability*100:.1f}%")
                with col_b:
                    st.metric("Threshold", f"{threshold*100:.0f}%")
                with col_c:
                    confidence = abs(probability - threshold) / threshold * 100
                    st.metric("Confidence", f"{min(confidence, 100):.1f}%")
            
            with col2:
                # Gauge chart
                gauge_fig = create_gauge_chart(probability, threshold)
                st.plotly_chart(gauge_fig, use_container_width=True)
            
            # Feature values chart
            st.subheader("📈 Current Sensor Readings")
            feature_dict = {
                'Temperature': temperature,
                'Vibration': vibration,
                'Pressure': pressure,
                'Humidity': humidity,
                # 'Power': power_consumption
            }
            feature_chart = create_feature_importance_chart(feature_dict)
            st.plotly_chart(feature_chart, use_container_width=True)
            
            # Recommendations
            st.subheader("💡 Recommendations")
            if probability >= threshold:
                st.error("🔧 **Action Required:**")
                st.markdown("""
                - Schedule immediate maintenance inspection
                - Monitor equipment closely for next 24 hours
                - Check sensor calibration
                - Review recent operational changes
                - Document current readings for maintenance records
                """)
            else:
                st.success("✅ **System Status:**")
                st.markdown("""
                - Equipment operating normally
                - Continue regular monitoring schedule
                - Next scheduled maintenance as planned
                - All parameters within acceptable range
                """)
    
    elif input_method == "📁 Upload CSV":
        st.subheader("Batch Prediction from CSV File")
        
        st.markdown('<div class="info-box">📋 <b>CSV Format Required:</b> Columns: temperature, vibration, pressure, humidity, power_consumption, equipment</div>', 
                   unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                input_df = pd.read_csv(uploaded_file)
                
                st.success(f"✅ Loaded {len(input_df)} records")
                
                # Show preview
                with st.expander("👀 Preview Data"):
                    st.dataframe(input_df.head(10))
                
                if st.button("🔍 Predict All Records", key="predict_batch"):
                    # Make predictions
                    predictions = model.predict(input_df)
                    probabilities = model.predict_proba(input_df)[:, 1]
                    
                    # Add results to dataframe
                    results_df = input_df.copy()
                    results_df['fault_probability'] = probabilities
                    results_df['prediction'] = ['Faulty' if p >= threshold else 'Normal' for p in probabilities]
                    results_df['status'] = ['⚠️' if p >= threshold else '✅' for p in probabilities]
                    
                    # Summary statistics
                    st.markdown("---")
                    st.subheader("📊 Batch Analysis Summary")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Records", len(results_df))
                    with col2:
                        faulty_count = (probabilities >= threshold).sum()
                        st.metric("Faulty Equipment", faulty_count, 
                                 delta=f"{faulty_count/len(results_df)*100:.1f}%")
                    with col3:
                        normal_count = (probabilities < threshold).sum()
                        st.metric("Normal Equipment", normal_count,
                                 delta=f"{normal_count/len(results_df)*100:.1f}%")
                    with col4:
                        avg_prob = probabilities.mean()
                        st.metric("Avg Fault Prob", f"{avg_prob*100:.1f}%")
                    
                    # Distribution chart
                    st.subheader("📈 Probability Distribution")
                    fig = px.histogram(results_df, x='fault_probability', 
                                      color='prediction',
                                      nbins=30,
                                      title='Distribution of Fault Probabilities',
                                      color_discrete_map={'Faulty': '#e74c3c', 'Normal': '#2ecc71'})
                    fig.add_vline(x=threshold, line_dash="dash", line_color="black", 
                                 annotation_text=f"Threshold: {threshold}")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Results table
                    st.subheader("📋 Detailed Results")
                    st.dataframe(
                        results_df.style.apply(
                            lambda x: ['background-color: #f8d7da' if v >= threshold else 'background-color: #d4edda' 
                                      for v in x], 
                            subset=['fault_probability']
                        ),
                        use_container_width=True
                    )
                    
                    # Download button
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results CSV",
                        data=csv,
                        file_name="fault_predictions.csv",
                        mime="text/csv"
                    )
                    
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
    
    # else:  # Random Sample
    #     st.subheader("Generate Random Test Sample")
        
    #     st.info("🎲 This will generate random sensor readings within realistic ranges for testing.")
        
    #     num_samples = st.slider("Number of samples to generate", 1, 20, 5)
        
    #     if st.button("🎲 Generate & Predict", key="predict_random"):
    #         # Generate random data
    #         np.random.seed(42)
    #         random_data = pd.DataFrame({
    #             'temperature': np.random.uniform(20, 100, num_samples),
    #             'vibration': np.random.uniform(0, 10, num_samples),
    #             'pressure': np.random.uniform(0, 200, num_samples),
    #             'humidity': np.random.uniform(0, 100, num_samples),
    #             'power_consumption': np.random.uniform(0, 500, num_samples),
    #             'equipment': np.random.choice(FEATURE_INFO['equipment']['options'], num_samples)
    #         })
            
    #         # Make predictions
    #         predictions = model.predict(random_data)
    #         probabilities = model.predict_proba(random_data)[:, 1]
            
    #         # Add results
    #         results_df = random_data.copy()
    #         results_df['fault_probability'] = probabilities
    #         results_df['prediction'] = ['Faulty' if p >= threshold else 'Normal' for p in probabilities]
    #         results_df['status'] = ['⚠️' if p >= threshold else '✅' for p in probabilities]
            
    #         # Display results
    #         st.markdown("---")
    #         st.subheader("📊 Generated Sample Results")
            
    #         col1, col2, col3 = st.columns(3)
    #         with col1:
    #             faulty_count = (probabilities >= threshold).sum()
    #             st.metric("Faulty", faulty_count)
    #         with col2:
    #             normal_count = (probabilities < threshold).sum()
    #             st.metric("Normal", normal_count)
    #         with col3:
    #             st.metric("Avg Probability", f"{probabilities.mean()*100:.1f}%")
            
    #         st.dataframe(
    #             results_df.style.apply(
    #                 lambda x: ['background-color: #f8d7da' if v >= threshold else 'background-color: #d4edda' 
    #                           for v in x], 
    #                 subset=['fault_probability']
    #             ),
    #             use_container_width=True
    #         )

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p><b>Factory Sensor Anomaly Detection System</b></p>
        <p>Powered by Machine Learning | Built with Streamlit</p>
        <p>⚠️ For production use, ensure regular model retraining and validation</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()