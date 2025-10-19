import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Heart Disease Predictor",
    layout="centered",
    page_icon="❤️",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling and visibility
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .section-header {
        font-size: 1.4rem;
        color: #2E86AB;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-bottom: 3px solid #2E86AB;
        padding-bottom: 0.5rem;
        font-weight: 600;
    }
    .info-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .prediction-high {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        border: 3px solid #ff4757;
        text-align: center;
        font-size: 1.6rem;
        font-weight: 700;
        margin: 1rem 0;
        box-shadow: 0 6px 20px rgba(255, 107, 107, 0.3);
    }
    .prediction-low {
        background: linear-gradient(135deg, #1dd1a1 0%, #10ac84 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        border: 3px solid #00b894;
        text-align: center;
        font-size: 1.6rem;
        font-weight: 700;
        margin: 1rem 0;
        box-shadow: 0 6px 20px rgba(29, 209, 161, 0.3);
    }
    .risk-factor-box {
        background-color: #2E86AB;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #FF6B6B;
        margin: 0.5rem 0;
        font-weight: 500;
    }
    .recommendation-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1dd1a1;
        margin: 1rem 0;
        border: 1px solid #dee2e6;
    }
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-size: 1.1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 1rem;
        color: #6c757d;
        border-top: 1px solid #dee2e6;
        font-size: 0.9rem;
    }
    .parameter-help {
        color: #6c757d;
        font-size: 0.85rem;
        font-style: italic;
        margin-top: 0.25rem;
    }
    .result-section {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #e9ecef;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def load_model_files():
    """Load model files with proper error handling"""
    try:
        model = joblib.load('heart_model.pkl')
        scaler = joblib.load('scaler.pkl')
        model_columns = joblib.load('model_columns.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
        return model, scaler, model_columns, label_encoders
    except FileNotFoundError as e:
        st.error(f"❌ Model file missing: {e}")
        st.info("🔧 Please run train.py first to generate model files")
        return None, None, None, None

def main():
    # Header section
    st.markdown('<div class="main-header">❤️ Heart Disease Predictor</div>', unsafe_allow_html=True)
    
    # Load model files
    model, scaler, model_columns, label_encoders = load_model_files()
    
    if model is None:
        return

    # Introduction box
    st.markdown("""
    <div class="info-box">
        <h3 style='color: white; margin: 0;'>Welcome to Heart Disease Risk Assessment</h3>
        <p style='color: white; margin: 0.5rem 0 0 0;'>
        This intelligent tool analyzes your health parameters to assess heart disease risk using clinical data.
        Provide accurate information for the most reliable assessment.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Create main interface
    display_input_section(label_encoders)
    
    # Display prediction results immediately below inputs
    if hasattr(st.session_state, 'prediction_made') and st.session_state.prediction_made:
        display_prediction_results(model, scaler, model_columns, label_encoders)
    
    # Footer
    display_footer()

def display_input_section(label_encoders):
    """Display all input parameters in an organized manner"""
    
    st.markdown('<div class="section-header">👤 Patient Health Parameters</div>', unsafe_allow_html=True)
    
    # Demographic Information
    st.subheader("Personal Information")
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("**Age**", 1, 120, 50, help="Your current age in years")
        st.markdown('<div class="parameter-help">Risk increases with age, especially after 45 for men and 55 for women</div>', unsafe_allow_html=True)
        
    with col2:
        sex = st.selectbox("**Biological Sex**", label_encoders['Sex'].classes_, help="Select your biological sex")
        st.markdown('<div class="parameter-help">Men generally have higher heart disease risk at younger ages</div>', unsafe_allow_html=True)

    # Heart & Blood Parameters
    st.subheader("Cardiac Measurements")
    col3, col4 = st.columns(2)
    
    with col3:
        restbp = st.slider("**Resting Blood Pressure (mm Hg)**", 50, 200, 120, help="Your resting blood pressure measurement")
        st.markdown('<div class="parameter-help">Normal: <120/80 mm Hg | Elevated: 120-129/<80 mm Hg</div>', unsafe_allow_html=True)
        
    with col4:
        chol = st.slider("**Cholesterol Level (mg/dL)**", 100, 600, 200, help="Your serum cholesterol level")
        st.markdown('<div class="parameter-help">Desirable: <200 mg/dL | Borderline: 200-239 mg/dL | High: ≥240 mg/dL</div>', unsafe_allow_html=True)

    # Symptoms and Tests
    st.subheader("Symptoms & Test Results")
    col5, col6 = st.columns(2)
    
    with col5:
        cp = st.selectbox("**Chest Pain Type**", label_encoders['ChestPainType'].classes_, help="Type of chest pain you experience")
        st.markdown('<div class="parameter-help">Typical angina suggests higher coronary artery disease risk</div>', unsafe_allow_html=True)
        
        fbs = st.selectbox("**Fasting Blood Sugar**", ['0', '1'], help="Fasting blood sugar > 120 mg/dL")
        st.markdown('<div class="parameter-help">>120 mg/dL indicates potential diabetes risk</div>', unsafe_allow_html=True)
    
    with col6:
        restecg = st.selectbox("**Resting ECG Results**", label_encoders['RestingECG'].classes_, help="Results from your resting electrocardiogram")
        st.markdown('<div class="parameter-help">Abnormal ECG may indicate heart muscle or rhythm issues</div>', unsafe_allow_html=True)
        
        exang = st.selectbox("**Exercise-Induced Angina**", label_encoders['ExerciseAngina'].classes_, help="Chest pain during physical activity")
        st.markdown('<div class="parameter-help">Chest pain during exercise suggests coronary artery disease</div>', unsafe_allow_html=True)

    # Exercise Test Parameters
    st.subheader("Exercise Stress Test Results")
    col7, col8 = st.columns(2)
    
    with col7:
        maxhr = st.slider("**Maximum Heart Rate Achieved**", 60, 220, 150, help="Highest heart rate during exercise")
        st.markdown('<div class="parameter-help">Estimated max HR = 220 - Age. Lower values may indicate fitness issues</div>', unsafe_allow_html=True)
        
    with col8:
        oldpeak = st.slider("**ST Depression (Oldpeak)**", 0.0, 10.0, 1.0, step=0.1, help="ST segment depression during exercise")
        st.markdown('<div class="parameter-help">Higher values indicate potential myocardial ischemia</div>', unsafe_allow_html=True)

    # Advanced Parameters
    st.subheader("Advanced Medical Parameters")
    col9, col10, col11 = st.columns(3)
    
    with col9:
        slope = st.selectbox("**ST Slope during Exercise**", label_encoders['ST_Slope'].classes_, help="Slope of ST segment during peak exercise")
        st.markdown('<div class="parameter-help">Downsloping suggests coronary artery disease</div>', unsafe_allow_html=True)
    
    with col10:
        ca = st.slider("**Number of Major Vessels**", 0, 3, 0, help="Number of major vessels colored by fluoroscopy")
        st.markdown('<div class="parameter-help">More vessels colored indicates more severe disease</div>', unsafe_allow_html=True)
    
    with col11:
        thal = st.slider("**Thalassemia Result**", 0, 3, 1, help="Thalassemia test result")
        st.markdown('<div class="parameter-help">Measures blood flow adequacy to heart muscle</div>', unsafe_allow_html=True)

    # Store all inputs in session state
    st.session_state.user_inputs = {
        'Age': age,
        'Sex': sex,
        'ChestPainType': cp,
        'RestingBP': restbp,
        'Cholesterol': chol,
        'FastingBS': int(fbs),
        'RestingECG': restecg,
        'MaxHR': maxhr,
        'ExerciseAngina': exang,
        'Oldpeak': oldpeak,
        'ST_Slope': slope,
        'Ca': ca,
        'Thal': thal
    }

    # Predict button at the bottom of input section
    if st.button("🔍 Analyze Heart Disease Risk", type="primary", use_container_width=True):
        st.session_state.prediction_made = True
        st.rerun()

def display_prediction_results(model, scaler, model_columns, label_encoders):
    """Display prediction results immediately below inputs"""
    
    st.markdown("---")
    st.markdown('<div class="section-header">📊 Prediction Result</div>', unsafe_allow_html=True)
    
    try:
        # Prepare input data
        input_df = pd.DataFrame([st.session_state.user_inputs])
        
        # Encode categorical variables
        for col, le in label_encoders.items():
            if col in input_df.columns:
                input_df[col] = le.transform(input_df[col])
        
        # Ensure correct column order
        input_df = input_df[model_columns]
        
        # Make prediction
        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)[0]
        prediction_probability = model.predict_proba(scaled_input)[0]
        
        # Display main prediction result
        display_main_prediction(prediction, prediction_probability)
        
        # Display risk factors analysis
        display_risk_factors()
        
        # Display recommendations
        display_health_recommendations(prediction)
        
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        st.info("Please check if all input parameters are correctly filled.")

def display_main_prediction(prediction, probability):
    """Display the main prediction result"""
    
    risk_percentage = probability[1] * 100
    
    st.subheader("🎯 Risk Assessment Summary")
    
    if prediction == 1:
        st.markdown(f"""
        <div class="prediction-high">
        ⚠️ HIGH RISK DETECTED
        <br>
        <span style='font-size: 1.2rem;'>Risk Probability: {risk_percentage:.1f}%</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.warning("""
        **Clinical Interpretation:** 
        Based on your health parameters, there is a significant probability of underlying heart disease. 
        This assessment suggests the need for immediate medical attention and comprehensive cardiac evaluation.
        """)
    else:
        st.markdown(f"""
        <div class="prediction-low">
        ✅ LOW RISK PROFILE
        <br>
        <span style='font-size: 1.2rem;'>Risk Probability: {risk_percentage:.1f}%</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.success("""
        **Clinical Interpretation:** 
        Your health parameters indicate a low probability of heart disease. 
        Continue maintaining healthy lifestyle habits and regular preventive checkups.
        """)

def display_risk_factors():
    """Analyze and display identified risk factors with proper visibility"""
    
    inputs = st.session_state.user_inputs
    risk_factors = []
    
    # Analyze risk factors based on medical guidelines
    if inputs['Age'] > 45 and inputs['Sex'] == 'M':
        risk_factors.append(f"Age above 45 (Male) - Current: {inputs['Age']} years")
    elif inputs['Age'] > 55 and inputs['Sex'] == 'F':
        risk_factors.append(f"Age above 55 (Female) - Current: {inputs['Age']} years")
    
    if inputs['RestingBP'] >= 130:
        risk_factors.append(f"Elevated blood pressure - Current: {inputs['RestingBP']} mm Hg")
    
    if inputs['Cholesterol'] >= 200:
        risk_factors.append(f"Elevated cholesterol - Current: {inputs['Cholesterol']} mg/dL")
    
    if inputs['FastingBS'] == 1:
        risk_factors.append("Impaired fasting glucose (>120 mg/dL)")
    
    if inputs['ExerciseAngina'] == 'Y':
        risk_factors.append("Exercise-induced angina (chest pain)")
    
    if inputs['Oldpeak'] >= 1.5:
        risk_factors.append(f"Significant ST depression - Value: {inputs['Oldpeak']}")
    elif inputs['Oldpeak'] > 0:
        risk_factors.append(f"Minor ST depression - Value: {inputs['Oldpeak']}")
    
    if inputs['Ca'] > 0:
        risk_factors.append(f"Fluoroscopy shows {inputs['Ca']} major vessel(s) affected")
    
    if inputs['ChestPainType'] in ['ATA', 'TA']:  # Atypical Angina, Typical Angina
        risk_factors.append(f"Symptomatic chest pain type: {inputs['ChestPainType']}")

    # Display risk factors section
    st.subheader("🔍 Identified Risk Factors")
    
    if risk_factors:
        st.info("The following risk factors were identified in your profile:")
        for factor in risk_factors:
            st.markdown(f'<div class="risk-factor-box">⚠️ {factor}</div>', unsafe_allow_html=True)
        
        st.metric("Total Risk Factors Identified", len(risk_factors))
    else:
        st.success("🎉 Excellent! No significant risk factors were identified in your profile.")

def display_health_recommendations(prediction):
    """Display personalized health recommendations"""
    
    st.subheader("💡 Health Recommendations & Next Steps")
    
    if prediction == 1:
        with st.expander("🚨 **Immediate Actions Required**", expanded=True):
            st.markdown("""
            ### 🏥 Medical Consultation (Urgent)
            - **Schedule cardiology appointment** within 1-2 weeks
            - **Complete blood work**: Lipid profile, HbA1c, CRP
            - **Diagnostic tests**: Stress ECG, Echocardiogram, Coronary CT Angiography
            - **Medication review** with healthcare provider
            
            ### 🏃 Lifestyle Modifications
            - **Dietary changes**: Reduce sodium, saturated fats, processed foods
            - **Exercise routine**: Start with 30 minutes walking daily
            - **Weight management**: Target BMI under 25
            - **Smoking cessation**: Complete tobacco avoidance
            - **Alcohol moderation**: Limit to 1 drink per day
            - **Stress management**: Meditation, yoga, adequate sleep
            """)
    else:
        with st.expander("🛡️ **Preventive Health Strategy**", expanded=True):
            st.markdown("""
            ### ✅ Maintenance Plan
            - **Annual health checkups** with complete cardiac screening
            - **Regular monitoring** of blood pressure and cholesterol
            - **Maintain healthy weight** with BMI 18.5-24.9
            - **Balanced nutrition**: Mediterranean diet recommended
            
            ### 🏋️‍♂️ Preventive Measures
            - **Exercise routine**: 150 minutes moderate or 75 minutes vigorous activity weekly
            - **Heart-healthy diet**: Fruits, vegetables, whole grains, lean proteins
            - **Stress reduction**: 7-9 hours quality sleep nightly
            - **Tobacco avoidance**: Complete smoking cessation
            """)
    
    # General disclaimer
    st.markdown("---")
    st.caption("""
    **Medical Disclaimer**: This assessment is for informational purposes only and does not constitute medical advice. 
    Always consult qualified healthcare professionals for personal medical decisions and treatment plans.
    """)

def display_footer():
    """Display footer with copyright information"""
    
    st.markdown("---")
    st.markdown(
        '<div class="footer">'
        '© 2024 Heart Disease Predictor | '
        'All Rights Reserved by GK | '
        'Developed with ❤️ for Better Heart Health | '
        'For Educational and Informational Purposes Only'
        '</div>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()