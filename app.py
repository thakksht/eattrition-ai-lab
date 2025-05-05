
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
import pickle
import base64
from io import BytesIO

# Set page configuration
st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="🧑‍💼",
    layout="wide"
)

# Title and description
st.title("Employee Attrition Prediction Tool")
st.markdown("""
This app predicts the likelihood of an employee leaving the company based on various features.
Upload your employee dataset or use the interactive form to make predictions for a single employee.
""")

# Function to load and preprocess data
@st.cache_data
def load_and_process_data(file):
    ds = pd.read_csv(file)
    
    # Convert binary features
    ds['Attrition'] = ds['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
    ds['Gender'] = ds['Gender'].apply(lambda x: 1 if x == 'Male' else 0)
    ds['Over18'] = ds['Over18'].apply(lambda x: 1 if x == 'Y' else 0)
    ds['OverTime'] = ds['OverTime'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    # One-hot encoding for categorical features
    ds = ds.join(pd.get_dummies(ds['BusinessTravel'], prefix='BusinessTravel')).drop('BusinessTravel', axis=1)
    ds = ds.join(pd.get_dummies(ds['Department'], prefix='Department')).drop('Department', axis=1)
    ds = ds.join(pd.get_dummies(ds['EducationField'], prefix='Education')).drop('EducationField', axis=1)
    ds = ds.join(pd.get_dummies(ds['JobRole'], prefix='JobRole')).drop('JobRole', axis=1)
    ds = ds.join(pd.get_dummies(ds['MaritalStatus'], prefix='MaritalStatus')).drop('MaritalStatus', axis=1)
    
    # Convert boolean to numeric
    ds = ds.map(lambda x: 1 if x is True else 0 if x is False else x)
    
    # Drop unnecessary columns
    ds = ds.drop(['EmployeeNumber', 'EmployeeCount', 'Over18', 'StandardHours'], axis=1, errors='ignore')
    
    return ds

# Function to train models
@st.cache_resource
def train_models(ds):
    x, y = ds.drop(['Attrition'], axis=1), ds['Attrition']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    # Scale the data
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    
    # Apply SMOTE to balance the dataset
    smote = SMOTE(random_state=42)
    x_train_balanced, y_train_balanced = smote.fit_resample(x_train_scaled, y_train)
    
    # Train RandomForest
    random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
    random_forest_model.fit(x_train_balanced, y_train_balanced)
    
    # Train XGBoost
    xgb_model = XGBClassifier(random_state=42)
    xgb_model.fit(x_train_balanced, y_train_balanced)
    
    # Train Logistic Regression
    logistic_model = LogisticRegression(random_state=42, class_weight='balanced')
    logistic_model.fit(x_train_balanced, y_train_balanced)
    
    # Get feature importance
    feature_importance = dict(zip(x.columns, random_forest_model.feature_importances_))
    
    return {
        'x_columns': x.columns,
        'scaler': scaler,
        'random_forest': random_forest_model,
        'xgboost': xgb_model,
        'logistic_regression': logistic_model,
        'feature_importance': feature_importance
    }

# Function to make prediction for a single employee
def predict_attrition(employee_data, models, model_choice='random_forest'):
    # Convert to dataframe with proper columns
    employee_df = pd.DataFrame([employee_data])[models['x_columns']]
    
    # Scale the data
    employee_scaled = models['scaler'].transform(employee_df)
    
    # Make prediction
    if model_choice == 'random_forest':
        prediction = models['random_forest'].predict(employee_scaled)[0]
        probability = models['random_forest'].predict_proba(employee_scaled)[0][1]
    elif model_choice == 'xgboost':
        prediction = models['xgboost'].predict(employee_scaled)[0]
        probability = models['xgboost'].predict_proba(employee_scaled)[0][1]
    else:  # logistic regression
        prediction = models['logistic_regression'].predict(employee_scaled)[0]
        probability = models['logistic_regression'].predict_proba(employee_scaled)[0][1]
    
    return prediction, probability

# Function to create a downloadable plot
def get_plot_download_link(fig, filename="plot.png", text="Download Plot"):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Function to generate the feature importance plot
def plot_feature_importance(models, top_n=15):
    sorted_importance = dict(sorted(models['feature_importance'].items(), key=lambda x: x[1], reverse=True))
    top_features = list(sorted_importance.keys())[:top_n]
    top_importances = list(sorted_importance.values())[:top_n]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(top_features[::-1], top_importances[::-1], color='skyblue')
    ax.set_xlabel('Feature Importance')
    ax.set_title(f'Top {top_n} Important Features - Random Forest')
    plt.tight_layout()
    
    return fig

# Main app
tab1, tab2, tab3 = st.tabs(["Single Employee Prediction", "Batch Prediction", "Model Insights"])

with tab1:
    st.header("Predict for a Single Employee")
    
    # Check if a model is already trained or if we need to upload data
    if 'models' not in st.session_state:
        st.info("Please either upload a CSV file with employee data to train the model, or use the default model.")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            uploaded_file = st.file_uploader("Upload your employee dataset (CSV)", type=["csv"])
            
            if uploaded_file is not None:
                with st.spinner("Processing data and training models..."):
                    # Load and process data
                    ds = load_and_process_data(uploaded_file)
                    # Train models
                    st.session_state.models = train_models(ds)
                    st.success("Models trained successfully!")
        
        with col2:
            if st.button("Use default model"):
                # This is just a placeholder. In a real app, you would load a pre-trained model
                st.warning("In a real app, this would load a pre-trained model. For this demo, please upload a CSV file.")
    
    # If models are trained, show the prediction form
    if 'models' in st.session_state:
        st.subheader("Enter Employee Details")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.slider("Age", 18, 65, 35)
            gender = st.selectbox("Gender", ["Male", "Female"])
            marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
            education = st.slider("Education Level (1-5)", 1, 5, 3)
            distance_from_home = st.slider("Distance From Home (miles)", 1, 30, 10)
            
        with col2:
            job_level = st.slider("Job Level (1-5)", 1, 5, 2)
            job_role = st.selectbox("Job Role", [
                "Sales Executive", "Research Scientist", "Laboratory Technician", 
                "Manufacturing Director", "Healthcare Representative", "Manager", 
                "Sales Representative", "Research Director", "Human Resources"
            ])
            department = st.selectbox("Department", [
                "Sales", "Research & Development", "Human Resources"
            ])
            business_travel = st.selectbox("Business Travel", [
                "Travel_Rarely", "Travel_Frequently", "Non-Travel"
            ])
            
        with col3:
            total_working_years = st.slider("Total Working Years", 0, 40, 10)
            years_at_company = st.slider("Years at Company", 0, 40, 5)
            years_in_current_role = st.slider("Years in Current Role", 0, 20, 3)
            years_since_last_promotion = st.slider("Years Since Last Promotion", 0, 15, 2)
            years_with_curr_manager = st.slider("Years with Current Manager", 0, 20, 3)
            
        # Additional features in expander
        with st.expander("Additional Features"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                environment_satisfaction = st.slider("Environment Satisfaction (1-4)", 1, 4, 3)
                job_involvement = st.slider("Job Involvement (1-4)", 1, 4, 3)
                job_satisfaction = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
                relationship_satisfaction = st.slider("Relationship Satisfaction (1-4)", 1, 4, 3)
                work_life_balance = st.slider("Work Life Balance (1-4)", 1, 4, 3)
                
            with col2:
                monthly_income = st.slider("Monthly Income ($)", 1000, 20000, 6000)
                percent_salary_hike = st.slider("Percent Salary Hike", 0, 25, 15)
                stock_option_level = st.slider("Stock Option Level (0-3)", 0, 3, 1)
                training_times_last_year = st.slider("Training Times Last Year", 0, 6, 2)
                num_companies_worked = st.slider("Number of Companies Worked", 0, 10, 2)
                
            with col3:
                daily_rate = st.slider("Daily Rate", 100, 1500, 800)
                hourly_rate = st.slider("Hourly Rate", 30, 100, 65)
                monthly_rate = st.slider("Monthly Rate", 5000, 30000, 15000)
                overtime = st.checkbox("Overtime", False)
                performance_rating = st.slider("Performance Rating (1-4)", 1, 4, 3)
                
        # Education field
        education_field = st.selectbox("Education Field", [
            "Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"
        ])
        
        # Prepare model choice
        model_choice = st.selectbox(
            "Select Prediction Model", 
            ["Random Forest", "XGBoost", "Logistic Regression"],
            index=0
        )
        
        if st.button("Predict Attrition"):
            # Prepare input data
            employee_data = {
                'Age': age,
                'DistanceFromHome': distance_from_home,
                'Education': education,
                'EnvironmentSatisfaction': environment_satisfaction,
                'JobInvolvement': job_involvement,
                'JobLevel': job_level,
                'JobSatisfaction': job_satisfaction,
                'MonthlyIncome': monthly_income,
                'NumCompaniesWorked': num_companies_worked,
                'PercentSalaryHike': percent_salary_hike,
                'PerformanceRating': performance_rating,
                'RelationshipSatisfaction': relationship_satisfaction,
                'StockOptionLevel': stock_option_level,
                'TotalWorkingYears': total_working_years,
                'TrainingTimesLastYear': training_times_last_year,
                'WorkLifeBalance': work_life_balance,
                'YearsAtCompany': years_at_company,
                'YearsInCurrentRole': years_in_current_role,
                'YearsSinceLastPromotion': years_since_last_promotion,
                'YearsWithCurrManager': years_with_curr_manager,
                'DailyRate': daily_rate,
                'Gender': 1 if gender == "Male" else 0,
                'HourlyRate': hourly_rate,
                'MonthlyRate': monthly_rate,
                'OverTime': 1 if overtime else 0,
            }
            
            # Add one-hot encoded features
            
            # Business Travel
            employee_data.update({
                'BusinessTravel_Non-Travel': 1 if business_travel == "Non-Travel" else 0,
                'BusinessTravel_Travel_Frequently': 1 if business_travel == "Travel_Frequently" else 0,
                'BusinessTravel_Travel_Rarely': 1 if business_travel == "Travel_Rarely" else 0
            })
            
            # Department
            employee_data.update({
                'Department_Human Resources': 1 if department == "Human Resources" else 0,
                'Department_Research & Development': 1 if department == "Research & Development" else 0,
                'Department_Sales': 1 if department == "Sales" else 0
            })
            
            # Education Field
            employee_data.update({
                'Education_Human Resources': 1 if education_field == "Human Resources" else 0,
                'Education_Life Sciences': 1 if education_field == "Life Sciences" else 0,
                'Education_Marketing': 1 if education_field == "Marketing" else 0,
                'Education_Medical': 1 if education_field == "Medical" else 0,
                'Education_Other': 1 if education_field == "Other" else 0,
                'Education_Technical Degree': 1 if education_field == "Technical Degree" else 0
            })
            
            # Job Role
            job_roles = [
                "Healthcare Representative", "Human Resources", "Laboratory Technician", 
                "Manager", "Manufacturing Director", "Research Director", 
                "Research Scientist", "Sales Executive", "Sales Representative"
            ]
            
            for role in job_roles:
                employee_data[f'JobRole_{role}'] = 1 if job_role == role else 0
            
            # Marital Status
            employee_data.update({
                'MaritalStatus_Divorced': 1 if marital_status == "Divorced" else 0,
                'MaritalStatus_Married': 1 if marital_status == "Married" else 0,
                'MaritalStatus_Single': 1 if marital_status == "Single" else 0
            })
            
            # Make prediction
            model_name = model_choice.lower().replace(" ", "_")
            prediction, probability = predict_attrition(employee_data, st.session_state.models, model_name)
            
            # Display result
            st.subheader("Prediction Result")
            
            if prediction == 1:
                st.error(f"⚠️ **Attrition Risk**: This employee is likely to leave the company.")
                st.warning(f"Probability of leaving: {probability:.2%}")
            else:
                st.success(f"✅ **Retention Likelihood**: This employee is likely to stay with the company.")
                st.info(f"Probability of leaving: {probability:.2%}")
            
            # Show top factors
            st.subheader("Top Factors Influencing This Prediction")
            
            # This is a simplified approach - in a more sophisticated app, 
            # you would use SHAP values or other techniques to get feature importance for this specific prediction
            top_features = dict(sorted(st.session_state.models['feature_importance'].items(), key=lambda x: x[1], reverse=True)[:5])
            
            for feature, importance in top_features.items():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{feature}:** {employee_data.get(feature, 'N/A')}")
                with col2:
                    st.write(f"Importance: {importance:.4f}")

with tab2:
    st.header("Batch Prediction")
    
    st.info("Upload a CSV file with employee data to make predictions for multiple employees.")
    
    # File uploader for batch prediction
    batch_file = st.file_uploader("Upload employee dataset for batch prediction", type=["csv"])
    
    if batch_file is not None:
        # Read and process the uploaded file
        st.info("Processing your file...")
        
        # If models are not trained yet, train them
        if 'models' not in st.session_state:
            with st.spinner("Training models first..."):
                # Load and process data
                ds = load_and_process_data(batch_file)
                # Train models
                st.session_state.models = train_models(ds)
                st.success("Models trained successfully!")
                
                # Reset file position to beginning for further processing
                batch_file.seek(0)
        
        # Process the batch file
        try:
            batch_data = pd.read_csv(batch_file)
            st.write("Preview of uploaded data:")
            st.dataframe(batch_data.head())
            
            # Check if the data needs preprocessing
            preprocess = st.checkbox("My data needs preprocessing (convert categorical values, etc.)", value=True)
            
            if preprocess:
                with st.spinner("Preprocessing data..."):
                    processed_data = load_and_process_data(batch_file)
                    
                    # Check if 'Attrition' column exists and remove it for prediction
                    if 'Attrition' in processed_data.columns:
                        X_batch = processed_data.drop('Attrition', axis=1)
                        has_attrition = True
                    else:
                        X_batch = processed_data
                        has_attrition = False
            else:
                X_batch = batch_data
                has_attrition = 'Attrition' in X_batch.columns
                if has_attrition:
                    X_batch = X_batch.drop('Attrition', axis=1)
            
            # Model selection
            batch_model = st.selectbox(
                "Select model for batch prediction",
                ["Random Forest", "XGBoost", "Logistic Regression"]
            )
            
            if st.button("Run Batch Prediction"):
                # Make predictions
                model_name = batch_model.lower().replace(" ", "_")
                
                # Ensure the columns match
                missing_cols = set(st.session_state.models['x_columns']) - set(X_batch.columns)
                if missing_cols:
                    st.error(f"Missing columns in your data: {', '.join(missing_cols)}")
                    st.stop()
                
                # Keep only the columns that are in the model
                X_batch = X_batch[st.session_state.models['x_columns']]
                
                # Scale the data
                X_batch_scaled = st.session_state.models['scaler'].transform(X_batch)
                
                # Make predictions
                if model_name == 'random_forest':
                    predictions = st.session_state.models['random_forest'].predict(X_batch_scaled)
                    probabilities = st.session_state.models['random_forest'].predict_proba(X_batch_scaled)[:, 1]
                elif model_name == 'xgboost':
                    predictions = st.session_state.models['xgboost'].predict(X_batch_scaled)
                    probabilities = st.session_state.models['xgboost'].predict_proba(X_batch_scaled)[:, 1]
                else:  # logistic regression
                    predictions = st.session_state.models['logistic_regression'].predict(X_batch_scaled)
                    probabilities = st.session_state.models['logistic_regression'].predict_proba(X_batch_scaled)[:, 1]
                
                # Create result dataframe
                result_df = batch_data.copy()
                result_df['Predicted_Attrition'] = predictions
                result_df['Attrition_Probability'] = probabilities
                result_df['Predicted_Attrition_Label'] = result_df['Predicted_Attrition'].apply(lambda x: 'Yes' if x == 1 else 'No')
                
                # Show results
                st.subheader("Prediction Results")
                st.dataframe(result_df)
                
                # Calculate summary statistics
                attrition_count = result_df['Predicted_Attrition'].sum()
                total_count = len(result_df)
                attrition_percentage = (attrition_count / total_count) * 100
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Predicted Attrition Count", f"{attrition_count} / {total_count}")
                with col2:
                    st.metric("Attrition Percentage", f"{attrition_percentage:.2f}%")
                
                # Download the results
                csv = result_df.to_csv(index=False)
                st.download_button(
                    label="Download Prediction Results",
                    data=csv,
                    file_name="attrition_predictions.csv",
                    mime="text/csv",
                )
                
                # If original data had attrition column, show model performance
                if has_attrition and preprocess:
                    st.subheader("Model Performance")
                    y_true = processed_data['Attrition']
                    
                    # Calculate metrics
                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                    
                    accuracy = accuracy_score(y_true, predictions)
                    precision = precision_score(y_true, predictions)
                    recall = recall_score(y_true, predictions)
                    f1 = f1_score(y_true, predictions)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Accuracy", f"{accuracy:.2f}")
                    with col2:
                        st.metric("Precision", f"{precision:.2f}")
                    with col3:
                        st.metric("Recall", f"{recall:.2f}")
                    with col4:
                        st.metric("F1 Score", f"{f1:.2f}")
                    
                    # Confusion matrix
                    conf_matrix = confusion_matrix(y_true, predictions)
                    
                    # Plot confusion matrix
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    ax.set_title('Confusion Matrix')
                    ax.set_xticklabels(['Stay', 'Leave'])
                    ax.set_yticklabels(['Stay', 'Leave'])
                    st.pyplot(fig)
        
        except Exception as e:
            st.error(f"Error processing file: {e}")

with tab3:
    st.header("Model Insights")
    
    # Check if models are trained
    if 'models' in st.session_state:
        # Feature importance
        st.subheader("Feature Importance")
        top_n = st.slider("Number of top features to show", 5, 30, 15)
        
        # Plot feature importance
        fig = plot_feature_importance(st.session_state.models, top_n)
        st.pyplot(fig)
        
        # Download link for the plot
        st.markdown(get_plot_download_link(fig, "feature_importance.png", "Download Feature Importance Plot"), unsafe_allow_html=True)
        
        # Feature description
        st.subheader("Feature Descriptions")
        
        feature_descriptions = {
            'OverTime': "Whether the employee works overtime",
            'MonthlyIncome': "Monthly salary",
            'Age': "Age of the employee",
            'JobLevel': "Level of the employee's job (1-5)",
            'TotalWorkingYears': "Total years of work experience",
            'YearsAtCompany': "Years at the current company",
            'StockOptionLevel': "Level of stock options (0-3)",
            'YearsInCurrentRole': "Years in current role",
            'YearsWithCurrManager': "Years with current manager",
            'YearsSinceLastPromotion': "Years since last promotion",
            'JobInvolvement': "Level of job involvement (1-4)",
            'JobSatisfaction': "Level of job satisfaction (1-4)",
            'WorkLifeBalance': "Work-life balance rating (1-4)",
            'EnvironmentSatisfaction': "Level of environment satisfaction (1-4)",
            'RelationshipSatisfaction': "Level of relationship satisfaction (1-4)",
            'DistanceFromHome': "Distance from home to work (miles)",
            'NumCompaniesWorked': "Number of companies worked at before",
            'PercentSalaryHike': "Percentage increase in salary last year",
            'TrainingTimesLastYear': "Number of training sessions last year",
            'PerformanceRating': "Performance rating (1-4)",
            'Education': "Level of education (1-5)",
        }
        
        # Show top 10 most important features with descriptions
        top_features_sorted = dict(sorted(st.session_state.models['feature_importance'].items(), key=lambda x: x[1], reverse=True))
        
        for i, (feature, importance) in enumerate(list(top_features_sorted.items())[:10]):
            with st.expander(f"{i+1}. {feature} (Importance: {importance:.4f})"):
                st.write(feature_descriptions.get(feature, "No description available"))
                
                # Show distribution of this feature if it's numeric
                if feature in ['Age', 'MonthlyIncome', 'DistanceFromHome', 'YearsAtCompany', 'TotalWorkingYears']:
                    # We would need the original dataset here
                    st.info(f"In a real app, this would show the distribution of {feature}.")
                    
        # Model explanations
        st.subheader("How to Interpret These Results")
        
        st.markdown("""
        ### Understanding the Models
        
        #### Random Forest
        - Builds multiple decision trees and merges them for a more accurate prediction
        - Less prone to overfitting than single decision trees
        - Good at handling complex relationships in the data
        
        #### XGBoost
        - Advanced implementation of gradient boosting
        - Often provides better performance than Random Forest
        - Can handle imbalanced data well
        
        #### Logistic Regression
        - Simpler model that estimates probabilities
        - More interpretable than tree-based models
        - Useful for understanding the direction of effects
        
        ### How to Use These Insights
        
        1. **Focus on the top factors** - Addressing issues with the most important factors can have the biggest impact on retention
        2. **Look for patterns** - Are there specific departments or job roles with higher attrition?
        3. **Personalize interventions** - Use the individual predictions to target employees who might be at risk
        4. **Consider contextual factors** - Some factors may be correlated with others
        
        ### Recommended Actions
        
        - **Monitor overtime** - Excessive overtime is often a strong predictor of attrition
        - **Evaluate compensation** - Competitive pay is crucial for retention
        - **Provide growth opportunities** - Years in current role and since last promotion are key factors
        - **Improve work-life balance** - This is consistently important across many studies
        - **Address management issues** - Years with current manager can be a significant factor
        """)
    else:
        st.info("Please train the models first by uploading a dataset in the Single Employee Prediction or Batch Prediction tabs.")
        
# Footer
st.markdown("---")
st.markdown("Employee Attrition Prediction Tool | © 2025")