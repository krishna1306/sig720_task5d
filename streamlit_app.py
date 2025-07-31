# build a streamlit app to predict the price of a property in Paris based on the top 5 features from the linear regression model
import streamlit as st
import joblib
import numpy as np

# Load the model and scaler
model_top5 = joblib.load('linear_model_top5.pkl')
scaler_top5 = joblib.load('scaler_top5.pkl')

# Function to predict the price of a property
def predict_price_top5(square_meters, floors, has_yard, has_pool, city_part_range):
    # Prepare the input data
    input_data = np.array([[square_meters, floors, has_yard, has_pool, city_part_range]])
    
    # Scale only square_meters and floors (first two features)
    input_data_copy = input_data.copy()
    input_data_copy[:, 0:2] = scaler_top5.transform(input_data[:, 0:2].reshape(1, -1))
    input_data_scaled = input_data_copy
    
    # Predict the price
    predicted_price = model_top5.predict(input_data_scaled)
    
    return predicted_price[0]

# Streamlit app layout
st.title("Paris Property Price Prediction")
st.write("Enter the details of the property to predict its price:")

# Input fields for property details
square_meters = st.number_input("Square Meters", min_value=0)
floors = st.number_input("Floors", min_value=0)
has_yard = st.checkbox("Has Yard")
has_pool = st.checkbox("Has Pool")
city_part_range = st.slider("City Part Range", 1, 10)

# Center and style the predict button
st.markdown("<br>", unsafe_allow_html=True)  # Add some spacing
col1, col2, col3 = st.columns([1, 1, 1])

with col2:
    # Custom styled predict button
    predict_clicked = st.button(
        "🔮 Predict Price", 
        type="primary",
        use_container_width=True,
        help="Click to get the estimated property price"
    )

if predict_clicked:
    # Validation: Check if square meters or floors is 0
    if square_meters == 0 and floors == 0:
        st.markdown("---")
        st.error("❌ **Error:** Both Square meters and Floors cannot be 0. Please enter valid values.")
        st.info("💡 **Tip:** Enter the actual size of the property in square meters and number of floors to get an accurate price prediction.")
    elif square_meters == 0:
        st.markdown("---")
        st.error("❌ **Error:** Square meters cannot be 0. Please enter a valid property size.")
        st.info("💡 **Tip:** Enter the actual size of the property in square meters to get an accurate price prediction.")
    elif floors == 0:
        st.markdown("---")
        st.error("❌ **Error:** Floors cannot be 0. Please enter a valid number of floors.")
        st.info("💡 **Tip:** Enter the actual number of floors in the property (minimum 1 floor).")
    else:
        predicted_price = predict_price_top5(square_meters, floors, has_yard, has_pool, city_part_range)
        
        # Display the predicted price in a large, centered box
        st.markdown("---")
        
        # Center the heading
        st.markdown(
            "<h3 style='text-align: center; color: #2E8B57; margin: 20px 0;'>🏠 Predicted Property Price</h3>",
            unsafe_allow_html=True
        )
        
        # Create centered columns for better layout
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            # Large styled container for the price
            st.markdown(
                f"""
                <div style="
                    background-color: #f0f2f6;
                    border: 3px solid #4CAF50;
                    border-radius: 15px;
                    padding: 30px;
                    text-align: center;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                    margin: 20px 0;
                ">
                    <h1 style="
                        color: #2E8B57;
                        font-size: 40px;
                        margin: 0;
                        font-weight: bold;
                    ">€{predicted_price:,.2f}</h1>
                    <p style="
                        color: #666;
                        font-size: 18px;
                        margin: 10px 0 0 0;
                    ">Estimated Property Value</p>
                </div>
                """,
                unsafe_allow_html=True
            )