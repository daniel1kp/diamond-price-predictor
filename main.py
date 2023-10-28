import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import altair as alt

# Set the page title and icon
st.set_page_config(page_title="Diamond Price Predictor", page_icon="💎")

# App title and description
st.title("Diamond Price Prediction!")
st.write("Upload a file in either CSV or JSON format to analyze the data and predict diamond prices!")

# File upload widget
uploaded_file = st.file_uploader("Upload a file", type=["csv", "json"])

selected_case = None  # Initialize selected_case
num_rows = 1000  # Default value for num_rows

if uploaded_file is not None:
    # Determine the file format based on the file extension
    file_extension = uploaded_file.name.split('.')[-1].lower()

    if file_extension == "csv":
        # Read the CSV file into a Pandas DataFrame
        df = pd.read_csv(uploaded_file,header=0)
        # Allow the user to select a case
        selected_case = st.selectbox("Case Study Options!", ["Select the case!", "Case 1 - 100 Rows", "Case 2 - 1000 Rows", "Case 3 - 2500 Rows"])
    elif file_extension == "json":
        df = pd.read_json(uploaded_file)
        st.error("JSON format is not supported currently, please upload a CSV file!")
        st.stop()

    # Data Preprocessing
    # One-hot encode categorical columns
    df_encoded = pd.get_dummies(df, columns=['cut', 'color', 'clarity'])

    if selected_case != "Select the case!":
        if selected_case == "Case 1 - 100 Rows":
            st.subheader("Data Preview of Rows 1-100")
            st.write(df_encoded.iloc[0:100])
        elif selected_case == "Case 2 - 1000 Rows":
            st.subheader("Data Preview of Rows 1-1000")
            st.write(df_encoded.iloc[0:1000])
        elif selected_case == "Case 3 - 2500 Rows":
            st.subheader("Data Preview of Rows 1-2500")
            st.write(df_encoded.iloc[0:2500])

        # Data Visualization (only shown when a case is selected)
        st.subheader("Data Visualization")

        # Allow users to select features for visualization
        selected_features = st.multiselect("Select features for visualization", df_encoded.columns)
        visualization_type = st.selectbox("Select visualization type", ["Scatter Plot", "Line Chart", "Bar Chart"])

        if selected_features:
            if visualization_type == "Scatter Plot":
                chart = alt.Chart(df_encoded.head(1000)).mark_circle().encode(
                    x=selected_features[0],
                    y=selected_features[1],
                    tooltip=selected_features
                ).interactive()
            elif visualization_type == "Line Chart":
                chart = alt.Chart(df_encoded.head(1000)).mark_line().encode(
                    x=selected_features[0],
                    y=selected_features[1],
                    tooltip=selected_features
                ).interactive()
            elif visualization_type == "Bar Chart":
                chart = alt.Chart(df_encoded.head(1000)).mark_bar().encode(
                    x=selected_features[0],
                    y=selected_features[1],
                    tooltip=selected_features
                ).interactive()
            st.altair_chart(chart, use_container_width=True)

    # Model Training (only shown when a case is selected)
    if selected_case != "Select the case!":
        X = df_encoded.drop(columns=['price'])
        y = df_encoded['price']

        num_rows = None  # Initialize num_rows

        if selected_case == "Case 1 - 100 Rows":
            num_rows = 100
        elif selected_case == "Case 2 - 1000 Rows":
            num_rows = 1000
        elif selected_case == "Case 3 - 2500 Rows":
            num_rows = 2500

        if num_rows is not None:
            X_train, X_test, y_train, y_test = train_test_split(X.iloc[:num_rows], y.iloc[:num_rows], test_size=0.2, random_state=42)

            # Feature Scaling
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)

            model = RandomForestRegressor()
            model.fit(X_train_scaled, y_train)

            # User-Friendly Interface in a sidebar
            st.sidebar.subheader("Predict Diamond Price")
            carat = st.sidebar.number_input("Carat", min_value=0.2, max_value=5.0, value=1.0)
            depth = st.sidebar.number_input("Depth", min_value=0.0, max_value=100.0, value=61.0, step=0.1)
            table = st.sidebar.number_input("Table", min_value=0.0, max_value=100.0, value=57.0, step=0.1)
            x = st.sidebar.number_input("x", min_value=0.0, max_value=10.0, value=5.0, step=0.01)
            y = st.sidebar.number_input("y", min_value=0.0, max_value=10.0, value=5.0, step=0.01)
            z = st.sidebar.number_input("z", min_value=0.0, max_value=10.0, value=5.0, step=0.01)

            if st.sidebar.button("Predict The Price"):
                input_data = pd.DataFrame({
                    'carat': [carat],
                    'depth': [depth],
                    'table': [table],
                    'x': [x],
                    'y': [y],
                    'z': [z]
                })

                # Reorder the columns to match the model's expected feature order
                input_data = input_data.reindex(columns=X.columns, fill_value=0)

                predicted_price = model.predict(input_data)[0]
                st.sidebar.write(f"Predicted Price is approximately: ${predicted_price:.2f}")
