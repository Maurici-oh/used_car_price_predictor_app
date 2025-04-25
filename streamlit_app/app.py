import streamlit as st
import numpy as np
import pandas as pd
import random
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def main():
    @st.cache_data
    def load_data():
        return pd.read_csv("streamlit_app/Used_cars_data.csv")

    raw_data = load_data()

    # This function was creating to avoid writing the same code multiple times.
    def data_preprocessing_complete():
        data_no_mv = raw_data.drop(["Model"], axis=1).dropna(axis=0)
        q_price = data_no_mv["Price"].quantile(0.99)
        data_1 = data_no_mv[data_no_mv["Price"] < q_price]
        q_mileage = data_1["Mileage"].quantile(0.99)
        data_2 = data_1[data_1["Mileage"] < q_mileage]
        data_3 = data_2[data_2["EngineV"] < 6.5]
        q_year = data_3["Year"].quantile(0.01)
        data_cleaned = data_3[data_3["Year"] > q_year].reset_index(drop=True)
        data_cleaned["log_price"] = np.log(data_cleaned["Price"])
        data_no_multicollinearity = data_cleaned.drop(["Year", "Price"], axis=1)
        data_with_dummies = pd.get_dummies(data_no_multicollinearity, drop_first=True)

        cols = [
            "log_price",
            "Mileage",
            "EngineV",
            "Brand_BMW",
            "Brand_Mercedes-Benz",
            "Brand_Mitsubishi",
            "Brand_Renault",
            "Brand_Toyota",
            "Brand_Volkswagen",
            "Body_hatch",
            "Body_other",
            "Body_sedan",
            "Body_vagon",
            "Body_van",
            "Engine Type_Gas",
            "Engine Type_Other",
            "Engine Type_Petrol",
            "Registration_yes",
        ]

        data_preprocessed = data_with_dummies[cols]

        return data_preprocessed

    st.sidebar.subheader("Used Car Price Predictor App")
    st.sidebar.markdown("---")

    tabs = st.sidebar.radio(
        "Go to",
        [
            "Introduction",
            "Data Cleaning",
            "OLS Assumptions",
            "Feature Engineering",
            "Modeling",
            "Testing",
            "Python Code",
        ],
    )

    # -------------------------------------------
    #    ::: SIDEBAR[0] - Introduction :::
    # -------------------------------------------
    if tabs == "Introduction":

        # Custom CSS for fullscreen centered layout and floating circles
        st.markdown(
            """
            <style>
            html, body, .main {
                height: 100vh; /* Use viewport height */
                margin: 0;
                padding: 0;
                overflow: auto;
            }

            .circle {
                position: absolute;
                border-radius: 50%;
                opacity: 0.15;
                animation: float 10s ease-in-out infinite;
                z-index: 0;
            }

            @keyframes float {
                0% { transform: translateY(0px); }
                50% { transform: translateY(-20px); }
                100% { transform: translateY(0px); }
            }
            
            .header {
                font-size: 3em;
                font-weight: bold;
                color: #ffffff;
                margin-bottom: 10px;
            }

            .subheader {
                font-size: 1.5em;
                font-weight: 500;
                color: #ffffff;
                margin: 20px 0 10px;
            }

            .info {
                font-size: 1.1em;
                max-width: 800px;
                line-height: 1.6;
                color: #ffffff;
                margin: 0 auto;
            }

            .centered-wrapper {
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                z-index: 1;
                text-align: center;
                padding: 0 10px;
                margin-top: -240px;
                width: 80%;
            }
            </style>
        """,
            unsafe_allow_html=True,
        )

        colors = ["#e74c3c", "#3498db"]
        for _ in range(30):
            size = random.randint(30, 90)
            top = random.randint(0, 10)
            left = random.randint(0, 70)
            color = random.choice(colors)
            st.markdown(
                f"""
                <div class="circle" style="
                    width: {size}px;
                    height: {size}px;
                    background-color: {color};
                    top: {top}vh;
                    left: {left}vw;
                    animation-delay: {random.uniform(0, 5):.2f}s;
                "></div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            """
                    <div class="centered-wrapper">
                        <div class='header'> Welcome to the Used Car Price Predictor App! </div>
                        <hr style="border: 1px solid #ccc; width: 100%;">
                        <div class='info'>
                            This Streamlit app leverages a machine learning model (Linear Regression) to estimate <br>
                            the price of a used car based on various features.  <br><br>
                            In this interactive walkthrough, we’ll explore the steps involved in preparing <br>
                            a real-world dataset for analysis. <br><br>
                            <strong>The project includes:</strong> <br>
                            • end-to-end data preprocessing<br>
                            • exploratory data analysis (EDA)<br>
                            • feature engineering<br>
                            • model evaluation<br>
                        </div>
                        <div class='subheader'>🚀 Let's begin our journey!</div>
                    </div>
                    """,
            unsafe_allow_html=True,
        )

    # -------------------------------------------
    #    ::: SIDEBAR[1] - Data Cleaning :::
    # -------------------------------------------
    elif tabs == "Data Cleaning":
        tab1, tab2, tab3 = st.tabs(
            ["Initial Data Visualization", "Missing Values", "Dealing with Outliers"]
        )

        # -------------------------------------------
        # ::: TAB 1 - Data visualization :::
        # -------------------------------------------
        with tab1:
            st.subheader("Raw Data")
            st.caption("We begin with a preview of the dataframe’s first few rows.")
            st.dataframe(raw_data.head())
            st.markdown("---")

            st.subheader("Data Description")
            st.caption(
                "We use this table to identify missing values and possible outliers in the data."
            )
            st.dataframe(raw_data.describe(include="all"))
            with st.expander("Click for a more detailed analysis..."):
                st.markdown("- The `count` reveals missing values in some columns.")
                st.markdown(
                    "- The high number of unique values in `Model` suggests it might be problematic for direct use in modeling."
                )
                st.markdown(
                    "- The dominant frequency of 'yes' in `Registration` indicates it might not be a very informative feature."
                )

            st.markdown("---")
            st.subheader("Dropping the 'Model' Column")
            st.caption(
                "Due to the high number of unique categories (high cardinality), creating hundreds of dummy variables from the 'Model' column is impractical. Therefore, we'll drop this column."
            )
            data_dropped_model = raw_data.drop(["Model"], axis=1)
            st.dataframe(data_dropped_model.head())

        # -------------------------------------------
        # ::: TAB 2- Missing Values:::
        # -------------------------------------------
        with tab2:
            col1, col2, col3 = st.columns([4, 0.5, 2])

            data = raw_data.drop(["Model"], axis=1)
            missing_values = data.isnull().sum()
            col1.subheader("Missing Value Counts")

            col3.dataframe(
                pd.DataFrame(missing_values, columns=["Missing Values Count"])
            )
            col1.markdown(
                "We observe missing values in both `Price` and `EngineV` columns."
            )
            col1.markdown(
                "As a rule of thumb, if less than 5% of the data is missing, we can often remove the rows with missing values without significant loss of information. Let's check the percentage of missing values in `EngineV`."
            )
            percent_missing_enginev = (missing_values["EngineV"] / len(data)) * 100
            col1.markdown(
                f"Percentage of missing values in `EngineV`: {percent_missing_enginev:.2f}%"
            )

            st.markdown("---")
            st.subheader("Removing Rows with Missing Values")
            st.markdown(
                "Since the percentage of missing values in `EngineV` is less than 5%, we will remove the rows containing them."
            )
            data_no_mv = data.dropna(axis=0)
            count_row = pd.DataFrame(data_no_mv.describe(include="all").loc["count"])
            st.dataframe(count_row.T)
            st.markdown(
                f"Number of rows after removing missing values: :blue-background[{len(data_no_mv)}]"
            )

        # -------------------------------------------
        # ::: TAB 3 - Outliers :::
        # -------------------------------------------
        with tab3:
            # --- Outliers in Price ---
            with st.container():
                st.subheader('Outliers in "Price" column')
                st.markdown(
                    "Let's visualize the distribution of the `Price` column to identify potential outliers using a histogram."
                )
                data_no_mv = raw_data.drop(["Model"], axis=1).dropna(axis=0)
                col1, col2 = st.columns(2)

                with col1:
                    fig_price_hist = px.histogram(
                        data_no_mv,
                        x="Price",
                        color_discrete_sequence=["#f03a11"],
                        title="Distribution of Price",
                    )
                    st.plotly_chart(fig_price_hist)

                with col2:
                    q_price = data_no_mv["Price"].quantile(0.99)
                    data_1 = data_no_mv[data_no_mv["Price"] < q_price]

                    fig_price_hist_cleaned = px.histogram(
                        data_1,
                        x="Price",
                        color_discrete_sequence=["#15e62a"],
                        title="Distribution of Price (Outliers Removed)",
                    )
                    st.plotly_chart(fig_price_hist_cleaned)

                st.markdown(
                    "The distribution is skewed, suggesting the presence of :blue[high-value outliers]. We'll remove the top 1% of 'Price' values."
                )
            st.markdown("---")

            # --- Outliers in Mileage ---
            with st.container():
                st.subheader('Outliers in "Mileage" column')
                st.markdown(
                    "Now, let's examine the `Mileage` column for outliers. We'll remove the top 1% of `Mileage` values."
                )
                col1, col2 = st.columns(2)

                with col1:
                    fig_mileage_hist = px.histogram(
                        data_1,
                        x="Mileage",
                        color_discrete_sequence=["#f03a11"],
                        title="Distribution of Mileage",
                    )
                    st.plotly_chart(fig_mileage_hist)

                with col2:
                    q_mileage = data_1["Mileage"].quantile(0.99)
                    data_2 = data_1[data_1["Mileage"] < q_mileage]
                    fig_mileage_hist_cleaned = px.histogram(
                        data_2,
                        x="Mileage",
                        color_discrete_sequence=["#15e62a"],
                        title="Distribution of Mileage (Outliers Removed)",
                    )
                    st.plotly_chart(fig_mileage_hist_cleaned)
            st.markdown("---")

            # --- Outliers in EngineV ---
            with st.container():
                st.subheader('Outliers in "EngineV" column')
                st.markdown(
                    "Let's check `EngineV` for unusual values. Sometimes, a value like `99.99` is used to indicate missing data, so we'll remove values above a reasonable threshold."
                )
                col1, col2 = st.columns(2)

                with col1:
                    fig_enginev_hist = px.histogram(
                        data_2,
                        x="EngineV",
                        color_discrete_sequence=["#f03a11"],
                        title="Distribution of EngineV",
                    )
                    fig_enginev_hist.update_layout(yaxis=dict(range=[0, 700]))

                    st.plotly_chart(fig_enginev_hist)
                with col2:
                    data_3 = data_2[
                        data_2["EngineV"] < 6.5
                    ]  # Using the same threshold as in the original code
                    fig_enginev_hist_cleaned = px.histogram(
                        data_3,
                        x="EngineV",
                        color_discrete_sequence=["#15e62a"],
                        title="Distribution of EngineV (Outliers Removed)",
                    )
                    st.plotly_chart(fig_enginev_hist_cleaned)
            st.markdown("---")

            # --- Outliers in Year ---
            with st.container():
                st.subheader('Outliers in "Year" column')
                st.markdown(
                    "Finally, let's look at the `Year` column. We'll remove the bottom 1% of values, assuming very old cars might be outliers for our analysis."
                )
                col1, col2 = st.columns(2)

                with col1:
                    fig_year_hist = px.histogram(
                        data_3,
                        x="Year",
                        color_discrete_sequence=["#f03a11"],
                        title="Distribution of Year",
                    )
                    st.plotly_chart(fig_year_hist)
                with col2:
                    q_year = data_3["Year"].quantile(0.01)
                    data_4 = data_3[data_3["Year"] > q_year]
                    fig_year_hist_cleaned = px.histogram(
                        data_4,
                        x="Year",
                        color_discrete_sequence=["#15e62a"],
                        title="Distribution of Year (Outliers Removed)",
                    )
                    st.plotly_chart(fig_year_hist_cleaned)
            st.markdown("---")

            with st.container():
                data_cleaned = data_4.reset_index(drop=True)
                st.subheader("Cleaned Data")
                st.caption(
                    "Finally, our dataset is free of missing values and outliers."
                )
                st.dataframe(data_cleaned.describe(include="all"))
                with st.expander("Click for a more detailed analysis..."):
                    st.markdown(
                        "- The `count` row shows the same amount of values in all columns."
                    )
                    st.markdown(
                        "- The `max` values in the processed columns is now closer to the `mean` value."
                    )

    # -------------------------------------------
    #    ::: SIDEBAR[2] - OLS Assumptions :::
    # -------------------------------------------
    elif tabs == "OLS Assumptions":
        tab1, tab2 = st.tabs(["Ordinary Least Squares", "Relaxing Assumptions"])

        # -------------------------------------------
        # ::: TAB 1 - Ordinary Least Squares :::
        # -------------------------------------------
        with tab1:

            col1, col2 = st.columns([3, 2])

            with col1:
                st.header("What is OLS?")
                st.markdown(
                    """
                            Ordinary Least Squares (OLS) regression is a technique used in linear regression 
                            to minimize the sum of squared differences between observed and predicted values, 
                            and obtain a straight line as close as possible to your data points.
                            """
                )
                st.markdown(
                    """
                            For OLS to give **unbiased**, **consistent**, and **efficient estimates**, several key :blue[assumptions] must hold. 
                            
                            Here are the ones that we will be analyzing:
                            * Linearity
                            * No Multicollinearity
                            * No Endogeneity
                            * Normality of Errors
                            * Zero Conditional Mean
                            * Homoscedasticity
                            * No Autocorrelation
                            """
                )

            with col2:
                # This is a random plot to display OLS graphically
                np.random.seed(42)
                X = np.linspace(0, 10, 50)
                Y = (
                    2 * X + 1 + np.random.normal(0, 2, size=X.shape)
                )  # y = 2x + 1 + noise

                # Fit linear regression using polyfit
                coeffs = np.polyfit(X, Y, deg=1)
                Y_pred = np.polyval(coeffs, X)

                # Create DataFrame
                df = pd.DataFrame(
                    {"X": X, "Y": Y, "Y_pred": Y_pred, "Residual": Y - Y_pred}
                )

                # Create Plotly figure
                fig = go.Figure()

                # Scatter plot: actual data
                fig.add_trace(
                    go.Scatter(
                        x=df["X"],
                        y=df["Y"],
                        mode="markers",
                        name="Actual Data",
                        marker=dict(color="violet"),
                    )
                )

                # Regression line
                fig.add_trace(
                    go.Scatter(
                        x=df["X"],
                        y=df["Y_pred"],
                        mode="lines",
                        name="OLS Regression Line",
                        line=dict(color="cyan", width=2),
                    )
                )

                # Residual lines
                for i in range(len(df)):
                    fig.add_trace(
                        go.Scatter(
                            x=[df["X"][i], df["X"][i]],
                            y=[df["Y"][i], df["Y_pred"][i]],
                            mode="lines",
                            line=dict(color="blue", dash="dot"),
                            showlegend=False,
                        )
                    )

                fig.update_layout(
                    title="OLS Regression: Fit Line and Residuals",
                    xaxis_title="X (Independent Variable)",
                    yaxis_title="Y (Dependent Variable)",
                    legend=dict(x=0.01, y=0.99),
                    height=450,
                    width=700,
                )

                st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")
            st.subheader("Potential Regressors")
            st.markdown(
                """
                        To proceed with the assumption analysis, we have identified several variables as 
                        potential regressors. These include the continuous variables in our dataset:
                        
                        - **Price**, **Year**, **EngineV**, and **Mileage**
                        
                        The categorical variables will be included as dummy variables, so there is no need 
                        to analyze them when checking the assumptions.
                        """
            )

        # -------------------------------------------
        # ::: TAB 2 - Relaxing Assumptions :::
        # -------------------------------------------
        with tab2:

            def preprocessing():
                data_no_mv = raw_data.drop(["Model"], axis=1).dropna(axis=0)
                q_price = data_no_mv["Price"].quantile(0.99)
                data_1 = data_no_mv[data_no_mv["Price"] < q_price]
                q_mileage = data_1["Mileage"].quantile(0.99)
                data_2 = data_1[data_1["Mileage"] < q_mileage]
                data_3 = data_2[data_2["EngineV"] < 6.5]
                q_year = data_3["Year"].quantile(0.01)
                data_cleaned = data_3[data_3["Year"] > q_year].reset_index(drop=True)
                return data_cleaned

            def assumpion_linearity():
                data_cleaned = preprocessing()

                st.markdown(
                    """
                            - :blue[**What it means**:] The relationship between the independent variables and the dependent variable is linear in parameters.

                            - :blue[**Why it matters**:] Ensures that the model is correctly specified and interpretable.
                            """
                )
                with st.container():
                    st.subheader("Checking for Linearity")
                    st.markdown(
                        "To check for linearity we'll visualize the relationship between the continuous variables (`Year`, `EngineV`, `Mileage`) and `Price` using scatter plots."
                    )

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        fig_scatter_year = px.scatter(
                            data_cleaned,
                            x="Year",
                            y="Price",
                            color="Price",
                            color_continuous_scale="Turbo",
                            title="Price vs Year",
                        )
                        fig_scatter_year.update_layout(coloraxis_showscale=False)
                        st.plotly_chart(fig_scatter_year)
                    with col2:
                        fig_scatter_enginev = px.scatter(
                            data_cleaned,
                            x="EngineV",
                            y="Price",
                            color="Price",
                            color_continuous_scale="Turbo",
                            title="Price vs EngineV",
                        )
                        fig_scatter_enginev.update_layout(coloraxis_showscale=False)
                        st.plotly_chart(fig_scatter_enginev)
                    with col3:
                        fig_scatter_mileage = px.scatter(
                            data_cleaned,
                            x="Mileage",
                            y="Price",
                            color="Price",
                            color_continuous_scale="Turbo",
                            title="Price vs Mileage",
                        )
                        fig_scatter_mileage.update_layout(coloraxis_showscale=False)
                        st.plotly_chart(fig_scatter_mileage)

                    st.markdown(
                        "The relationships don't appear strictly linear, but quite exponential."
                    )

                with st.container():
                    col1, col2 = st.columns([2, 5], gap="medium")

                    with col1:
                        data_no_mv = raw_data.drop(["Model"], axis=1).dropna(axis=0)
                        q_price = data_no_mv["Price"].quantile(0.99)
                        data_1 = data_no_mv[data_no_mv["Price"] < q_price]

                        fig_price_hist_cleaned = px.histogram(
                            data_1,
                            x="Price",
                            color_discrete_sequence=["#15e62a"],
                            title="Price distribution",
                        )
                        fig_price_hist_cleaned.update_layout(width=400, height=300)

                        st.plotly_chart(fig_price_hist_cleaned)

                    with col2:
                        st.markdown("###")
                        st.markdown(
                            """
                                    Recalling the Price distribution plot we can notice that it is not normally distributed, 
                                    and from there it's relationships with the other rather normally distributed features are not linear.
                                    """
                        )
                        st.markdown("###")
                        st.markdown(
                            "To address this, we'll transform the `Price` variable using a :blue-background[logarithm]."
                        )

                with st.container():
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        data_cleaned["log_price"] = np.log(data_cleaned["Price"])
                        fig_scatter_log_year = px.scatter(
                            data_cleaned,
                            x="Year",
                            y="log_price",
                            color="log_price",
                            color_continuous_scale="Viridis",
                            title="Log Price vs Year",
                        )
                        fig_scatter_log_year.update_layout(coloraxis_showscale=False)
                        st.plotly_chart(fig_scatter_log_year)
                    with col2:
                        fig_scatter_log_enginev = px.scatter(
                            data_cleaned,
                            x="EngineV",
                            y="log_price",
                            color="log_price",
                            color_continuous_scale="Viridis",
                            title="Log Price vs EngineV",
                        )
                        fig_scatter_log_enginev.update_layout(coloraxis_showscale=False)
                        st.plotly_chart(fig_scatter_log_enginev)
                    with col3:
                        fig_scatter_log_mileage = px.scatter(
                            data_cleaned,
                            x="Mileage",
                            y="log_price",
                            color="log_price",
                            color_continuous_scale="Viridis",
                            title="Log Price vs Mileage",
                        )
                        fig_scatter_log_mileage.update_layout(coloraxis_showscale=False)
                        st.plotly_chart(fig_scatter_log_mileage)
                    st.markdown(
                        """The relationships now appear more linear after the log transformation of `Price`. 
                                We'll proceed with the regression using this transformed variable.
                                """
                    )
                    data_cleaned = data_cleaned.drop(["Price"], axis=1)

            def assumpion_no_multicollinearity():
                st.markdown(
                    """
                            - :blue[**What it means**:] Multicollinearity happends when two or more independent variables in a regression model are highly correlated, 
                            indicating a strong linear relationship among the predictor variables.

                            - :blue[**Why it matters**:] Perfect correlation makes it impossible to estimate the unique effect of each variable.
                            """
                )

                st.markdown(
                    """
                            Logically, `Year` and `Mileage` are expected to be correlated (the newer the car, the lower the mileage). 
                            To check for multicollinearity, we will use the Variance Inflation Factor (VIF).
                            """
                )
                st.markdown("##")

                data_cleaned = preprocessing()
                with st.container():
                    variables = data_cleaned[["Mileage", "Year", "EngineV"]]
                    vif = pd.DataFrame()
                    vif["VIF"] = [
                        variance_inflation_factor(variables.values, i)
                        for i in range(variables.shape[1])
                    ]
                    vif["features"] = variables.columns

                    st.subheader("Variance Inflation Factor (VIF)")
                    col1, col2, col3 = st.columns([5, 0.5, 2])
                    with col1:
                        st.markdown(
                            """
                                    The Variance Inflation Factor (VIF) estimates how much larger the standard error of a coefficient 
                                    becomes compared to a scenario in which the predictor is not correlated with any other predictors.
                                    
                                    """
                        )
                        st.markdown(
                            """
                                    A VIF greater than `5` or `10` is generally considered indicative of high 
                                    multicollinearity.   
                                    Here, `Year` has a relatively high VIF, suggesting it's correlated with 
                                    other independent variables.
                                    """
                        )

                    with col3:
                        st.dataframe(vif)

                st.markdown("##")
                st.markdown(
                    "We'll remove the `Year` column from the dataframe to solve this issue."
                )
                data_no_multicollinearity = data_cleaned.drop(["Year"], axis=1)
                st.dataframe(data_no_multicollinearity.head())

            def assumption_no_endogeneity():
                st.markdown(
                    """
                            - :blue[**What it means**:] The error term should not be correlated with any of the independent variables. 
                            
                            - :blue[**Why it matters**:] This ensures that the variation in the independent variables explains the 
                            dependent variable, and not vice versa. If this assumption is violated, then your estimates are biased 
                            and inconsistent, even with large samples.
                    
                            """
                )

            def assumpion_normality_of_errors():
                st.markdown(
                    """
                            - :blue[**What it means**:] The error terms are normally distributed.

                            - :blue[**Why it matters**:] Required for valid confidence intervals and hypothesis testing (especially in small samples).
                            """
                )
                st.markdown("##")
                st.markdown(
                    """
                            Normality is assumed in a big sample following the **Central Limit Theorem**.
                            
                            This theorem states that when you take a large number of random samples from any population (regardless of its original 
                            distribution), the distribution of the sample means will approximate a normal distribution (bell curve), as long as the 
                            sample size is sufficiently large.
                            """
                )

            def assumpion_zero_conditional_mean():
                st.markdown(
                    """
                            - :blue[**What it means**:] The expected value of the error term, given the independent variables, is zero.

                            - :blue[**Why it matters**:] Guarantees that the independent variables are not correlated with the error term, ensuring unbiased estimates.
                            """
                )
                st.markdown(
                    "This was accomplished through the inclusion of the intercept in the regression."
                )

            def assumpion_homoscedasticity():
                st.markdown(
                    """
                            - :blue[**What it means**:] The variance of the error term is constant across all levels of the independent variables.

                            - :blue[**Why it matters**:] Without it, standard errors could be biased, affecting hypothesis tests (like t-tests).

                            """
                )
                st.markdown("##")
                st.markdown(
                    """
                            We can conclude that this assumption holds by looking at the continuous variable's (`Mileage`, `EngineV`, `Year`) 
                            plots after the log transformation of `Price`.
                            
                            The reason being that the log transformation is the most common fix for homoscedasticity.
                            """
                )
                data_cleaned = preprocessing()
                with st.container():
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        data_cleaned["log_price"] = np.log(data_cleaned["Price"])
                        fig_scatter_log_year = px.scatter(
                            data_cleaned,
                            x="Year",
                            y="log_price",
                            color="log_price",
                            color_continuous_scale="Viridis",
                            title="Log Price vs Year",
                        )
                        fig_scatter_log_year.update_layout(coloraxis_showscale=False)
                        st.plotly_chart(fig_scatter_log_year)
                    with col2:
                        fig_scatter_log_enginev = px.scatter(
                            data_cleaned,
                            x="EngineV",
                            y="log_price",
                            color="log_price",
                            color_continuous_scale="Viridis",
                            title="Log Price vs EngineV",
                        )
                        fig_scatter_log_enginev.update_layout(coloraxis_showscale=False)
                        st.plotly_chart(fig_scatter_log_enginev)
                    with col3:
                        fig_scatter_log_mileage = px.scatter(
                            data_cleaned,
                            x="Mileage",
                            y="log_price",
                            color="log_price",
                            color_continuous_scale="Viridis",
                            title="Log Price vs Mileage",
                        )
                        fig_scatter_log_mileage.update_layout(coloraxis_showscale=False)
                        st.plotly_chart(fig_scatter_log_mileage)

            def assumpion_no_autocorrelation():
                st.markdown(
                    """
                            - :blue[**What it means**:] The error terms are not correlated across observations (especially in time series data).

                            - :blue[**Why it matters**:] Correlated errors can lead to inefficient estimates and incorrect standard errors.
                            """
                )

                st.markdown("##")
                st.markdown(
                    """
                            The observations we have do not come from time series or panel data. Since each row represents a 
                            different customer in a used car sales dataset, we can assume that the observations are independent of each other.
                            """
                )

            assumption_pages = {
                "Linearity": assumpion_linearity,
                "No Multicollinearity": assumpion_no_multicollinearity,
                "No Endogeneity": assumption_no_endogeneity,
                "Normality of Errors": assumpion_normality_of_errors,
                "Zero Conditional Mean": assumpion_zero_conditional_mean,
                "Homoscedasticity": assumpion_homoscedasticity,
                "No Autocorrelation": assumpion_no_autocorrelation,
            }

            selected_page = st.selectbox(
                "Select an assumption from the list below:", assumption_pages.keys()
            )

            assumption_pages[selected_page]()

    # -------------------------------------------
    #    ::: SIDEBAR[3] - Feature Engineering :::
    # -------------------------------------------
    elif tabs == "Feature Engineering":
        data_no_mv = raw_data.drop(["Model"], axis=1).dropna(axis=0)
        q_price = data_no_mv["Price"].quantile(0.99)
        data_1 = data_no_mv[data_no_mv["Price"] < q_price]
        q_mileage = data_1["Mileage"].quantile(0.99)
        data_2 = data_1[data_1["Mileage"] < q_mileage]
        data_3 = data_2[data_2["EngineV"] < 6.5]
        q_year = data_3["Year"].quantile(0.01)
        data_cleaned = data_3[data_3["Year"] > q_year].reset_index(drop=True)
        data_cleaned["log_price"] = np.log(data_cleaned["Price"])
        data_no_multicollinearity = data_cleaned.drop(["Year", "Price"], axis=1)

        st.subheader("Dummy Variables")
        st.markdown(
            "To create dummy variables we first need to get a list of the dataframe's categorical features."
        )
        categorical_cols = data_no_multicollinearity.select_dtypes(
            include="object"
        ).columns.tolist()
        st.write(categorical_cols)
        st.markdown("---")

        st.subheader("Creating Dummy Variables")
        st.markdown(
            """
                    For categorical features like `Brand`, `Body`, `Engine Type`, and `Registration`, we 
                    need to create dummy variables (one-hot encoding) so they can be used in linear regression 
                    models. We'll use `pd.get_dummies()` with `drop_first=True` to avoid multicollinearity among 
                    the dummy variables.
                    """
        )
        code = """
            data_with_dummies = pd.get_dummies(data_cleaned, drop_first=True)
            """
        st.code(code, language="python")
        st.markdown("##")
        data_with_dummies = pd.get_dummies(data_no_multicollinearity, drop_first=True)
        # st.markdown(f"Number of columns after creating dummies: :blue-background[{len(data_with_dummies.columns)}]")

        cols = [
            "log_price",
            "Mileage",
            "EngineV",
            "Brand_BMW",
            "Brand_Mercedes-Benz",
            "Brand_Mitsubishi",
            "Brand_Renault",
            "Brand_Toyota",
            "Brand_Volkswagen",
            "Body_hatch",
            "Body_other",
            "Body_sedan",
            "Body_vagon",
            "Body_van",
            "Engine Type_Gas",
            "Engine Type_Other",
            "Engine Type_Petrol",
            "Registration_yes",
        ]

        data_preprocessed = data_with_dummies[cols]
        st.markdown(
            "After rearranging the dataframe, these are the top columns of the final preprocessed dataset, ready for building a linear regression model."
        )
        st.dataframe(data_preprocessed.head(20))

    # -------------------------------------------
    #    ::: SIDEBAR[4] - Modeling :::
    # -------------------------------------------
    elif tabs == "Modeling":

        data_preprocessed = data_preprocessing_complete()

        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(
            [
                "Features and Targets",
                "Scaling",
                "Data Split",
                "Model Training",
                "Make Predictions",
                "Training Visualization",
                "Residuals Distribution",
                "R² Score",
                "Coefficients",
            ]
        )
        
        # -------------------------------------------
        # ::: TAB 1 - Features & Targets :::
        # -------------------------------------------
        with tab1:
            with st.container():
                st.subheader("Step 1: Selecting Features and Target")
                st.markdown(
                    "Define the target variable (what we want to predict) and the input features (what we use for prediction)"
                )
                targets = data_preprocessed["log_price"]
                inputs = data_preprocessed.drop(["log_price"], axis=1)
                code = """
                    targets = data_preprocessed['log_price']
                    inputs = data_preprocessed.drop(['log_price'],axis=1)
                    """
                st.code(code, language="python")

        # -------------------------------------------
        # ::: TAB 2 - Scaling :::
        # -------------------------------------------
        with tab2:
            with st.container():
                st.subheader("Step 2: Standardize the input features")
                st.markdown(
                    "`StandardScaler` ensures all features have **mean = 0** and **standard deviation = 1**."
                )
                st.markdown(
                    "This is especially important when features are on different scales."
                )
                scaler = StandardScaler()
                scaler.fit(inputs)
                inputs_scaled = scaler.transform(inputs)
                code = """
                    scaler = StandardScaler()
                    scaler.fit(inputs)
                    inputs_scaled = scaler.transform(inputs)
                    """
                st.code(code, language="python")

        # -------------------------------------------
        # ::: TAB 3 - Data Split:::
        # -------------------------------------------
        with tab3:
            with st.container():
                st.subheader("Step 3: Split the data into training and test sets")
                st.markdown(
                    "80% of the data is used to train the model, and 20% is reserved for testing."
                )
                x_train, x_test, y_train, y_test = train_test_split(
                    inputs_scaled, targets, test_size=0.2, random_state=42
                )
                code = """
                    x_train, x_test, y_train, y_test = train_test_split(inputs_scaled, targets, test_size=0.2, random_state=42)
                    """
                st.code(code, language="python")

        # -------------------------------------------
        # ::: TAB 4 - Model Training :::
        # -------------------------------------------
        with tab4:
            with st.container():
                st.subheader("Step 4: Training the Linear Regression Model")
                reg = LinearRegression()
                reg.fit(x_train, y_train)
                code = """
                    reg = LinearRegression()
                    reg.fit(x_train,y_train)
                    """
                st.code(code, language="python")
                st.success("✅ Model trained successfully!")

        # -------------------------------------------
        # ::: TAB 5 - Make Predictions :::
        # -------------------------------------------
        with tab5:
            with st.container():
                st.subheader("Step 5: Make predictions")
                st.markdown(
                    "Make predictions on the training set to evaluate model fit."
                )
                y_hat = reg.predict(x_train)
                code = """
                    y_hat = reg.predict(x_train)
                    """
                st.code(code, language="python")

        # -------------------------------------------
        # ::: TAB 6 - Training Visualization :::
        # -------------------------------------------
        with tab6:
            with st.container():
                st.subheader("Step 6: Training Set: Actual vs Predicted values")
                targets_vs_predictions = pd.DataFrame(
                    {"targets": y_train, "predictions": y_hat}
                )
                targets_vs_predictions["random_color"] = np.random.rand(
                    len(targets_vs_predictions)
                )

                col1, col2 = st.columns(2, gap="medium")
                with col1:
                    fig_targets_predict = px.scatter(
                        targets_vs_predictions,
                        x="targets",
                        y="predictions",
                        color="random_color",
                        color_continuous_scale="Mint",
                        labels={
                            "targets": "Actual Values (y_train)",
                            "predictions": "Predicted Values (y_hat)",
                            "random_color": "Random Color",
                        },
                        #  title='Targets vs Predictions with Random Colors',
                        width=500,
                        height=500,
                    )

                    fig_targets_predict.add_trace(
                        go.Scatter(
                            x=[6, 13],
                            y=[6, 13],
                            mode="lines",
                            line=dict(color="red", dash="dash"),
                            showlegend=False,
                        )
                    )

                    fig_targets_predict.update_traces(opacity=0.4)

                    fig_targets_predict.update_layout(
                        xaxis=dict(
                            showgrid=True,
                            gridcolor="lightgray",
                            range=[6, 13],
                            scaleanchor="y",
                            scaleratio=1,
                        ),
                        yaxis=dict(showgrid=True, gridcolor="lightgray", range=[6, 13]),
                        height=450,
                        width=450,
                        coloraxis_showscale=False,
                    )
                    st.plotly_chart(fig_targets_predict)

                with col2:
                    st.markdown("#")
                    st.markdown(
                        """
                                A good model will have predictions close to the actual values 
                                (ideally on a 45° line).
                                """
                    )
                    st.markdown("##")
                    st.markdown(
                        """
                                We observe that lower values are closer to the drawn 45° line 
                                compared to higher values. This suggests that our model performs 
                                better when predicting lower values.
                                """
                    )

        # -------------------------------------------
        # ::: TAB 7 - Residuals Distribution :::
        # -------------------------------------------
        with tab7:
            with st.container():
                st.subheader(
                    "Step 7: Plot the Residuals Distribution (prediction errors)"
                )
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.write("###")
                    st.markdown(
                        "We can double check the accuracy of our model by creating a residuals plot. (targets - predictions)"
                    )
                    st.markdown(
                        "Ideally, residuals should be normally distributed around 0, indicating unbiased predictions"
                    )
                    with st.expander("Click for a more detailed explanation..."):
                        st.markdown(
                            """
                                    An important assumption in regression analysis is that of normality and homoscedasticity, 
                                    specifically, the residuals (i.e., the difference between the actual and predicted target 
                                    values) should be normally distributed with a mean of zero. 
                                    
                                    However, the data reveals certain 
                                    observations where the residuals (y_train - ŷ) are significantly negative. This indicates 
                                    that the model overestimates the target variable, leading to predicted prices that are substantially 
                                    higher than the actual observed values.
                                    """
                        )

                with col2:
                    targets_vs_predictions["residuals"] = (
                        targets_vs_predictions["targets"]
                        - targets_vs_predictions["predictions"]
                    )
                    fig_residuals = px.histogram(
                        targets_vs_predictions["residuals"],
                        x="residuals",
                        color_discrete_sequence=["#15e62a"],
                    )

                    fig_residuals.add_vline(
                        x=0,
                        line_dash="dash",  # optional: "solid", "dot", "dash"
                        line_color="#f03a11",
                        line_width=2,
                    )
                    st.plotly_chart(fig_residuals)

        # -------------------------------------------
        # ::: TAB 8 - R² Score :::
        # -------------------------------------------
        with tab8:
            with st.container():
                st.subheader(
                    "Step 8: Model performance with R² (coefficient of determination)"
                )
                st.markdown(
                    "`R²` is a statistical measure that quantifies how well a regression model fits the observed data."
                )
                st.markdown(
                    "It explains how much of the variation in the `target` variable is explained by the model."
                )
                reg.score(x_train, y_train)
                code = """
                    reg.score(x_train, y_train)
                    """
                st.code(code, language="python")
                r2_train = reg.score(x_train, y_train)
                r_2 = round(r2_train, 3)
                r_2 = r_2 * 100
                st.caption("R² Score")
                st.markdown(
                    f"<div style='font-size:50px;'>{r_2} %</div>",
                    unsafe_allow_html=True,
                )
                # st.metric("R² Score", r_2)

                st.write("")
                st.markdown(
                    """
                            R-squared values range from 0 to 1 (or 0% to 100%). A value of 1 indicates a perfect fit, 
                            meaning the model perfectly predicts the dependent variable, while 0 indicates no predictive power
                            """
                )

        # -------------------------------------------
        # ::: TAB 9 - Coefficients :::
        # -------------------------------------------
        with tab9:
            with st.container():
                st.subheader("Step 9: Examine the model coefficients")
                st.markdown(
                    "- :blue[Intercept] (**bias**) is the predicted value when all features are 0."
                )
                st.markdown(
                    "- :blue[Coefficients] (**weights**) show the effect of each feature on the prediction."
                )

                code = """
                    reg.intercept_  # Bias term
                    reg.coef_       # Feature weights
                    """
                st.code(code, language="python")
                st.markdown("##")
                st.markdown(
                    "We created a summary table to display feature names and their corresponding weights."
                )

                col1, col2, col3 = st.columns([6, 2, 4])
                with col1:
                    st.markdown(
                        """
                                Interpretation of weights:
                                - For continuous variables:
                                    > :green[Positive weight] → as the feature's weight increases, log_price increases (direct relationship)  
                                    > :red[Negative weight] → as the feature's weight increases, log_price decreases (inverse relationship)
                                - For categorical (dummy) variables:
                                    > :green[Positive weight] → this category is more expensive than the baseline (e.g., Audi)
                                """
                    )

                with col3:
                    reg_summary = pd.DataFrame(
                        inputs.columns.values, columns=["Feature"]
                    )
                    reg_summary["Weight"] = reg.coef_
                    st.dataframe(reg_summary)

    # -------------------------------------------
    #    ::: SIDEBAR[5] - Testing ::::
    # -------------------------------------------
    elif tabs == "Testing":
        data_preprocessed = data_preprocessing_complete()
        targets = data_preprocessed["log_price"]
        inputs = data_preprocessed.drop(["log_price"], axis=1)
        scaler = StandardScaler()
        scaler.fit(inputs)
        inputs_scaled = scaler.transform(inputs)
        x_train, x_test, y_train, y_test = train_test_split(
            inputs_scaled, targets, test_size=0.2, random_state=42
        )
        reg = LinearRegression()
        reg.fit(x_train, y_train)
        y_hat = reg.predict(x_train)
        y_hat_test = reg.predict(x_test)

        st.header("Model Testing")
        st.markdown("---")
        st.markdown(
            """
                    To start testing we are going to find the predictions and store them in the `y_hat_test` variable.
                    """
        )
        code = """
                y_hat_test = reg.predict(x_test)
                """
        st.code(code, language="python")
        st.markdown(
            """
                    We want predictions to align closely with actual values, especially across all price ranges.
                    
                    To visualize the test results we will plot the test targets against the predicted targets:
                    """
        )

        # --- Testing Targets vs Predictions plot ---
        with st.container():
            col1, col2, col3 = st.columns([1, 6, 1])
            with col2:
                y_hat_test = reg.predict(x_test)

                df_testing = pd.DataFrame(
                    {"targets": y_test, "predictions": y_hat_test}
                )
                df_testing["random_color"] = np.random.rand(len(df_testing))

                fig_testing = px.scatter(
                    df_testing,
                    x="targets",
                    y="predictions",
                    color="random_color",
                    color_continuous_scale="Plasma",
                    labels={
                        "targets": "Targets (y_test)",
                        "predictions": "Predicted Values (y_hat_test)",
                    },
                    title="Targets vs Predictions",
                    width=700,
                    height=500,
                )

                fig_testing.add_trace(
                    go.Scatter(
                        x=[6, 13],
                        y=[6, 13],
                        mode="lines",
                        line=dict(color="red", dash="dash"),
                        showlegend=False,
                    )
                )

                fig_testing.update_traces(opacity=0.6)

                fig_testing.update_layout(
                    xaxis=dict(
                        showgrid=True,
                        gridcolor="lightgray",
                        range=[6, 13],
                        scaleanchor="y",
                        scaleratio=1,
                    ),
                    yaxis=dict(showgrid=True, gridcolor="lightgray", range=[6, 13]),
                    coloraxis_showscale=False,
                )
                st.plotly_chart(fig_testing)

            st.markdown(
                """
                        ### Observations:
                        - For higher prices, we have a higher concentration of values closer to the 45° line, 
                        therefore, the model is very good at predicting higher prices.
                        - Lower values seem more scattered, pointing to the fact that the predicted prices might not be correct.
                        """
            )

            st.markdown("---")

        with st.container():
            # --- Further Analysis ---
            st.subheader("Further Analysis")
            st.markdown(
                "To explore the results in more detail, we created additional dataframes like the one below."
            )
            st.markdown("###")

            df_pf = pd.DataFrame(np.exp(y_hat_test), columns=["Prediction"])
            df_pf["Target"] = np.exp(y_test.reset_index(drop=True))
            df_pf["Residual"] = df_pf["Target"] - df_pf["Prediction"]
            df_pf["Difference%"] = np.abs(df_pf["Residual"] / df_pf["Target"] * 100)

            col1, col2 = st.columns(2)
            with col1:
                st.write("📄 Prediction values vs Actual values (in Price)")
                st.dataframe(
                    df_pf.sort_values(by="Difference%").reset_index(drop=True).head(30)
                )

            with col2:
                # Optional display tweaks for better viewing
                pd.options.display.max_rows = 100
                pd.set_option("display.float_format", lambda x: "%.2f" % x)
                st.markdown("Sort predictions by smallest to largest error percentage")
                st.dataframe(df_pf.sort_values(by=["Difference%"]))

            st.write("📈 Descriptive Statistics of the Dataframe")
            st.dataframe(df_pf.describe())
            st.markdown(
                """
                        ### Observations:
                        - In this last table, we can see the minimum and maximum percentage differences. 
                        The minimum value is exceptional, while the maximum value indicates the opposite. 
                        This suggests that the model can be improved to achieve more consistent results.
                        - The percentiles in the table suggest that for most of our predictions we got relatively close.
                        """
            )

            # --- MODEL IMPROVEMENTS ---
            st.markdown("---")
            st.subheader("Suggestions to Improve the Model")
            st.markdown(
                """
                        - 🔁 **Add/Remove Features**: Include new predictors or remove irrelevant ones.
                        - 🧹 **Outlier Removal**: Handle data points that skew the regression line.
                        - 🔧 **Try Other Models**: Consider Ridge, Lasso, or tree-based models.
                        - 🧪 **Feature Engineering**: Create new features that may better capture relationships in data.
                        """
            )

    # -------------------------------------------
    #    ::: SIDEBAR[6] - Python Code :::
    # -------------------------------------------
    elif tabs == "Python Code":
        st.subheader("Python Script")
        st.caption(
            "The following Python code performs the data preprocessing and linear regression."
        )
        # Read and display the current script
        with open("used_car_price_predictor.py", "r") as f:
            script_content = f.read()

        # Display as formatted Python code
        st.code(script_content, language="python")


if __name__ == "__main__":
    st.set_page_config(
        page_title="Car Price Predictor", page_icon="regression.png", layout="wide"
    )
    main()
