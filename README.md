# 🚗 Used Car Price Predictor App

This Streamlit app leverages a machine learning model (Linear Regression) to estimate the price of a used car based on various features. The project includes end-to-end data preprocessing, exploratory data analysis (EDA), feature engineering, and model evaluation.

## 📊 Project Overview
**Goal**: Predict the price of used cars using numerical and categorical features from a dataset.

The application helps users estimate a fair market price for used vehicles by analysing:

* Mileage
* Engine Volume
* Brand
* Body Type
* Engine Type
* Year
* Registration Status

## 🧠 Machine Learning Workflow
**1. Data Cleaning**:
* Removed high-cardinality columns (e.g, `Model`)
* Handled missing values
* Remove outliers from `Price`, `Mileage`, `EngineV`, and `Year`
* Visualized distributions and relationships

**2. Relaxing OLS Assumptions**:
* Identified skewness and applied log transformation to `Price`
* Removed multicollinearity (via Variance Inflation Factor)

**3. Feature Engineering**:
* One-hot encoded categorical variables (Dummy variables)

**4. Modeling**:
* Scaled features using `StandardScaler`
* Trained a `LinearRegression` model
* Evaluated performance using R² score and residuals

**5. Testing**:
* Compared predicted vs actual values on the test set
* Calculated performance metrics and visualized results

## 🛠️ Tech Stack

**Python**  
**Pandas**  
**Numpy**  
**Scikit-learn**  
**Statsmodels**  
**Seaborn**  
**Matplotlib**  
**Streamlit**  
**Plotly**


## ⚙️ How to Run Locally

**1.** Clone the repository:
```bash
git clone https://github.com/Maurici-oh/used_car_price_predictor_app.git
cd used-car-price-predictor_app/
```
**2.** Install dependencies:

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the requirements file.

```bash
pip install -r requirements.txt
```
**3.** Run the Streamlit app:

```bash
cd used-car-price-predictor_app/streamlit_app/
streamlit run app.py
```
**4.** Open the app in your browser at: `http://localhost:8501`

## 🧾 Folder Structure
<pre>
used_car_price_predictor_app/  
├── streamlit_app/ 
│   └── .streamlit/
│   │   └── config.toml
│   ├── app.py
│   ├── regression.png
│   ├── used_car_price_predictor.py
│   └── Used_cars_data.csv 
│
├── python_regression_code/
│   ├── used_car_price_predictor.py
│   └── Used_cars_data.csv
│
├── requirements.txt
├── LICENSE
└── README.md
</pre>

**Note**: The `.streamlit/` folder contains the `config.toml` file, which disables the app’s responsiveness to the browser’s theme and forces it to use the dark theme.

## 📈 Example Output

* R² Score on Train Set: ~0.77 (may vary depending on data)
* Visual plots of predictions, residuals, and feature importance

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or pull requests to improve the app.

## 📄 License

This project is licensed under the [MIT](https://choosealicense.com/licenses/mit/) License.



