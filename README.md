# SCT_DS_3
A Python-based interactive web application to **analyze wine dataset**, visualize data (heatmap, histograms, pairplots), and predict wine quality using a **Random Forest Classifier**. Built with **Streamlit** for GUI support.

---

## Features

* Load and explore the Wine Quality dataset from UCI.
* Visualize data:

  * Correlation heatmap
  * Histograms
  * Pairplots
* Train and test a **Random Forest Classifier**.
* Check model accuracy.
* Interactive input sliders to predict wine quality.

---

## Requirements

* Python 3.8+
* Packages:

  * pandas
  * scikit-learn
  * streamlit
  * matplotlib
  * seaborn

---

## Setting up Environment

1. **Clone or download the repository**.

2. **Open terminal or PowerShell** and navigate to the project folder:

   ```powershell
   cd path\to\project\folder
   ```

3. **Create a virtual environment** (recommended):

   ```powershell
   python -m venv .venv
   ```

4. **Activate the virtual environment**:

   * On Windows (PowerShell):

     ```powershell
     .\.venv\Scripts\Activate
     ```

   * On macOS/Linux:

     ```bash
     source .venv/bin/activate
     ```

5. **Install required packages**:

   ```powershell
   pip install pandas scikit-learn streamlit matplotlib seaborn
   ```

---

## Running the Application

1. Make sure your virtual environment is activated.

2. Run the Streamlit app:

   ```powershell
   streamlit run app.py
   ```

3. A browser window will open automatically with the GUI, or open the URL shown in the terminal (usually `http://localhost:8501`).

---

## How to Use

* Explore the dataset using checkboxes.
* Visualize graphs by enabling the checkboxes for heatmap, histograms, or pairplots.
* Use sliders under **Predict Wine Quality** to input values for wine features.
* Click **Predict Quality** to see the predicted wine quality.

---

## Notes

* Pairplots are generated using a sample of the dataset for faster performance.
* Make sure you have a stable internet connection to fetch the dataset from UCI.

---


