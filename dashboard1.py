import marimo

__generated_with = "0.19.2"
app = marimo.App(width="medium")


@app.cell
def _():
    # Import libaries
    import marimo as mo
    import plotly.express as px
    import pandas as pd
    import numpy as np
    return mo, np, pd, px


@app.cell
def _(pd):
    # Load data - fix later to relative path
    red = pd.read_csv('data/winequality-red.csv', sep=';')
    white = pd.read_csv('data/winequality-white.csv', sep=';')
    return red, white


@app.cell
def _(pd, red, white):
    # merge them into one
    red['color'] = 'red'
    white['color'] = 'white'

    df = pd.concat([red, white], axis=0)
    print(df.shape)
    return (df,)


@app.cell
def _(mo):
    mo.md(text = "# Wine Quality Prediction via SVM")
    return


@app.cell
def _(mo):
    mo.md(text = "We want to distinct between 'excellent' and 'not excellent' wines via SVM classification based on their chemical properties. First, let's dig into the data")
    return


@app.cell
def _(mo):
    mo.md(text = "## Data Overview")
    return


@app.cell
def _(df):
    df.sample(5)
    return


@app.cell
def _(mo):
    mo.md(text = "We see that the dataset contains various chemical properties of wines along with their quality ratings and color (red or white). Let's have a look at the distribution of different features. Select any feature you want to visualize. The 'quality' feature is of particular interest as we will derive the target variable we want to predict from it.")
    return


@app.cell
def _(df, mo):
    variable_distribution = mo.ui.dropdown(
        options=df.columns.tolist(),
        label="Select feature to visualize its distribution",
        value="alcohol"  # default value
    )

    color_toggle = mo.ui.checkbox(
        label="Distinguish by wine color"
    )

    mo.hstack([variable_distribution,color_toggle])
    return color_toggle, variable_distribution


@app.cell
def _(color_toggle, df, mo, px, variable_distribution):
    use_color = color_toggle.value
    selected_var = variable_distribution.value

    hist = px.histogram(
        df, 
        x=selected_var,
        color='color' if use_color else None,
        barmode='overlay' if use_color else 'relative',
        nbins=30, 
        title=f"Distribution of {selected_var}"
    )

    box = px.box(
        df, 
        y=selected_var,
        x='color' if use_color else None,
        color='color' if use_color else None,
        title=f"Boxplot of {selected_var}"
    )

    mo.ui.tabs({
        "Histogram": hist,
        "Boxplot": box
    })
    return


@app.cell
def _(df):
    df["quality"].value_counts().sort_index()
    return


@app.cell
def _(mo):
    mo.md(text = "## Creating target variable 'excellence'")
    return


@app.cell
def _(df, mo, np):
    # Cell 1: Create the target variable and explain
    df['excellence'] = np.where(df['quality'] >= 7, 1, 0)

    mo.md("We define wines with quality ratings of 7 or higher as 'excellent' (1), and those with ratings below 7 as 'not excellent' (0). Let's see the ratio of our target variable.")
    return


@app.cell
def _(df):
    df['excellence'].value_counts()/(len(df))
    return


@app.cell
def _(mo):
    mo.md("""
    A **ratio** of **1:5** is okay-ish for classification. We don't have to worry about sever imbalance for now. Next, let's split the data into training and testing sets.
    """)
    return


@app.cell
def _(df):
    df['color_encoded'] = df['color'].map({'red': 0, 'white': 1})

    # Splitting data into X and y
    X = df.drop(columns=['color', 'excellence', 'quality'])
    y = df['excellence']
    return X, y


@app.cell
def _(mo):
    mo.md("""
    ## Try out different parameters for SVM and see how they affect performance.
    """)
    return


@app.cell
def _(mo):
    test_size_slider = mo.ui.slider(
        start=0.05,
        stop=0.95,
        step=0.05,
        value=0.2,
        label="Test Set Size"
    )

    kernel_select = mo.ui.dropdown(
        options=['linear', 'rbf', 'poly'],
        label="Select Kernel",
        value='rbf'
    )

    c_slider = mo.ui.slider(
        start=0.1,   # changed from 0.01
        stop=100,
        step=0.1,
        value=1.0,
        label="Penalty Parameter C"
    )

    mo.hstack([test_size_slider, kernel_select, c_slider])
    return c_slider, kernel_select, test_size_slider


@app.cell
def _(X, mo, test_size_slider, y):
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size_slider.value,  # use slider value
        random_state=24, 
        stratify=y
    )

    mo.md(f"""
    **Train/Test Split Results:**
    - Test size: {test_size_slider.value:.0%}
    - Training samples: {len(X_train)}
    - Test samples: {len(X_test)}
    """)
    return X_test, X_train, y_test, y_train


@app.cell
def _(X_test, X_train, c_slider, kernel_select, y_test, y_train):
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, classification_report

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train
    svc = SVC(kernel=kernel_select.value, C=c_slider.value)
    svc.fit(X_train_scaled, y_train)

    # Predict
    test_pred = svc.predict(X_test_scaled)

    # Evaluate
    train_acc = svc.score(X_train_scaled, y_train)
    test_acc = svc.score(X_test_scaled, y_test)
    return test_acc, test_pred, train_acc


@app.cell
def _(mo, pd, test_acc, test_pred, train_acc, y_test):
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Metrics for positive class (excellence = 1)
    precision = precision_score(y_test, test_pred)
    recall = recall_score(y_test, test_pred)
    f1 = f1_score(y_test, test_pred)

    # Create confusion matrix
    cm = confusion_matrix(y_test, test_pred)

    # Create heatmap figure
    fig, ax = plt.subplots(figsize=(5, 4))
    cm_heatmap = pd.DataFrame(
        data=cm, 
        columns=['Predicted OK', 'Predicted Excellent'], 
        index=['Actual OK', 'Actual Excellent']
    )
    sns.heatmap(cm_heatmap, annot=True, fmt='d', cmap='YlGnBu', ax=ax)
    ax.set_title('Confusion Matrix')
    plt.tight_layout()

    # Metrics text
    metrics = mo.md(f"""
    **Model Performance:**

    | Metric | Score |
    |--------|-------|
    | Training Accuracy | {train_acc:.4f} |
    | Test Accuracy | {test_acc:.4f} |
    | Precision | {precision:.4f} |
    | Recall | {recall:.4f} |
    | F1-Score | {f1:.4f} |
    """)

    # Display side by side
    mo.hstack([metrics, fig])
    return


@app.cell
def _(mo):
    mo.vstack([
        mo.md("**Recall:** Of all actual positives, how many did we find? = TP / (TP + FN)"),
        mo.md("**Precision:** Of all predicted positives, how many were correct? = TP / (TP + FP)"),
        mo.md("**F1-Score:** Harmonic mean of precision and recall = 2 * (Precision * Recall) / (Precision + Recall)")
    ])
    return


if __name__ == "__main__":
    app.run()
