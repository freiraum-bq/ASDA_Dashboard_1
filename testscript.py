import marimo

__generated_with = "0.19.2"
app = marimo.App(width="columns")


@app.cell(column=0)
def _():
    import pandas as pd
    import marimo as mo
    return mo, pd


@app.cell
def _(mo, pd):
    ## 1. Fetching the Excel File
    @mo.cache()
    def fetch_excel_file():
        base_url = "https://docs.google.com/spreadsheets/d/"
        url_id = "1Wz5eeD8xRxKgIJWtteZEO0nHPcGfVsUc/"
        export = "export/format=excel"
        whole_url = base_url + url_id + export
        file = pd.ExcelFile(whole_url)
        return file
    excel_file = fetch_excel_file()
    return (excel_file,)


@app.cell
def _(excel_file):
    excel_file.info()
    return


@app.cell
def _(excel_file):
    ## 2. Parsing the different dataframes
    emp_df = excel_file.parse("empirical_coded(n=1755)")
    con_df = excel_file.parse("conceptual_coded(n=1131)")
    rev_df = excel_file.parse("review_coded(n=115)")
    return con_df, emp_df, rev_df


@app.cell
def _(emp_df):
    emp_df
    return


@app.cell
def _(con_df, emp_df, mo, rev_df):
    ## 3. Creating option tabs to visualizen the different dataframes
    tabs = mo.ui.tabs(
        {"Empirical articles": emp_df, "Conceptual articles": con_df, "Review articles": rev_df}, value="Heading 2"
    )
    return (tabs,)


@app.cell
def _(tabs):
    tabs
    return


@app.cell
def _(mo):
    mo.ui.multiselect
    return


@app.cell
def _(mo):
    multiselect = mo.ui.multiselect(
        options=[10, 20, 30], label="choose some options"
    )
    return (multiselect,)


@app.cell
def _(multiselect):
    multiselect
    return


@app.cell
def _(multiselect):
    adding = multiselect.value[0] + multiselect.value[1]
    return (adding,)


@app.cell
def _(adding):
    adding
    return


@app.cell
def _(emp_df, mo):
    menu = mo.ui.dropdown(emp_df.columns)
    menu
    return


@app.cell
def _(emp_df):
    emp_df
    return


@app.cell
def _():
    return


@app.cell(column=1)
def _():
    return


if __name__ == "__main__":
    app.run()
