import streamlit as st
import datetime
import pandas as pd
import requests
import numpy as np
import plotly.graph_objects as go
from dotenv import load_dotenv
import plotly.express as px
from scipy.stats import gaussian_kde
import seaborn as sns
import os

load_dotenv()


# Keys are stored in .env file
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"
NEWS_URL = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={NEWS_API_KEY}"

# page settings
st.set_page_config(page_title="EDA Generator by Ofer Tzvi", layout="wide")
st.markdown("""
    <style>
    [data-testid="stSidebar"] {background-color: #e8f0fa;}
    </style>
    """, unsafe_allow_html=True)

pd.set_option('display.float_format', '{:.2f}'.format)

# session state init
if "name" not in st.session_state: st.session_state.name = ""
if "city" not in st.session_state: st.session_state.city = ""
if "jobject" not in st.session_state: st.session_state.jobject = dict()
if "jnews" not in st.session_state: st.session_state.jnews = dict()
if "uploaded_file" not in st.session_state: st.session_state.uploaded_file = None
if "file_uploader_key" not in st.session_state: st.session_state.file_uploader_key = 0
if "df" not in st.session_state: st.session_state.df = None  # Store loaded dataframe

# returns the jason results from openWeatherMap
def get_weather(city):
    url = f"{WEATHER_URL}?q={city}&appid={WEATHER_API_KEY}&units=metric"
    resp = requests.get(url)
    return resp.json() if resp.status_code == 200 else {}

# returns the jason results from NewsApi
def get_news():
    resp = requests.get(NEWS_URL)
    return resp.json() if resp.status_code == 200 else {}

# read_data csv or excel
def read_data(ftype, as_bool=True, **kwargs):
    if ftype:
        res = pd.read_csv(st.session_state.uploaded_file, **kwargs)
    else:
        res = pd.read_excel(st.session_state.uploaded_file, **kwargs)
    binary_type = 'bool' if as_bool else 'int8'
    for col in res.columns:
        if set(res[col].dropna().unique()) <= {0, 1}:  # Detect binary cols
            res[col] = res[col].astype(binary_type)
    return res


# login page
if st.session_state.name == "":
    st.markdown("<h2 style='text-align: center; color: darkblue;'>Welcome to the Automatic EDA Generator</h2>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([2, 2, 2])
    with col2:
        st.subheader("Please enter your name and city to continue")
        name_input = st.text_input("Enter your name:")
        city_input = st.text_input("Enter your city:")
    if name_input and city_input:
        st.session_state.name = name_input
        st.session_state.city = city_input
        st.session_state.jobject = get_weather(city_input)
        st.session_state.jnews = get_news()
        st.rerun()

#page after login
else:
    # sidebar before file upload
    with st.sidebar:
        st.write(f"Welcome *{st.session_state.name}!*")
        # st.write(f":timer_clock: Current Time: {time.strftime('%H:%M')}")
        w = st.session_state.jobject
        tz_offset = w.get("timezone", 0)
        utc_time = datetime.datetime.now(datetime.UTC)
        city_time = utc_time + datetime.timedelta(seconds=tz_offset)
        st.write(f":timer_clock: Current Time in {st.session_state.city}: **{city_time.strftime('%H:%M')}**")
        if st.session_state.uploaded_file is None:
            st.info("Please upload a CSV or Excel file to start.")
            live_placeholder = st.empty()
            with live_placeholder.container():
                if len(st.session_state.jobject) == 0:
                    st.warning(f"No weather data for {st.session_state.city}.")
                else:
                    w = st.session_state.jobject
                    st.markdown(f"<u>Weather in {st.session_state.city}</u>:", unsafe_allow_html=True)
                    st.markdown(f"<span style='font-size:15px; color:#0c6297;'>Temp: {w['main']['temp']}°C, "
                        f"Humidity: {w['main']['humidity']}%, {w['weather'][0]['description']}</span>",
                        unsafe_allow_html=True)
                    st.markdown("#### :newspaper: Latest News")
                    for article in st.session_state.jnews.get("articles", [])[:3]:
                        st.markdown(f"- [{article['title']}]({article['url']})")
                    bottom_placeholder = st.sidebar.empty()
                    with bottom_placeholder.container():
                        st.markdown(
                            "<p style='position:fixed; bottom:0; font-size:14px;'>\U0001F4E8 For Comments: ofer@il-fts.com</p>",
                            unsafe_allow_html=True)

        else:
            # Sidebar AFTER file upload
            st.success(f"\u2705 File Uploaded: *{st.session_state.uploaded_file.name}*")
            if st.button(":repeat: Replace File"):
                st.session_state.uploaded_file = None
                st.session_state.file_uploader_key += 1
                st.session_state.df = None
                st.rerun()

            st.markdown("###  EDA Options")
            show_overview = st.checkbox("Dataset Overview", True)
            show_corr = st.checkbox("Correlation Analysis", True)
            show_dist = st.checkbox("Distribution Plots", True)
            show_cat = st.checkbox("Categorial Data Plots", True)
            show_pair = st.checkbox("Pair Plots", True)

            dist_columns = []
            add_kde = False

            bottom_placeholder = st.sidebar.empty()
            with bottom_placeholder.container():
                st.markdown(
                    "<p style='position:fixed; bottom:0; font-size:14px;'>\U0001F4E8 For Comments: ofer@il-fts.com</p>",
                    unsafe_allow_html=True)
    # file upload
    if st.session_state.uploaded_file is None:
        st.markdown("<h2 style='text-align: center; color: darkblue;'>Please Upload Your Data</h2>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader(":file_folder: Upload CSV or Excel File", type=["csv", "xlsx"], key=f"file_uploader_{st.session_state.file_uploader_key}")
        if uploaded_file is not None:
            st.session_state.uploaded_file = uploaded_file
            st.session_state.df = read_data(uploaded_file.name.endswith(".csv"))  # Load DF once here
            st.rerun()

    else:
        try:
            df = st.session_state.df  # Use cached dataframe

            if df is None or df.shape[0] == 0:
                st.warning(":warning: The uploaded file is empty or invalid.")
            else:
                st.markdown("<h3 style='text-align: center; color: darkblue;'>Your Automatic EDA Generator Results</h3>", unsafe_allow_html=True)

                # data analysis begins here
                if show_overview:
                    st.markdown("#### Dataset Overview")
                    col1, col2 = st.columns([1, 4], vertical_alignment="top")
                    col_info = pd.DataFrame({"Data Type": df.dtypes.astype(str)})
                    with col1:
                        st.markdown("###### Columns Data Types: ")
                        st.dataframe(col_info)
                    with col2:
                        st.markdown("###### Data Rows and Columns - 5 records for example: ")
                        st.write(f"*Rows:* {df.shape[0]} | *Columns:* {df.shape[1]}")
                        st.dataframe(df.head(5))

                    # convert column types
                    col1, col2, col3 = st.columns([1,2, 3])
                    with col1:
                        n_unique = df.nunique()
                        n_unique.name = 'unique_values'
                        st.write(n_unique)
                    with col2:
                        st.empty()
                    with col3:
                        st.markdown("###### Select a column to change its type ")
                        change_type_col = st.selectbox("Select column for type conversion:", df.columns)
                        new_type = st.selectbox("Change to a new data type:", ["string", "numeric", "datetime"])

                    if st.button("Change Type"):
                        if new_type == "string":
                            df[change_type_col] = df[change_type_col].astype(str)
                        elif new_type == "numeric":
                            df[change_type_col] = pd.to_numeric(df[change_type_col], errors="coerce")
                        elif new_type == "datetime":
                            df[change_type_col] = pd.to_datetime(df[change_type_col], errors="coerce")

                        st.rerun()



                    st.write("")
                    col1, col2 = st.columns([4, 3], vertical_alignment="top")
                    with col1:
                        st.markdown("###### Column Statistics Numerical Columns: ")
                        st.write(df.describe())
                    with col2:
                        st.markdown("###### Categorial Columns Properties: ")
                        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
                        stats = []
                        for col in categorical_cols:
                            counts = df[col].value_counts(dropna=False)  # Frequency of each category
                            stats.append({
                                "Column": col,
                                "Unique Categories": df[col].nunique(dropna=True),
                                "Most Frequent": counts.index[0],
                                "Most Frequent Count": counts.iloc[0],
                                "Missing Values": df[col].isna().sum()
                            })

                        stats_df = pd.DataFrame(stats)
                        st.dataframe(stats_df)
                    if isinstance(df, pd.DataFrame):
                        col1, col2, col3 = st.columns([4, 1, 4], vertical_alignment="center")
                        with col1:
                            missing_counts = df.isnull().sum()
                            missing_percent = (missing_counts / len(df)) * 100

                            missing_df = pd.DataFrame({
                                'Column': missing_counts.index,
                                'Missing Count': missing_counts.values,
                                'Missing %': missing_percent.values
                            })

                            if isinstance(df, pd.DataFrame):
                                missing_df = pd.DataFrame({
                                    'Column': missing_counts.index,
                                    'Missing Count': missing_counts.values,
                                    'Missing %': missing_percent.values
                                })

                                missing_df = missing_df[missing_df['Missing Count'] > 0]

                                if not missing_df.empty:
                                    missing_df = missing_df.sort_values(by='Missing Count', ascending=False)

                                    view_option = st.radio("View missing values as:", ["Count", "Percentage"])

                                    if view_option == "Count":
                                        fig = px.bar(
                                            missing_df,
                                            x='Column', y='Missing Count', text=missing_df['Missing %'].apply(lambda x: f"{x:.1f}%"),
                                            title="Missing Values per Column (Count)", labels={'Missing Count': 'Missing Values'}
                                        )
                                        fig.update_traces(textposition='outside')

                                    else:
                                        missing_df = missing_df.sort_values(by='Missing %', ascending=False)
                                        fig = px.bar(
                                            missing_df, x='Column', y='Missing %', text=missing_df['Missing %'].apply(lambda x: f"{x:.1f}%"),
                                            title="Missing Values per Column (Percentage)", labels={'Missing %': 'Missing Values (%)'}
                                        )
                                        fig.update_traces(textposition='outside')

                                    #Adjust font sizes
                                    fig.update_layout(
                                        xaxis_tickangle=-45, font=dict(size=12), title=dict(font=dict(size=16)), xaxis=dict(title_font=dict(size=14), tickfont=dict(size=12)),
                                        yaxis=dict(title_font=dict(size=14), tickfont=dict(size=12))
                                    )
                                    st.plotly_chart(fig)
                                with col2:
                                    st.empty()
                                with col3:
                                    st.write("###### Missing Values Table")
                                    st.dataframe(missing_df.reset_index(drop=True))

                            else:
                                st.success("No missing values detected in the dataset!")
                    else:
                        st.error("data object must be a Pandas DataFrame!")
                # correlation plots
                if show_corr:
                    st.markdown("##### Correlation Analysis")
                    corr = df.select_dtypes(include=["number"]).corr()
                    col1, col2 = st.columns([2, 3], vertical_alignment="center" )
                    with col1:
                        st.markdown("##### Correlation Matrix")
                        st.write(corr)
                    with col2:
                        fig = px.imshow(
                            corr, text_auto=True, color_continuous_scale="Greys",
                            aspect="auto", title="Correlation Heatmap"
                        )
                        fig.update_layout(
                            width=600, height=500,
                            title={
                                'text': "Correlation Heatmap",
                                'font': {'size': 18},
                                'x': 0.5
                            }
                        )
                        st.plotly_chart(fig, use_container_width=True)

                #boxplot
                col1, col2 = st.columns([2, 3])
                if show_dist:
                    with col1:
                        numeric_cols = df.select_dtypes(include=['number']).columns
                        st.markdown("#### Boxplot Numerical Columns")
                        fig = go.Figure()
                        for col in numeric_cols:
                            fig.add_trace(go.Box(y=df[col], name=col, visible=True, boxmean=True))
                        buttons = [
                            dict(label="All Columns", method="update",
                                 args=[{"visible": [True]*len(numeric_cols)}, {"title": "Boxplot of All Numeric Columns"}])
                        ]
                        for i, col in enumerate(numeric_cols):
                            buttons.append(dict(label=col, method="update",
                                                args=[{"visible": [j==i for j in range(len(numeric_cols))]},
                                                      {"title": f"Boxplot of {col}"}]))
                        fig.update_layout(updatemenus=[dict(active=0, buttons=buttons, x=0.5, xanchor="center", y=1.15, yanchor="top")])
                        st.plotly_chart(fig, use_container_width=True)

                    # dist plots
                    with col2:
                        st.markdown("#### Distribution Plot")

                        # numeric_cols = df.select_dtypes(include=['number']).columns
                        if len(numeric_cols) > 0:
                            dist_columns = st.multiselect("Select column(s) for distribution plot:", numeric_cols,
                                                          key="dist_cols_main")
                            add_kde = st.checkbox("Overlay KDE (Density Curve)", False, key="kde_main")

                            if dist_columns:
                                fig = go.Figure()
                                colors = px.colors.qualitative.Set2

                                for idx, col in enumerate(dist_columns):
                                    # histogram
                                    fig.add_trace(go.Histogram(
                                        x=df[col], nbinsx=30, name=f"{col} (Hist)", marker_color=colors[idx % len(colors)], opacity=0.6
                                    ))

                                    # KDE
                                    if add_kde:
                                        valid_data = df[col].dropna()
                                        if len(valid_data) > 1 and valid_data.nunique() > 1:
                                            kde = gaussian_kde(valid_data)
                                            x_vals = np.linspace(valid_data.min(), valid_data.max(), 200)
                                            y_vals = kde(x_vals)
                                            bin_width = np.diff(np.histogram_bin_edges(valid_data, bins=30))[0]
                                            y_scaled = y_vals * len(valid_data) * bin_width
                                            fig.add_trace(go.Scatter(
                                                x=x_vals, y=y_scaled,
                                                mode='lines', name=f"{col} (KDE)",
                                                line=dict(color=colors[idx % len(colors)], width=2)
                                            ))
                                        else:
                                            st.warning(
                                                f"KDE skipped for '{col}' (not enough unique values or all NaN).")

                                fig.update_layout(
                                    xaxis_title="Value", yaxis_title="Count", barmode='overlay', bargap=0.05
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("Please select at least one numeric column to display the distribution plot.")
                        else:
                            st.warning("No numeric columns found for distribution plots.")

                # bar plots

                if show_cat and st.session_state.df is not None:
                    st.markdown("#### Bar Plots of Categorical Features")
                    col1, col2, col3 = st.columns([1, 4, 1])
                    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns
                    with col1:
                        st.empty()
                    with col2:
                        if len(categorical_cols) > 0:
                            selected_cat = st.selectbox("Select a categorical column to visualize:", categorical_cols,
                                                        key="barplot_cat")

                            if selected_cat:
                                cat_counts = df[selected_cat].value_counts().reset_index()
                                cat_counts.columns = [selected_cat, 'Count']

                                fig = px.bar(
                                    cat_counts, x=selected_cat, y='Count', text='Count', title=f"Distribution of {selected_cat}",
                                    color=selected_cat, color_discrete_sequence=px.colors.qualitative.Set2
                                )

                                fig.update_traces(textposition='outside')
                                fig.update_layout(
                                    xaxis_title=selected_cat, yaxis_title="Count", xaxis_tickangle=-45,
                                    font=dict(size=12), title=dict(font=dict(size=16))
                                )
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("No categoricals found for bar plot.")
                    with col3:
                        st.empty()

                if show_pair and st.session_state.df is not None:
                    st.markdown("#### Seaborn PairPlot")
                    col1, col2, col3 = st.columns([1, 4, 1])
                    numeric_cols = df.select_dtypes(include=['number']).columns
                    with col1:
                        st.empty()
                    with col2:
                        if len(numeric_cols) >= 2:
                            categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns
                            hue_col = None
                            if len(categorical_cols) > 0:
                                hue_col = st.selectbox("Color by (Hue):", ["None"] + list(categorical_cols),
                                                       key="hue_main")
                                if hue_col == "None":
                                    hue_col = None

                            sns.set_context("paper", font_scale=1.8)
                            sns.set_style("whitegrid")

                            # Seaborn pairplot
                            @st.cache_data
                            def generate_pairplot(data, hue):
                                return sns.pairplot(data, hue=hue, diag_kind="hist", height=3)

                            fig = generate_pairplot(df, hue_col)
                            st.pyplot(fig)
                        else:
                            st.info("Not enough numeric columns available to generate a scatter matrix.")
                    with col3:
                        st.empty()

        except Exception as e:
            st.error(f" Error loading file: {e}")
            st.session_state.uploaded_file = None
            st.session_state.df=None