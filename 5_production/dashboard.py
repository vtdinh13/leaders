######################################
# Import libraries
######################################
import streamlit as st
import pandas as pd
import plotly.graph_objects as go



import plotly.express as px
# from emotions_analysis import create_emotions_matrix, one_hot_encoding, create_clusters, create_embeddings
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np


from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image

import nltk
from nltk.util import ngrams
import numpy as np

from sklearn.preprocessing import MultiLabelBinarizer
from sentence_transformers import SentenceTransformer


#######################################
# Page configuration
#######################################
st.set_page_config(
    page_title="Trump Sentiment Analysis",
    # page_icon=":bar_chart:",
    layout="wide",
    # initial_sidebar_state="expanded"
)

######################################
# Load data
######################################
@st.cache_data

def load_data():    
    df = pd.read_parquet('./1_data_collection/trump_second_term_v2.parquet')
    return df
df = load_data()

########################################
# Side bar
########################################
with st.sidebar:
    # with st.expander("About", expanded = True):
    st.title("About")
    st.write("""
            - Data was scraped from scraping [Rollcall](https://rollcall.com/).
             """)
    




########################################
# Date slider 

# st.title("Date filter")
df["date"] = pd.to_datetime(df["date"]).dt.date

date_range = (df["date"].min(), df["date"].max())
start_date, end_date = st.slider(
    "Select a date range",
    value=date_range,
    min_value=date_range[0],
    max_value=date_range[1],
    format="YYYY-MM-DD"
)
# st.write("Selected date range:", df["date"].min(), "to", df["date"].max())
# st.write("Selected date range:", x)

# Filter DataFrame based on selected range
mask = (df['date'] >= start_date) & (df['date'] <= end_date)
filtered_df = df.loc[mask]
########################################





##############################
# Visualization
###############################
sentiment_map = {
    "very negative" : 0, 
    "negative": 1, 
    "neutral": 2,
    "positive": 3, 
    "very positive": 4
}

map_num_to_sentiment = {
    0: "Very Negative",
    1: "Negative",
    2: "Neutral",
    3: "Positive",
    4: "Very Positive"
}

intent_mapping = {
    "very negative" : "Very Negative 😡",
    "negative" : "Negative 😠",
    "neutral" : "Neutral 😐",
    "positive": "Positive 🙂",
    "very positive": "Very Positive 😊",
}

def create_sentiment_pie_chart(df):
    sent_df = pd.DataFrame(df.groupby("sentiment").size(), columns=["count"]).reset_index()

    sent_df["sentiment"]= sent_df["sentiment"].map(sentiment_map)
    sent_df["proportion"] = sent_df["count"].transform(lambda x:x/x.sum()).round(2)
    sent_df_sorted = sent_df.sort_values("sentiment")

    pie = go.Figure(data=[go.Pie(
        labels=sent_df_sorted["sentiment"].map(map_num_to_sentiment),
        values=sent_df_sorted["proportion"],
        customdata=sent_df_sorted[["count"]],
        sort=False,
        hovertemplate='<b>%{label}</b><br>Count: %{customdata[0]}<br>Percent: %{percent}<extra></extra>'
    )])

    # pie.update_layout(
    #     legend_title_text='Sentiments by Vertex AI')

    pie.update_layout(showlegend=False)
    return pie

def create_emotions_stacked(df):

    # Create subset of original dataframe
    df_emotion_sentiment = df[["sentiment", "lemmatized_emotions_clean"]].reset_index(drop=True)

    # Create emotions matrix
    mlb = MultiLabelBinarizer()
    emotions_matrix = pd.DataFrame(mlb.fit_transform(df_emotion_sentiment["lemmatized_emotions_clean"]), columns=mlb.classes_)
    emotions_matrix = emotions_matrix.reset_index(drop=True)


    # Create sentiment-emotion crosstab
    e_s_merged = pd.concat([df_emotion_sentiment["sentiment"], emotions_matrix], axis=1).reset_index(drop=True)

    df_long = e_s_merged.melt(id_vars="sentiment", value_vars=emotions_matrix.columns, 
                              var_name="emotion", value_name="present")
    
    df_long = df_long[df_long["present"]==1]

    crosstab = pd.crosstab(df_long["sentiment"], df_long["emotion"])

    grouped = crosstab.T.groupby("emotion").sum()

    grouped.columns = grouped.columns.map(sentiment_map)

    # Prepare data
    top_n = grouped.sum(axis=1).nlargest(20).index

    grouped_top = grouped.loc[top_n]


    grouped_percent = grouped_top.div(grouped_top.sum(axis=1), axis=0) * 100
    grouped_percent_sorted = grouped_percent[sorted(grouped_percent.columns)]
    
    grouped_percent_sorted_renamed = grouped_percent_sorted.rename(columns=map_num_to_sentiment)
    
    # Create a stacked bar chart
    fig = go.Figure()

    for i, sentiment in enumerate(grouped_percent_sorted_renamed.columns):
        fig.add_trace(go.Bar(
            x=grouped_percent_sorted_renamed.index,
            y=grouped_percent_sorted_renamed[sentiment],
            name=sentiment,
            # marker_color=colors[i % len(colors)],
            customdata=grouped_percent_sorted_renamed[[sentiment]].to_numpy(),  # <-- pass raw counts as customdata
            hovertemplate='%{y:.0f}% (%{customdata[0]} counts)<extra>%{fullData.name}</extra>'
        ))

    # Customize layout
    fig.update_layout(
        barmode='stack',
        title='Top 20 Emotions by Total Sentiment Mentions',
        xaxis_title='Affect Categories',
        yaxis_title='Percent',
        xaxis_tickangle=-45,
    )
    fig.update_layout(
        legend=dict(
            title="Sentiments by Vertex AI",
            orientation="v",   # vertical
            traceorder="normal"  # or "reversed" to flip
        )
    )
    return fig

def create_wordcloud(df):
    topics_clean = df["entities_clean"].explode()
    words = ",".join(topics_clean)
    words = "_".join(words.split(" "))
    words = words.replace(",", " ")
    
    words_to_remove = {"donald_trump", "(", ")", "."}
    filtered_text = [w for w in words.split() if w not in words_to_remove]
    ngram_freq = Counter(filtered_text)

    mask = np.array(Image.open("/Users/vancescadinh/Documents/Projects/leaders/3_explore_and_feature_engineering/map.png"))
    wordcloud = WordCloud(
        width=800,
        height=800,
        background_color='white',
        mask=mask,             
    ).generate_from_frequencies(ngram_freq)
    plt.figure(figsize=(10, 10))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    return plt

################################
# Main dashbaord
################################
top_row = st.columns((3, 2), gap='medium')

with top_row[0]:
    st.subheader("This is for a display of intent.")   
    if not filtered_df.empty:
        intent_df = filtered_df[["sentiment", "intent"]]
        intent_df.loc[:, "sentiment"] = intent_df.loc[:,"sentiment"].map(intent_mapping)
        st.dataframe(intent_df.reset_index(drop=True))    
# with col[1]:
#     st.subheader("Sentiment Pie Chart")
#     # st.write("This pie chart shows the distribution of sentiments in the selected date range with data based on documents with Trump as the primary speaker.")

#     if not filtered_df.empty:
#         pie = create_sentiment_pie_chart(filtered_df)   
#         st.plotly_chart(pie, use_container_width=True)
#     else:
#         st.write("No data available for the selected date range.")
#         st.write("Please select a different date range.")

with top_row[1]:
    st.subheader("Top 20 Emotions by Total Sentiment Mentions")
    # st.write("This stacked bar chart shows the top 20 emotions by total sentiment mentions in the selected date range with data based on documents with Trump as the primary speaker.")
    if not filtered_df.empty:
        stacked = create_emotions_stacked(filtered_df)
        st.plotly_chart(stacked, use_container_width=True)
    else:
        st.write("No data available for the selected date range.")
        st.write("Please select a different date range.")


bottom_row = st.columns((3,2))
with bottom_row[0]:    
    if not filtered_df.empty:
        cloud = create_wordcloud(filtered_df)
        st.pyplot(cloud)

with bottom_row[1]:
# st.subheader("Sentiment Pie Chart")
# st.write("This pie chart shows the distribution of sentiments in the selected date range with data based on documents with Trump as the primary speaker.")

    if not filtered_df.empty:
        pie = create_sentiment_pie_chart(filtered_df)   
        st.plotly_chart(pie, use_container_width=True)
    else:
        st.write("No data available for the selected date range.")
        st.write("Please select a different date range.")

