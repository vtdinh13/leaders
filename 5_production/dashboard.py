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
    layout="wide"
)

######################################
# Load data
######################################
@st.cache_data

def load_data():    
    df = pd.read_parquet('./1_data_collection/trump_second_term_v2.parquet')
    return df
df = load_data()
df["text_word_count"] = df["text"].str.split().apply(len)

########################################
# Side bar
########################################
with st.sidebar: 
    with st.expander("Dataset Information", expanded=True):
        st.markdown("# Dataset")
    # st.markdown(
    #     "<h1 style='margin-top: 0rem;'>About</h1>",
    #     unsafe_allow_html=True)
        
        st.markdown("Over 4000 documents from Donald Trump were scraped from [Roll Call](https://rollcall.com/) covering the period from January 2016 to April 2025. The original dataset includes raw speeches and transcriptions of remarks, commentaries, and public messages spoken by Trump. Secondary sources, such as analyses of Trump, as well as short tweets from X and Truth Social were excluded. This means that the compiled dataset reflects what Trump has directly said himself -- raw and unfiltered. The final dataset comprised of responses from [Gemini 1.5 Flash](https://ai.google.dev/gemini-api/docs/models#gemini-1.5-flash).")

    # st.title("Date filter")
    df["date"] = pd.to_datetime(df["date"]).dt.date

    date_range = (df["date"].min(), df["date"].max())

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("#### Given that insights are based on a particular timeframe, please select the desired date range below:")
    start_date, end_date = st.slider("",
        value=date_range,
        min_value=date_range[0],
        max_value=date_range[1],
        format="YYYY-MM-DD"
    )

    # Filter DataFrame based on selected range
    mask = (df['date'] >= start_date) & (df['date'] <= end_date)
    filtered_df = df.loc[mask]

    # st.markdown("The final dataset, comprised of responses from [Gemini 1.5 Flash](https://ai.google.dev/gemini-api/docs/models#gemini-1.5-flash), was employed to conduct analyses." )
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
            '<h6>Made in &nbsp<img src="https://streamlit.io/images/brand/streamlit-mark-color.png" alt="Streamlit logo" height="16">&nbsp by <a href="https://www.linkedin.com/public-profile/settings?trk=d_flagship3_profile_self_view_public_profile">Vancesca Dinh</a></h6>'
            ,unsafe_allow_html=True)


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
    colors = custom_color_scheme()
    pie = go.Figure(data=[go.Pie(
        labels=sent_df_sorted["sentiment"].map(map_num_to_sentiment),
        values=sent_df_sorted["proportion"],
        customdata=sent_df_sorted[["count"]],
        domain=dict(x=[0.15, 0.95], y=[0.45, 0.9]),
        sort=False,
        hole=0.5,
        marker=dict(colors=colors),
        hovertemplate='<b>%{label}</b><br>Count: %{customdata[0]}<br>Percent: %{percent}<extra></extra>'
    )])

    pie.update_layout(
        legend_title_text='Sentiments by Gemini 1.5 Flash')

    pie.update_layout(showlegend=True)

    # Control the legend position
    pie.update_layout(
        legend=dict(
            x=0.3,       # horizontal position (0 to 1)
            y=0.3,       # vertical position (0 to 1)
            xanchor='left',  # anchor point on the x-axis ('left', 'center', 'right')
            yanchor='top',   # anchor point on the y-axis ('top', 'middle', 'bottom')
        ),
        height=400,
        width=500,
        margin=dict(l=5, r=5, t=5, b=5)
    )

    return pie

# def sample_colorscale(colorscale, n):
#     return [colorscale[int(i)] for i in np.linspace(0, len(colorscale) - 1, n)]

def custom_color_scheme():
    colors = [
    "#1d8348",  
    "#27ae60",  
    "#fdebd0",  
    "#3498db",  
    "#2874a6",  
    
]
    return colors
def create_heatmap(df):

    # Create subset of original dataframe
    # df_emotion_sentiment = df[["sentiment", "lemmatized_emotions_clean"]].reset_index(drop=True)

    # Create emotions matrix
    mlb = MultiLabelBinarizer()
    emotions_matrix = pd.DataFrame(mlb.fit_transform(df["lemmatized_emotions_clean"]), columns=mlb.classes_)
    emotions_matrix = emotions_matrix.reset_index(drop=True)


    # Create sentiment-emotion crosstab
    e_s_merged = pd.concat([df["sentiment"], emotions_matrix], axis=1).reset_index(drop=True)

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
    grouped_percent_sorted_renamed_T = grouped_percent_sorted_renamed.T
    # Create plot
    # colors = px.colors.sequential.Greens
    # colors = px.colors.sequential.RdBu
    # colors = px.colors.diverging.RdBu[::-1]
    # colors = px.colors.diverging.RdBu
    # colors = sample_colorscale(px.colors.diverging.RdBu, 5)
    # colors = custom_color_scheme()

    # fig = go.Figure()

    # for i, sentiment in enumerate(grouped_percent_sorted_renamed.columns):
    #     fig.add_trace(go.Bar(
    #         x=grouped_percent_sorted_renamed.index,
    #         y=grouped_percent_sorted_renamed[sentiment],
    #         name=sentiment,
    #         marker_color=colors[i % len(colors)]
    #         # customdata=grouped_percent_sorted[[sentiment]].to_numpy(),  # <-- pass raw counts as customdata
    #         # hovertemplate='%{y:.0f}% (%{customdata[0]} counts)<extra>%{fullData.name}</extra>'
    #     ))
  
   
    # # Customize layout
    # fig.update_layout(
    #     barmode='stack',
    #     title='Top 20 Emotions by Total Sentiment Mentions',
    #     xaxis_title='Affect Categories',
    #     yaxis_title='Percent',
    #     xaxis_tickangle=-45
    # )
    # fig.update_layout(
    #     legend=dict(
    #         title="Sentiments by Gemini 1.5 Flash",
    #         orientation="v",   # vertical
    #         traceorder="normal"  # or "reversed" to flip
    #     )
    # )

    fig = px.imshow(
    grouped_percent_sorted_renamed.T,
    text_auto='.0f',
    # text_auto=False,
    labels=dict(x="Associated Emotion", y="Sentiment", color="Percentage"),
    # labels = dict(x="Sentiment", y="Associated Emotion", color="Percentage"),
    # title="Sentiment vs Emotion Cluster (%)",
    color_continuous_scale="Greens"
)
    fig.update_layout(height=350, width=600, xaxis_title_font=dict(size=18, family='Arial', color='black'),  # bold via default font weight
        yaxis_title_font=dict(size=18, family='Arial', color='black'),
        xaxis=dict(
            tickfont=dict(size=14, family='Arial', color='black')
        ),
        yaxis=dict(
            tickfont=dict(size=14, family='Arial', color='black')
        ),
        # title={
        #     'text': "Sentiment Analysis of DJT Documents",
        #     'font': {
        #         'size': 24,          # font size
        #         'family': 'Arial',   # font family
        #         'color': 'black'  # font color
        #     },
        #     'x': 0.5,                # center title
        #     'xanchor': 'center'
        # },
        margin=dict(l=10, r=10, t=10, b=10))
    # Update colorbar (legend) to be horizontal
    fig.update_layout(
        coloraxis_colorbar=dict(
            orientation='v',      # horizontal
            x=1.1,                # center the colorbar
            xanchor='center',
            y=.53,               # adjust vertical position if needed
            title='Percentage'     # optional label
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

    mask = np.array(Image.open("./3_explore_and_feature_engineering/map.png"))
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

# # Set background color with HTML + CSS
# st.markdown(
#     """
#     <style>
#     body {
#         background-color: #eaeded;  /* Light blue background */
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

st.markdown(
    """
    <style>
    /* Main page background */
   .stApp {
    background-color: #f4f6f6;
}
[data-testid="stSidebar"] {
    background-color: #d4efdf;
}
    </style>
    """,
    unsafe_allow_html=True
)

row0 = st.columns((.4, 1, 1, 1, .4), gap="medium")
with row0[1]:
    if not filtered_df.empty:
        num_doc = len(filtered_df)
        st.markdown(f"""
            <div style="
                width: 190px;
                height: 150px;
                border: .5px solid #ddd;
                border-radius: 12px;
                padding:2px;
                background-color: #f9f9f9;
                display: inline-block;
                vertical-align: top;
                text-align: center;
                margin: 15px;">
                <h4 style="margin: 0;">No. of Documents</h4>
                <p style="font-size: 44px; margin: 0;"><strong>{num_doc}</strong></p>
            </div>
        """, unsafe_allow_html=True)

with row0[2]:
    if not filtered_df.empty:
        max_doc = filtered_df["text_word_count"].max()
        st.markdown(f"""
            <div style="
                width: 190px;
                height: 150px;
                border: .5px solid #ddd;
                border-radius: 12px;
                padding:2px;
                background-color: #f9f9f9;
                display: inline-block;
                vertical-align: top;
                text-align: center;
                margin: 15px;">
                <h4 style="margin: 0px auto;">Most No. of Words</h4>
                <p style="font-size: 44px; margin: 0;"><strong>{max_doc}</strong></p>
            </div>
        """, unsafe_allow_html=True)


with row0[3]:
    if not filtered_df.empty:
        min_doc = filtered_df["text_word_count"].min()
        st.markdown(f"""
            <div style="
                width: 190px;
                height: 150px;
                border: .5px solid #ddd;
                border-radius: 12px;
                padding:2px;
                background-color: #f9f9f9;
                display: inline-block;
                vertical-align: top;
                text-align: center;
                margin: 15px;">
                <h4 style="margin: 0;">Least No. of Words</h4>
                <p style="font-size: 44px; margin: 0;"><strong>{min_doc}</strong></p>
            </div>
        """, unsafe_allow_html=True)


# with row0[3]:
   


row1 = st.columns((1))

with row1[0]:
    # st.subheader("Top 20 Emotions by Total Sentiment Mentions")
    # st.write("This stacked bar chart shows the top 20 emotions by total sentiment mentions in the selected date range with data based on documents with Trump as the primary speaker.")
    if not filtered_df.empty:
        hm = create_heatmap(filtered_df)
        st.plotly_chart(hm, use_container_width=True)
    else:
        st.write("No data available for the selected date range.")
        st.write("Please select a different date range.")

# with row1[1]:
#     st.subheader("Intended Purpose of Documents")
#     st.markdown("Gemini's response to the summary of a given document and the purpose that it serves.")   
#     if not filtered_df.empty:
#         intent_df = filtered_df[["date", "sentiment", "intent"]]
#         intent_df = intent_df[~intent_df["intent"].isna()]
#         intent_df.loc[:, "sentiment"] = intent_df.loc[:,"sentiment"].map(intent_mapping)
#         intent_df.columns = ["Date", "Sentiment", "Summary and Intended Purpose of Document"]
#         intent_df.index = intent_df["Date"]
#         intent_df = intent_df.drop("Date", axis=1)
#         st.dataframe(intent_df)    







row2 = st.columns((1.6,1))
with row2[0]:    
    if not filtered_df.empty:
        st.subheader("Named-Entity Recognition (NER) Extraction")
        cloud = create_wordcloud(filtered_df)
        st.pyplot(cloud)

with row2[1]:
    st.subheader("Sentiment Distribution")
    if not filtered_df.empty:
        pie = create_sentiment_pie_chart(filtered_df)   
        st.plotly_chart(pie, use_container_width=True)

   
