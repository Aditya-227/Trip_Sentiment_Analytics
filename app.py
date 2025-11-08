import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import io

st.set_page_config(page_title="Trip Sentiment Analytics Dashboard", layout="wide")

st.image("assets/zingbus_logo.png", width=220)
st.markdown("<h2 style='text-align:center;'>🚌 Trip Experience & Feedback Sentiment Analytics</h2>", unsafe_allow_html=True)

DATA_PATH = r"data_clean/google_sentiment_output.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
    df['Year'] = df['created_at'].dt.year
    df['Month'] = df['created_at'].dt.strftime("%B")
    return df

df = load_data()

st.sidebar.header("🔍 Filter & Search")
year = st.sidebar.multiselect("Select Year", sorted(df['Year'].dropna().unique()))
month = st.sidebar.multiselect("Select Month", sorted(df['Month'].dropna().unique()))
sentiment = st.sidebar.multiselect("Select Sentiment", df['BERT_Sentiment'].unique())
rating = st.sidebar.multiselect("Select Rating", sorted(df['rating'].dropna().unique()))
search_text = st.sidebar.text_input("Search in Reviews")

filtered_df = df.copy()
if year:
    filtered_df = filtered_df[filtered_df['Year'].isin(year)]
if month:
    filtered_df = filtered_df[filtered_df['Month'].isin(month)]
if sentiment:
    filtered_df = filtered_df[filtered_df['BERT_Sentiment'].isin(sentiment)]
if rating:
    filtered_df = filtered_df[filtered_df['rating'].isin(rating)]
if search_text:
    filtered_df = filtered_df[filtered_df['text'].str.contains(search_text, case=False, na=False)]

avg_conf = round(filtered_df['Confidence'].mean() * 100, 2) if not filtered_df.empty else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("🟢 Positive Reviews", filtered_df[filtered_df['BERT_Sentiment'] == 'POSITIVE'].shape[0])
col2.metric("🟡 Neutral Reviews", filtered_df[filtered_df['BERT_Sentiment'] == 'NEUTRAL'].shape[0])
col3.metric("🔴 Negative Reviews", filtered_df[filtered_df['BERT_Sentiment'] == 'NEGATIVE'].shape[0])
with col4:
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=avg_conf,
        title={'text': "Avg Confidence (%)"},
        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#4CAF50"}}
    ))
    st.plotly_chart(gauge, use_container_width=True)

tab1, tab2, tab3 = st.tabs(["📈 Sentiment Trends", "⭐ Ratings & Distribution", "☁️ Word Cloud"])

with tab1:
    trend = filtered_df.groupby(['Year', 'BERT_Sentiment']).size().reset_index(name='Count')
    if not trend.empty:
        fig = px.line(trend, x='Year', y='Count', color='BERT_Sentiment',
                      color_discrete_map={'POSITIVE': '#4CAF50', 'NEUTRAL': '#FFC107', 'NEGATIVE': '#F44336'},
                      markers=True)
        fig.update_layout(xaxis_title="Year", yaxis_title="Count of Reviews")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data available for selected filters.")

with tab2:
    colA, colB = st.columns(2)
    dist = filtered_df.groupby(['rating', 'BERT_Sentiment']).size().reset_index(name='Count')
    with colA:
        if not dist.empty:
            fig = px.bar(dist, x='rating', y='Count', color='BERT_Sentiment', barmode='group',
                         color_discrete_map={'POSITIVE': '#4CAF50', 'NEUTRAL': '#FFC107', 'NEGATIVE': '#F44336'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available for selected filters.")
    with colB:
        if not filtered_df.empty:
            fig = px.pie(filtered_df, names='rating', hole=0.55,
                         title='Rating Share', color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available for selected filters.")

with tab3:
    text = " ".join(filtered_df['text'].dropna().astype(str))
    if text.strip():
        wc = WordCloud(width=800, height=400, background_color="white",
                       colormap="plasma", max_words=100).generate(text)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)
    else:
        st.info("No text available for word cloud.")

st.subheader("📋 Sample Reviews")
st.dataframe(filtered_df[['created_at', 'rating', 'BERT_Sentiment', 'Confidence', 'text']].head(20), use_container_width=True)

if not filtered_df.empty:
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Filtered Data", csv, "filtered_reviews.csv", "text/csv")
