import streamlit as st
import psycopg2
import pandas as pd
import plotly.express as px
import time

DB = {"host": "postgres", "port": 5432, "database": "aerostream", "user": "aerostream", "password": "aerostream123"}

def get_data():
    try:
        conn = psycopg2.connect(**DB)
        df = pd.read_sql("SELECT * FROM predictions ORDER BY created_at DESC", conn)
        conn.close()
        return df
    except:
        return pd.DataFrame()

st.set_page_config(page_title="AeroStream", layout="wide")
st.title("✈️ AeroStream Analytics")

placeholder = st.empty()
counter = 0

while True:
    counter += 1
    with placeholder.container():
        df = get_data()
        
        if df.empty:
            st.warning("Waiting for data...")
        else:
            # KPIs
            c1, c2, c3 = st.columns(3)
            c1.metric("📊 Tweets", f"{len(df):,}")
            c2.metric("🛫 Airlines", df['airline'].nunique())
            c3.metric("😞 Negative", f"{(df['sentiment']=='negative').mean()*100:.1f}%")
            
            st.divider()
            
            # Charts row
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Volume par Compagnie")
                vol = df['airline'].value_counts().reset_index()
                vol.columns = ['Airline', 'Count']
                st.plotly_chart(px.bar(vol, x='Airline', y='Count', color='Airline'), use_container_width=True, key=f"v{counter}")
            
            with col2:
                st.subheader("Sentiments")
                sent = df['sentiment'].value_counts().reset_index()
                sent.columns = ['Sentiment', 'Count']
                st.plotly_chart(px.pie(sent, values='Count', names='Sentiment', 
                    color='Sentiment', color_discrete_map={'positive':'green','neutral':'gray','negative':'red'}), 
                    use_container_width=True, key=f"s{counter}")
            
            # Satisfaction
            st.subheader("Satisfaction par Compagnie")
            sat = df.groupby('airline').apply(lambda x: (x['sentiment']=='positive').mean()*100).reset_index()
            sat.columns = ['Airline', 'Satisfaction %']
            st.plotly_chart(px.bar(sat.sort_values('Satisfaction %', ascending=False), 
                x='Airline', y='Satisfaction %', color='Satisfaction %', color_continuous_scale='RdYlGn'), 
                use_container_width=True, key=f"sat{counter}")
            
            # Negative causes
            st.subheader("Causes Négatives (mots fréquents)")
            neg = df[df['sentiment']=='negative']
            if not neg.empty:
                words = ' '.join(neg['cleaned_text'].fillna('')).split()
                stops = {'the','a','an','is','was','were','with','and','or','to','for','my','i','me','on','it'}
                words = [w for w in words if len(w)>3 and w not in stops]
                wf = pd.Series(words).value_counts().head(10).reset_index()
                wf.columns = ['Word', 'Count']
                st.plotly_chart(px.bar(wf, x='Count', y='Word', orientation='h'), use_container_width=True, key=f"w{counter}")
            
            # Recent tweets
            st.subheader("Tweets Récents")
            st.dataframe(df[['created_at','airline','sentiment','original_text']].head(10), use_container_width=True)
            st.caption(f"Updated: {pd.Timestamp.now().strftime('%H:%M:%S')}")
    
    time.sleep(30)
