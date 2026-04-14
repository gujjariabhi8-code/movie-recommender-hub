import streamlit as st
import pickle
import pandas as pd
import requests
from google import genai
import plotly.express as px

# ==========================================
# 1. SETUP & SECURE API KEYS
# ==========================================
# Keys are pulled from Streamlit's secrets manager for cloud deployment
OMDB_API_KEY = st.secrets["OMDB_API_KEY"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

# Initialize Gemini 2.5 Flash Client
client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_ID = 'gemini-2.5-flash'

# ==========================================
# 2. FUNCTIONS (Updated for Method 1)
# ==========================================
def fetch_poster(movie_title):
    formatted_title = movie_title.replace(" ", "+")
    url = f"http://www.omdbapi.com/?t={formatted_title}&apikey={OMDB_API_KEY}"
    try:
        response = requests.get(url)
        data = response.json()
        if data.get('Response') == 'True' and data.get('Poster') != 'N/A':
            return data['Poster']
        else:
            return "https://via.placeholder.com/500x750?text=No+Poster"
    except Exception as e:
        return "https://via.placeholder.com/500x750?text=Error"

def recommend(movie):
    # Find the index of the selected movie
    movie_index = movies[movies['title'] == movie].index[0]
    
    # NEW METHOD 1 LOGIC: No calculating or sorting!
    # Retrieve the pre-computed top 5 nearest neighbors directly from the dictionary
    movies_list = similarity[movie_index][:5] 
    
    recommended_movies = []
    recommended_movies_posters = []
    
    for i in movies_list:
        movie_title = movies.iloc[i[0]].title
        recommended_movies.append(movie_title)
        recommended_movies_posters.append(fetch_poster(movie_title))
        
    return recommended_movies, recommended_movies_posters

# ==========================================
# 3. LOAD DATA
# ==========================================
# Load the ML data
movies_dict = pickle.load(open('movie_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)

# Load the newly compressed Top 10 similarity dictionary
similarity = pickle.load(open('similarity_top10.pkl', 'rb'))

# Load the EDA data for the dashboard
try:
    movies_eda = pickle.load(open('movies_eda.pkl', 'rb')) 
except FileNotFoundError:
    st.error("⚠️ movies_eda.pkl not found! The dashboard tab will not work.")
    movies_eda = pd.DataFrame()

# ==========================================
# 4. APP UI & TABS
# ==========================================
st.set_page_config(page_title="Movie Hub", layout="wide")
st.title('🎬 Movie Recommender & Analytics Hub')

tab1, tab2 = st.tabs(["🤖 Recommendation Engine", "📊 Data Analytics Dashboard"])

# ------------------------------------------
# TAB 1: RECOMMENDER & CHATBOT
# ------------------------------------------
with tab1:
    st.header("Find Your Next Movie")
    
    selected_movie_name = st.selectbox(
        'Select a movie to get recommendations:',
        movies['title'].values
    )

    if st.button('Recommend'):
        names, posters = recommend(selected_movie_name)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.text(names[0])
            st.image(posters[0])
        with col2:
            st.text(names[1])
            st.image(posters[1])
        with col3:
            st.text(names[2])
            st.image(posters[2])
        with col4:
            st.text(names[3])
            st.image(posters[3])
        with col5:
            st.text(names[4])
            st.image(posters[4])

    # --- CHATBOT SECTION ---
    st.markdown("---")
    st.subheader("🤖 Ask the AI Movie Assistant")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! Ask me for recommendations, search by genre/theme, or ask why a specific movie was recommended!"}
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("E.g., Suggest thriller movies with strong female leads..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        current_movie_context = ""
        if selected_movie_name:
            movie_index = movies[movies['title'] == selected_movie_name].index[0]
            movie_tags = movies.iloc[movie_index].tags
            current_movie_context = f"\nContext: The user is currently looking at the movie '{selected_movie_name}'. Its metadata tags are: {movie_tags[:500]}..."

        system_prompt = f"""
        You are an expert movie recommendation assistant. 
        1. If they ask for movies similar to a specific movie, provide 5 movies with a short reason.
        2. If they ask for specific themes, suggest 5 highly acclaimed movies that fit.
        3. If they ask "Why did you recommend this?", use the context below to explain similarities.
        {current_movie_context}
        User Query: {prompt}
        """

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            try:
                response = client.models.generate_content(
                    model=MODEL_ID,
                    contents=system_prompt,
                )
                reply = response.text
                message_placeholder.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})
            except Exception as e:
                error_msg = f"Sorry, I encountered an error connecting to the AI: {e}"
                message_placeholder.markdown(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# ------------------------------------------
# TAB 2: PLOTLY DASHBOARD
# ------------------------------------------
with tab2:
    if not movies_eda.empty:
        st.header("Explore the TMDB Dataset")
        st.markdown("Interactive visualizations exploring the metadata of 4,800+ movies.")

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.subheader("1. Most Popular Genres")
            genres_exploded = movies_eda.explode('genres')
            genre_counts = genres_exploded['genres'].value_counts().reset_index()
            genre_counts.columns = ['Genre', 'Count']
            
            fig_genres = px.bar(
                genre_counts.head(10), 
                x='Count', 
                y='Genre', 
                orientation='h',
                color='Count',
                color_continuous_scale='Viridis',
            )
            fig_genres.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_genres, use_container_width=True)

        with chart_col2:
            st.subheader("2. Most Frequent Directors")
            crew_exploded = movies_eda.explode('crew')
            director_counts = crew_exploded['crew'].value_counts().reset_index()
            director_counts.columns = ['Director', 'Movies Directed']
            
            fig_directors = px.pie(
                director_counts.head(10), 
                names='Director', 
                values='Movies Directed',
                hole=0.4,
            )
            fig_directors.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_directors, use_container_width=True)

        st.subheader("3. Most Frequently Cast Actors")
        cast_exploded = movies_eda.explode('cast')
        actor_counts = cast_exploded['cast'].value_counts().reset_index()
        actor_counts.columns = ['Actor', 'Appearances']
        
        fig_actors = px.bar(
            actor_counts.head(15), 
            x='Actor', 
            y='Appearances',
            color='Appearances',
            color_continuous_scale='Plasma',
        )
        st.plotly_chart(fig_actors, use_container_width=True)
