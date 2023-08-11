import pickle
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Steam Game's Recommender",
    page_icon="🐼",
    layout="wide",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    st.title("Contact Me")
    st.divider()
    st.write("Instagram: https://www.instagram.com/ashishprasad__/")
    st.write("LinkedIn: https://www.linkedin.com/in/ashish-prasad-92223a228/")
    st.write("Email: ashishprasd@gamil.com")


def recommend(game):
    index = new_df[new_df['Name'] == game].index[0]
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
    game_lists=[]
    game_image_list=[]
    for i in distances[1:7]:
        game_lists.append(new_df.iloc[i[0]].Name)
        game_image_list.append(game_image.iloc[i[0]].Headerimage)
    return game_lists,game_image_list
        
# uplode data
new_df=pickle.load(open("/workspaces/codespaces-jupyter/notebooks/game_data.pkl",'rb'))
similarity=pickle.load(open("/workspaces/codespaces-jupyter/notebooks/similarity.pkl",'rb'))
game_image=pd.read_csv("/workspaces/codespaces-jupyter/notebooks/image_data.csv")
game_name = new_df['Name'].values

st.title("Steam Game's Recommendation System by Ashish 🐼")
st.warning('There was only New Game Data Since 2021 to 2023')

st.divider()

option = st.selectbox(
    'Select or type The Game Name: ⬇️',
    game_name)

st.write('You selected:', option)
if st.button('🔍 Find!'):
    st.toast("🙃 Enjoy!")
    st.balloons()
    st.header("Here are The Top 6 Game You also like 🎮:")
    rec,img=recommend(option)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(img[0])
        st.subheader(rec[0])
    with col2:
        st.image(img[1])
        st.subheader(rec[1])
    with col3:
        st.image(img[2])
        st.subheader(rec[2])
        
    col4,col5,col6 =st.columns(3)
    with col4:
        st.image(img[3])
        st.subheader(rec[3])
    with col5:
        st.image(img[4])
        st.subheader(rec[4])
    with col6:
        st.image(img[5])
        st.subheader(rec[5])
    st.success("🐰 Match Found!")

