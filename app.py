import streamlit as st
import base64


# Configure the app layout
st.set_page_config(
    page_title="Multi-Page Application",
    page_icon="🌟",
    layout="wide",
)

# Main header
st.title("Welcome to Project Portfolio")
st.sidebar.success("Select a page from the sidebar.")

st.write("This is the main app page. Use the sidebar to navigate between pages.")



######################################### Profiles   #################################################


# Function to encode image as Base64
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Convert images to Base64
hacker_rank_img = get_base64_image("Images/icons/HackerRank.png")
github_img = get_base64_image("Images/icons/Github.png")
linkedin_img = get_base64_image("Images/icons/Linkedin.png")

# Display social media icons
st.markdown(f'''
<p align="left" style="display: flex; gap: 50px;">
    <a href="https://www.hackerrank.com/profile/hardikmpatil23" target="_blank">
        <img src="data:image/png;base64,{hacker_rank_img}" alt="HackerRank" title="HackerRank Profile" height="40" width="40" />
    </a>
    <a href="https://github.com/Hardikpatil23-HP" target="_blank">
        <img src="data:image/png;base64,{github_img}" alt="GitHub" title="GitHub Profile" height="40" width="40" />
    </a>
    <a href="https://www.linkedin.com/in/hardik-patil-164066226" target="_blank">
        <img src="data:image/png;base64,{linkedin_img}" alt="LinkedIn" title="LinkedIn Profile" height="40" width="40" />
    </a>
</p>
''', unsafe_allow_html=True)


