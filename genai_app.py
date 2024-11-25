import streamlit as st
from PIL import Image

from langchain_google_genai import ChatGoogleGenerativeAI
import base64

f = open(r"G:\Innomatics-Data Science\GenAI\Key\Geminikey.txt")
GOOGLE_API_KEY = f.read()

chat_model = ChatGoogleGenerativeAI(google_api_key=GOOGLE_API_KEY,model="gemini-1.5-flash")

st.title("AI Assistant for Visually Impaired Individuals")

st.header("Upload an Image")

uploaded_file = st.file_uploader("")
if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    bytes_data = uploaded_file.getvalue()
    image_data = base64.b64encode(bytes_data).decode("utf-8")



def object_and_obstacle_detection_for_safe_navigation(final_image_data):

    from langchain_core.prompts import ChatPromptTemplate

    chat_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", """You are now an AI assistant for Visually Impaired Individuals.
            You need to identify objects or obstacles within the image and need to highlight them and print the image. 
            By identifying the objects or obstacles within the image you need to give insights to enhance user safety and also
            give insights to know about the situation in the image to enhance situational awareness for the individual."""),
        (
            "user",
            [
                {"type": "text", "text": "Describe the image"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,{image}"},
                }
            ],
        ),
    ]
    )
    from langchain_core.output_parsers import StrOutputParser
    output_parser = StrOutputParser()


    chain = chat_prompt_template | chat_model | output_parser
    user_input = {"image":final_image_data}

    st.header("Description and Context of Image:")
    st.write(chain.invoke(user_input))


def personalized_assistance_for_daily_tasks(final_image_data):

    from langchain_core.prompts import ChatPromptTemplate

    chat_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", """You are now an AI assistant for Visually Impaired Individuals.
            You need to recognize items in the image and understand the labels if any and provide context specific information. 
            And also provide the context in the image and describe the importance of the context in the image effectively,which reflects human tasks in day to day life.
            Provide the information point wise."""),
        (
            "user",
            [
                {"type": "text", "text": "Describe the image"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,{image}"},
                }
            ],
        ),
    ]
    )
    


    from langchain_core.output_parsers import StrOutputParser
    output_parser = StrOutputParser()


    chain = chat_prompt_template | chat_model | output_parser
    user_input = {"image":final_image_data}

    st.header("Description and Context of Image:")
    st.write(chain.invoke(user_input))

st.subheader("Select the AI")
button_click_1 = st.button("AI for Safe Navigation")
button_click_2 = st.button("AI For Daily Tasks")


if button_click_1 == True:
    object_and_obstacle_detection_for_safe_navigation(image_data)

if button_click_2 == True:
    personalized_assistance_for_daily_tasks(image_data)











