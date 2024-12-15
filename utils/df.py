import os
from deepface import DeepFace
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
load_dotenv()
IMAGE_PATH=os.getenv("IMAGE_PATH")
API_KEY = os.getenv("API_KEY")

objs = DeepFace.analyze(
    img_path=IMAGE_PATH,
    actions=["emotion", "age", "gender", "race"],
    enforce_detection=False,
)
print(objs)

chat = ChatGroq(temperature=0,
                            groq_api_key="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", 
                            model_name="llama-3.1-70b-versatile")

system = "You are a helpful assistant and does not generate anything else other than face description"
human = "Generate a natural language description of the face appearance from {text} "
prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

chain = prompt | chat
res = chain.invoke({"text": f"These are facial attributes detected from an image: {objs}"})
print(res.content)