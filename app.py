import chainlit as cl
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Load image model (BLIP for captioning)
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# Load LLM (using Ollama)
llm = Ollama(model="llama3")  # Make sure llama3 is running with: `ollama run llama3`

# Prompt Template for Diagnosis
prompt = PromptTemplate(
    input_variables=["caption", "question"],
    template="""
You are an expert agriculture assistant. A farmer has shown you this plant image with the following description: "{caption}"

Their question is: "{question}"

Based on this, provide a helpful and detailed response regarding any issues or recommendations.
"""
)

# LangChain Chain
qa_chain = LLMChain(prompt=prompt, llm=llm)

# Function to process image and generate caption
def caption_image(img: Image.Image) -> str:
    img = img.convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(device)
    out = blip_model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# Chainlit frontend
@cl.on_chat_start
async def start():
    await cl.Message("👩‍🌾 Welcome, farmer! Upload a photo of your crop and ask your question.").send()

@cl.on_message
async def handle_message(message: cl.Message):
    files = None
    if message.elements:
        files = [e for e in message.elements if isinstance(e, cl.File)]

    question = message.content.strip()
    caption = "No image provided."

    # Process image if attached
    if files:
        image_file = files[0]
        image_path = await image_file.get_path()
        img = Image.open(image_path)
        caption = caption_image(img)

    # Use LangChain to generate response
    result = qa_chain.run(caption=caption, question=question)

    # Send back result
    await cl.Message(
        content=f"🧠 **Image Analysis:** {caption}\n\n💡 **AI Advice:** {result}"
    ).send()
