
import chainlit as cl
from PIL import Image, ImageOps
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from datetime import datetime
import os
from typing import Optional, Dict, List
import json

# -----------------------------
# 🛠️ Configuration
# -----------------------------
class Config:
    CAPTION_MODEL = "Salesforce/blip-image-captioning-large"
    LLM_MODEL = "llama3"
    HISTORY_FILE = "conversation_history.json"
    MAX_HISTORY = 5
    ALLOWED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/webp"]
    SAFETY_CHECK = True

config = Config()

# -----------------------------
# 🔍 Image Processor
# -----------------------------
class ImageProcessor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = BlipProcessor.from_pretrained(config.CAPTION_MODEL)
        self.model = BlipForConditionalGeneration.from_pretrained(config.CAPTION_MODEL).to(self.device)
        
    def caption_image(self, img: Image.Image, context: str = None) -> str:
        try:
            img = ImageOps.exif_transpose(img).convert("RGB")
            if context:
                inputs = self.processor(images=img, text=context, return_tensors="pt").to(self.device)
            else:
                inputs = self.processor(images=img, return_tensors="pt").to(self.device)
            out = self.model.generate(**inputs, max_new_tokens=100)
            return self.processor.decode(out[0], skip_special_tokens=True)
        except Exception as e:
            return f"Error processing image: {str(e)}"

# -----------------------------
# 🧠 Agriculture Assistant
# -----------------------------
class AgricultureAssistant:
    def __init__(self):
        self.llm = Ollama(
            model=config.LLM_MODEL,
            temperature=0.7,
            top_p=0.9,
            repeat_penalty=1.1
        )
        self.memory = ConversationBufferMemory(memory_key="chat_history", input_key="question")
        self.setup_prompts()
        
    def setup_prompts(self):
        self.diagnosis_prompt = PromptTemplate(
            input_variables=["caption", "question", "chat_history", "current_date"],
            template="""
**Agriculture Expert System** (Date: {current_date})

**Image Context**: {caption}

**Conversation History**:
{chat_history}

**Farmer's Question**: {question}

As an expert agriculture assistant with 20 years of field experience, provide:
1. Detailed diagnosis based on visual symptoms
2. Likely causes (ranked by probability)
3. Recommended treatments (organic and conventional options)
4. Prevention strategies
5. When to consult a human expert

Structure your response with clear headings and bullet points.
"""
        )
        
        self.general_prompt = PromptTemplate(
            input_variables=["question", "chat_history", "current_date"],
            template="""
**Agriculture Expert System** (Date: {current_date})

**Conversation History**:
{chat_history}

**Question**: {question}

Provide comprehensive, science-based agricultural advice considering:
- Current best practices
- Regional variations
- Sustainable approaches
- Cost-effectiveness

Include relevant examples and statistics when possible.
"""
        )
        
        self.diagnosis_chain = LLMChain(prompt=self.diagnosis_prompt, llm=self.llm, memory=self.memory)
        self.general_chain = LLMChain(prompt=self.general_prompt, llm=self.llm, memory=self.memory)
    
    def get_current_date(self):
        return datetime.now().strftime("%Y-%m-%d")
    
    async def generate_response(self, caption: Optional[str], question: str) -> str:
        current_date = self.get_current_date()
        if caption and caption != "No image was provided.":
            return await self.diagnosis_chain.arun(
                caption=caption,
                question=question,
                current_date=current_date
            )
        else:
            return await self.general_chain.arun(
                question=question,
                current_date=current_date
            )

# -----------------------------
# 📊 History Manager
# -----------------------------
class HistoryManager:
    def __init__(self):
        self.history_file = config.HISTORY_FILE
        
    def load_history(self) -> List[Dict]:
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            return []
        except Exception:
            return []
            
    def save_history(self, history: List[Dict]):
        try:
            if len(history) > config.MAX_HISTORY:
                history = history[-config.MAX_HISTORY:]
            with open(self.history_file, 'w') as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            print(f"Error saving history: {e}")

# -----------------------------
# 💬 Chainlit Events
# -----------------------------
@cl.on_chat_start
async def start_chat():
    cl.user_session.set("img_processor", ImageProcessor())
    cl.user_session.set("assistant", AgricultureAssistant())
    cl.user_session.set("history_manager", HistoryManager())
    
    history = cl.user_session.get("history_manager").load_history()
    if history:
        last_msg = history[-1]
        await cl.Message(
            content=f"👋 Welcome back! Last time we discussed: {last_msg.get('question', '')}"
        ).send()
    
    actions = [
        cl.Action(name="upload_image", value="upload", label="📷 Upload Plant Image"),
        cl.Action(name="common_issues", value="common_issues", label="🌱 Common Plant Problems"),
        cl.Action(name="pest_guide", value="pest_guide", label="🐛 Pest Identification"),
        cl.Action(name="reset", value="reset", label="🔄 Reset Conversation"),
        cl.Action(name="clear_history", value="clear_history", label="🗑️ Clear History")
    ]
    
    await cl.Message(
        content="""👩‍🌾 **Welcome to AgriExpert Assistant!**

I can help with:
- Plant disease diagnosis
- Pest identification
- Growing advice
- Soil management
- And much more!

How can I assist you today?""",
        actions=actions
    ).send()

@cl.action_callback("upload_image")
async def on_action_upload(action: cl.Action):
    await cl.Message(content="Please upload an image of your plant for analysis.").send()

@cl.action_callback("common_issues")
async def on_action_issues(action: cl.Action):
    assistant = cl.user_session.get("assistant")
    response = await assistant.generate_response(
        caption=None,
        question="List the 5 most common plant problems with brief identification tips"
    )
    await cl.Message(content=response).send()

@cl.action_callback("pest_guide")
async def on_action_pests(action: cl.Action):
    assistant = cl.user_session.get("assistant")
    response = await assistant.generate_response(
        caption=None,
        question="Provide a quick reference guide for common garden pests including visual identification markers and organic control methods"
    )
    await cl.Message(content=response).send()

@cl.action_callback("reset")
async def on_action_reset(action: cl.Action):
    cl.user_session.set("assistant", AgricultureAssistant())
    await cl.Message(content="Conversation has been reset. How can I help you?").send()

@cl.action_callback("clear_history")
async def on_clear_history(action: cl.Action):
    history_manager = cl.user_session.get("history_manager")
    history_manager.save_history([])
    await cl.Message(content="📂 Conversation history cleared.").send()

@cl.on_message
async def handle_message(message: cl.Message):
    img_processor = cl.user_session.get("img_processor")
    assistant = cl.user_session.get("assistant")
    history_manager = cl.user_session.get("history_manager")
    
    question = message.content.strip()
    caption = "No image was provided."
    image_elements = []
    
    if message.elements:
        for element in message.elements:
            if (isinstance(element, cl.Image) or 
                (isinstance(element, cl.File) and getattr(element, 'mime', '') in config.ALLOWED_IMAGE_TYPES)):
                try:
                    image_path = await element.get_path()
                    img = Image.open(image_path)

                    if config.SAFETY_CHECK and not is_plant_image(img):
                        await cl.Message(content="⚠️ The image doesn't appear to be of a plant. Please upload a clear image of your plant for diagnosis.").send()
                        return

                    initial_caption = img_processor.caption_image(img)
                    refined_caption = img_processor.caption_image(img, f"Describe this plant in detail focusing on any visible issues relevant to: {question}")
                    
                    caption = f"Initial observation: {initial_caption}\nDetailed analysis: {refined_caption}"

                    image_elements.append(cl.Image(
                        name="plant_image",
                        display="inline",
                        path=image_path,
                        size="large"
                    ))
                except Exception as e:
                    caption = f"Error processing image: {str(e)}"
    
    response = await assistant.generate_response(caption, question)

    history = history_manager.load_history()
    history.append({
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "caption": caption,
        "response": response[:500] + "..." if len(response) > 500 else response
    })
    history_manager.save_history(history)
    
    response_message = cl.Message(content="", elements=image_elements)
    if caption != "No image was provided.":
        response_message.content += f"## 🖼️ Image Analysis\n{caption}\n\n"
    response_message.content += f"## 🌱 Expert Advice\n{response}"
    response_message.actions = [
        cl.Action(name="helpful", value="helpful", label="👍 Helpful"),
        cl.Action(name="needs_improvement", value="needs_improvement", label="👎 Needs Improvement"),
        cl.Action(name="follow_up", value="follow_up", label="💬 Ask Follow-up")
    ]
    await response_message.send()

@cl.action_callback("helpful")
async def on_helpful(action: cl.Action):
    await cl.Message(content="Thank you for your feedback! I'm glad I could help.").send()

@cl.action_callback("needs_improvement")
async def on_needs_improvement(action: cl.Action):
    await cl.Message(content="I appreciate your feedback. Could you please specify what I could improve?").send()

@cl.action_callback("follow_up")
async def on_follow_up(action: cl.Action):
    await cl.Message(content="Please ask your follow-up question about this plant.").send()

def is_plant_image(img: Image.Image) -> bool:
    if img.mode == 'RGB':
        pixels = img.getdata()
        green_pixels = sum(1 for r, g, b in pixels if g > r and g > b)
        return green_pixels / len(pixels) > 0.3
    return True

# -----------------------------
# ✅ Main Entry Point
# -----------------------------
if __name__ == "__main__":
    from chainlit.cli.main import run_chainlit
    run_chainlit(path=__file__)
