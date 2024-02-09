import google.generativeai as genai
import gradio as gr
from dotenv import load_dotenv
from pathlib import Path
import os

# 
load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

generative_config = {
    "temperature": 0.8,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 4096
}

safety_settings = [
    {"category": f"HARM_CATEGORY_{category}", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
     for category in ["HARASSMENT", "HATE_SPEECH", "SEXUALLY_EXPLICIT", "DANGEROUS_CONTENT"]
]

model = genai.GenerativeModel(
    model_name="gemini-pro-vision",
    generation_config=generative_config,
    safety_settings=safety_settings
)

def read_image(filepath):
    imagepath = Path(filepath) # read path
    if not imagepath.exists():
        raise FileNotFoundError(f"Could not find image: {imagepath}")
    return {
        "mime_type": "image/jpeg",
        "data": imagepath.read_bytes()
    }

def generate_gemini_response(prompt, imagepath):
    image_data = read_image(imagepath)
    response = model.generate_content([prompt, image_data])
    return response.text


input_prompt = """
Analyze the image and demonstrate your culinary expertise by providing the following for each dish:

## Ingredients:
List all the ingredients displayed in the picture, categorized for clarity (e.g., protein, vegetables, aromatics, spices, sauces).
For complex dishes, identify any hidden or implied ingredients based on your knowledge and experience.
Specify quantities using standard units (e.g., grams, cups, tablespoons) whenever possible.

## Nutrition Information:
Estimate the nutritional value per serving, including calories, macronutrients (carbohydrates, protein, fat), and any prominent vitamins or minerals.
Base your estimation on standard serving sizes and typical ingredient profiles.
Clearly state any assumptions or limitations in your nutritional analysis.

## Cooking Steps:
Provide a clear and concise step-by-step guide to recreate the dish.
Include cooking methods, temperatures, and timings when visible or inferable from the image.
Consider potential variations or alternative techniques based on your expertise.
Aim for a format suitable for home cooks, balancing detail with conciseness.


## Remember:
You are a Vietnamese master chef, showcasing your knowledge and understanding of diverse culinary styles and techniques.
Be creative and resourceful, inferring information from the image and drawing upon your vast culinary experience.
Present your findings in a clear, concise, and informative manner, suitable for an audience of aspiring cooks.
Let your culinary prowess shine! Analyze each dish image and provide a comprehensive breakdown of its ingredients, cooking steps, and nutritional value.

Important. Answer everything in Vietnamese
"""

def process_uploaded_files(files):
    filepath = files[0].name if files else None
    response = generate_gemini_response(input_prompt, filepath)
    return filepath, response


## code UI
with gr.Blocks() as demo:
    file_output = gr.Textbox()
    image_output = gr.Image()
    combined_output = [image_output, file_output]

    upload_button = gr.UploadButton(
        "click to Upload an Image", 
        file_types=["Image"],
        file_count="multiple"
    )

    upload_button.upload(process_uploaded_files, upload_button, combined_output)

demo.launch(debug=True)