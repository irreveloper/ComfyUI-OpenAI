import io
import os
import openai
import base64
import numpy as np
from PIL import Image
from dotenv import load_dotenv
import json

load_dotenv()
MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "chatgpt-4o-latest",
    "gpt-4-turbo"
]

class OpenAICaptionImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_in" : ("IMAGE", {}),
                "model": (MODELS, ),
                "system_prompt": ("STRING", {"multiline": True,"default": "You are a helpful assistant."}),
                "caption_prompt": ("STRING", {"multiline": True,"default": "What's in this image?"}),
                "max_tokens": ("INT", {"default": 300}),
                "temperature": ("FLOAT", {"default": 0.5}),
                "use_custom_response_format" : (["enabled","disabled"], {"default": "disabled"}),  # Added this line 
                "custom_response_format": ("STRING", {"multiline": True,"default": """ {
                // See /docs/guides/structured-outputs
                type: "json_schema",
                json_schema: {
                    name: "email_schema",
                    schema: {
                        type: "object",
                        properties: {
                            email: {
                                description: "The email address that appears in the input",
                                type: "string"
                            }
                        },
                        additionalProperties: false
                    }
                }
                }"""}),  # Added this line

            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text_out",)
    CATEGORY = "openai"
    FUNCTION = "caption"

    def caption(self, image_in, model, system_prompt, caption_prompt, max_tokens, temperature,use_custom_response_format, custom_response_format):
        # image to base64, image is bwhc tensor
        
        use_custom_response_format = use_custom_response_format == "enabled"

        # Convert tensor to PIL Image
        pil_image = Image.fromarray(np.clip(255. * image_in.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
        
        # Convert PIL Image to base64
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Set up OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        client = openai.OpenAI(api_key=api_key)

            # Prepare API call parameters
        api_params = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": caption_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_str}"}}
                    ],
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        # Add response_format if use_custom_response_format is True
        if use_custom_response_format:
            #convert custom_response_format to JSON
            custom_response_format = json.loads(custom_response_format)
            api_params["response_format"] = custom_response_format

        # Make API call to OpenAI
        response = client.chat.completions.create(**api_params)
        
        if response.choices[0].message.content is None:
            raise ValueError("No content in response")

        # Extract and return the caption
        caption = response.choices[0].message.content.strip()
        return (caption,)
