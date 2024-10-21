# import os
import google.generativeai as genai

genai.configure(api_key="AIzaSyDn4MtF99w6vIP2lafquqUGSET63pHg6B0")

# Create the model
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-1.5-flash",
  generation_config=generation_config,
  #safety_settings = Adjust safety settings
  #See https://ai.google.dev/gemini-api/docs/safety-settings
)

chat_session = model.start_chat(
  history=[
  ]
)

response = chat_session.send_message("give me summary for story - There once was a boy who grew bored while watching over the village sheep. He wanted to make things more exciting. So, he yelled out that he saw a wolf chasing the sheep. All the villagers came running to drive the wolf away. However, they saw no wolf. The boy was amused, but the villagers were not. They told him not to do it again. Shortly after, he repeated this antic. The villagers came running again, only to find that he was lying. Later that day, the boy really sees a wolf sneaking amongst the flock. He jumped up and called out for help. But no one came this time because they thought he was still joking around. At sunset, the villagers looked for the boy. He had not returned with their sheep. They found him crying. He told them that there really was a wolf, and the entire flock was gone. An old man came to comfort him and told him that nobody would believe a liar even when they are being honest.")

print(response.text)