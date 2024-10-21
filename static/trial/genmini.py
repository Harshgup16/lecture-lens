import os
import google.generativeai as genai
import markdown

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

response = chat_session.send_message("give me short summary for story in md format - There once was a boy who grew bored while watching over the village sheep. He wanted to make things more exciting. So, he yelled out that he saw a wolf chasing the sheep. All the villagers came running to drive the wolf away. However, they saw no wolf. The boy was amused, but the villagers were not. They told him not to do it again. Shortly after, he repeated this antic. The villagers came running again, only to find that he was lying. Later that day, the boy really sees a wolf sneaking amongst the flock. He jumped up and called out for help. But no one came this time because they thought he was still joking around. At sunset, the villagers looked for the boy. He had not returned with their sheep. They found him crying. He told them that there really was a wolf, and the entire flock was gone. An old man came to comfort him and told him that nobody would believe a liar even when they are being honest.")

print(response.text)

mdfile=open('mdfile.md','w')

mdfile.write(f"""{response.text}""")

mdfile.close()

# def markdown_to_html(input_file,output_file):
#     try:
#         with open(input_file, 'r') as md_file:
#             text = md_file.read()
#             html_content = markdown.markdown(text)
#             with open(output_file, 'w') as html_file:
#                 html_file.write(html_content)
            

#     except FileNotFoundError:
#         print(f"Error: Markdown file '{input_file}' not found.")

# if __name__=='__main__':
#     input_file ='mdfile.md'
#     output_file = 'mdfile.html'
#     markdown_to_html(input_file, output_file)


# import os
# from langchain_google_genai import GoogleGenerativeAI
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough

# # Set your Google API key
# os.environ["GOOGLE_API_KEY"] = "AIzaSyDXg5GinUuT4axifXMgNvIWkvtgs7NuepE"  # Replace with your actual Gemini API key

# def get_text_summary(text):
#     """
#     Gets a summary of the given text using Google's Gemini API and LangChain.

#     Args:
#         text (str): The text to summarize.

#     Returns:
#         str: The summary of the text.
#     """

#     # Check if API key is set
#     if "GOOGLE_API_KEY" not in os.environ:
#         raise ValueError("Please set your Google API key as an environment variable 'GOOGLE_API_KEY'")

#     # Create a Gemini LLM instance
#     try:
#         llm = GoogleGenerativeAI(model="gemini-pro", temperature=0.5)  # Adjust temperature as needed
#     except Exception as e:
#         print(f"Error creating LLM instance: {e}")
#         return None  # Handle errors gracefully

#     # Define a prompt template
#     prompt_template = PromptTemplate.from_template("Summarize the following text:\n{text}")

#     # Create a chain using the new method
#     chain = (
#         {"text": RunnablePassthrough()}
#         | prompt_template
#         | llm
#         | StrOutputParser()
#     )

#     # Run the chain to generate the summary
#     try:
#         summary = chain.invoke(text)
#         return summary
#     except Exception as e:
#         print(f"Error generating summary: {e}")
#         return None

# # Example usage
# text_to_summarize = "here once was a boy who grew bored while watching over the village sheep. He wanted to make things more exciting. So, he yelled out that he saw a wolf chasing the sheep. All the villagers came running to drive the wolf away. However, they saw no wolf. The boy was amused, but the villagers were not. They told him not to do it again. Shortly after, he repeated this antic. The villagers came running again, only to find that he was lying. Later that day, the boy really sees a wolf sneaking amongst the flock. He jumped up and called out for help. But no one came this time because they thought he was still joking around. At sunset, the villagers looked for the boy. He had not returned with their sheep. They found him crying. He told them that there really was a wolf, and the entire flock was gone. An old man came to comfort him and told him that nobody would believe a liar even when they are being honest"
# summary = get_text_summary(text_to_summarize)
# print(summary)