import streamlit as st
import os
from transformers import pipeline

import requests

@st.cache
def query(payload, model_id, api_token):
	headers = {"Authorization": f"Bearer {api_token}"}
	API_URL = f"https://api-inference.huggingface.co/models/{model_id}"
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()


api_token = os.getenv("api_token")
st.header("Ways to Improve Your Conversational Agents using Language Models 🤗")
st.write("There are many ways to improve your conversational agents using language models. In this blog post, I will walk you through a couple of tricks that will improve your conversational agent. 👾") 
st.write("There are multiple ways of building conversational agents, you can build intent-action based chatbots for automating processes or build language-model based chatbots like [DialoGPT](https://huggingface.co/docs/transformers/model_doc/dialogpt). This blog post is for intent-action based agents, but in the last part, I'm talking about a hybrid agent that switches between our chatbot and a dialogue language model.")

st.subheader("Data Augmentation with Generative Models ✨")
st.write("There are cases where you will not be allowed to keep data, you will have to start from scratch or you will have very little amount of data. Let's see how we can solve it with a use case.")
st.write("Imagine you're making a chatbot that will answer very general questions about emergency situations at home.")
st.write("If you have very little amount of data, you could actually augment it through language models. There are regex based tools you can use but they tend to create bias due to repetitive patterns, so it's better to use language models for this case. A good model to use is a generative model (namely, [T5](https://huggingface.co/docs/transformers/model_doc/t5) here) fine-tuned on [Quora Question Pairs dataset](https://www.tensorflow.org/datasets/catalog/glue?hl=en#glueqqp). This dataset consists of question pairs that are paraphrase of one another, and T5 can generate a paraphrased question given a source question. There's a similar dataset called [MRPC](https://www.tensorflow.org/datasets/catalog/glue?hl=en#gluemrpc) that assesses if one sentence is a paraphrase of another, you can choose between one of them.")
st.write("Try it yourself here 👇🏻")
default_value_gen = "How can I put out grease fire?"
sent = st.text_area(label = "Input", value = default_value_gen, height = 10)
outputs = query(payload = sent, model_id = "mrm8488/t5-small-finetuned-quora-for-paraphrasing", api_token = api_token)
st.write("Paraphrased Example:")
try:
	st.write(outputs[0]["generated_text"])
except:
	st.write("Inference API loads model on demand, please wait for 10 secs and try again 🤗 ")
st.subheader("Multilingual Models using Translation Models 🙌🏼")
st.write("Scaling your chatbot across different languages is expensive and cumbersome. There are couple of ways on how to make your chatbot speak a different language. You can either translate the intent classification data and responses and train another model and deploy it,, or you can put translation models at two ends. There are advantages and disadvantages in both approaches. For the first one, you can assess the performance of the model and hand your responses to a native speaker to have more control over what your bot says, but it requires more resources compared to second one. For the second one, assume that you're making a chatbot that is in English and want to have another language, say, German. You need two models, from German to English and from English to German.")
st.image("./Translation.png")
st.write("Your English intent classification model will be between these two models, your German to English model will translate the input to English and the output will go through the intent classification model, which will classify intent and select appropriate response (which is currently in English). The response will be translated back to German, which you can do in advance and do proofreading with a native speaker or directly pass it to a from English to German language model. For this use case, I highly recommend specific translation models instead of using sequence-to-sequence multilingual models like T5. ")



model_id = "Helsinki-NLP/opus-mt-en-fr"

default_value_tr = "How are you?"
tr_input = st.text_area(label = "Input in English", value = default_value_tr, height = 5)
tr = query(tr_input, model_id, api_token)
st.write("Translated Example:")
try:
	st.write(tr[0]["translation_text"])
except:
	st.write("Inference API loads model on demand, please wait for 10 secs and try again 🤗 ")
st.write("You can check out this [link](https://huggingface.co/models?pipeline_tag=translation&sort=downloads&search=helsinki-nlp) for available translation models.")


st.subheader("Easy Information Retrieval 🧩")
st.write("If you're making a chatbot that needs to provide information to user, you can take user's query and search for the answer in the documents you have, using question answering models. Look at the example and try it yourself here 👇🏻")


model_id_q = "distilbert-base-cased-distilled-squad"
question = st.text_area(label = "Question", value = "What does transformers do?", height = 5)
context = st.text_area(label = "Context", value = "🤗 Transformers provides thousands of pretrained models to perform tasks on different modalities such as text, vision, and audio.")
data = {"inputs": {"question": question, "context": context}}
output_answer = query(payload = data, model_id = model_id_q, api_token = api_token)
st.write("Answer:")
try:
	st.write(output_answer["answer"])
except:
	st.write("Inference API loads model on demand, please wait for 10 secs and try again 🤗 ")

st.subheader("Add Characters to Your Conversational Agent 🧙🏻🦹🏻") 
st.write("When trained, language models like GPT-2 or DialoGPT is capable of talking like any character you want. If you have a friend-like chatbot (instead of a chatbot built for RPA) you can give your users options to talk to their favorite character. There are couple of ways of doing this, you can either fine-tune DialoGPT with sequences of conversation turns, maybe movie dialogues, or infer with a large model like GPT-J. Note that these models might have biases and you will not have any control over output, unless you make an additional effort to filter it.")
st.write("You can see an [example](https://huggingface.co/docs/transformers/model_doc/dialogpt) of a chatbot that talks like Gandalf, that is done simply by sending a request to GPT-J through Inference API.")

st.write("I've written the inferences in this blog post with only three lines of code (🤯), using [pipelines](https://huggingface.co/docs/transformers/main_classes/pipelines) and for translation example, [Inference API](https://huggingface.co/inference-api), which you can use for building your chatbots as well! Check out the code of the post [here](https://huggingface.co/spaces/merve/chatbot-blog/blob/main/app.py) on how you can do it too! 🤗 ")