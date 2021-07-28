import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering, BertForQuestionAnswering,BertTokenizer
import requests
from bs4 import BeautifulSoup
import torch
import validators

@st.cache(allow_output_mutation=True)
def load_qa_model():
	model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
	model = BertForQuestionAnswering.from_pretrained(model_name)
	tokenizer = BertTokenizer.from_pretrained(model_name)
	model = pipeline('question-answering', model=model, tokenizer=tokenizer)
	return model

qa = load_qa_model()
st.title("Questions Answering System")
context = st.text_area('Please paste your article or the url for the articlr:', height=30)
# st.title("OR")
# URL = st.text_input("Please paste the link to the article :")
question = st.text_input("Questions from this article?")
button = st.button("Get me Answers")


if validators.url(context):
	page = requests.get(context)
	soup = BeautifulSoup(page.content, "html.parser")
	paragraph_elements = soup.find_all("p")
	par = ''''''
	for i in paragraph_elements:
  		par = par + i.text
	context = par



with st.spinner("Discovering Answers.."):
    if button and context:
        answers = qa(question=question, context=context)
        st.write(answers['answer'])