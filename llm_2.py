import os
from dotenv  import load_dotenv

from pydantic import BaseModel, Field

import pandas as pd

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage


load_dotenv()
openai_key = os.environ.get('OPENAI_APIKEY', None)

train_df = pd.read_csv('train.csv')

template = f"""
    Você foi encarregado de desenvolver um modelo de inferência para prever se um passageiro do Titanic sobreviveria ou não com base em seus atributos pessoais.
    Esta é uma tarefa importante que requer a capacidade de analisar informações contextuais e tomar decisões com base em padrões observados nos dados.
    
    Leve em consideração os seguintes dados históricos para responder:
    {train_df.sample(n=20).to_string()}
"""

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            template,
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

llm = ChatOpenAI(
    temperature=0,
    openai_api_key=openai_key,
    model_name='gpt-3.5-turbo-1106'
)

chain = prompt | llm

messages = []
while True:
    try:
        user_input = input("Digite sua pergunta (ou 'exit' para parar): ")

        if user_input.lower() == 'exit':
            print("Exiting the loop.")
            break

        messages.append(HumanMessage(content=user_input))
        
        result = chain.invoke(messages)
        print(result.content)
        messages.append(AIMessage(content=result.content))
    except Exception as ex:
        print('Erro. Tente novamente', str(ex))
