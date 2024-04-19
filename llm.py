import os
from dotenv  import load_dotenv

from pydantic import BaseModel, Field

import pandas as pd

from langchain_openai import ChatOpenAI
from langchain.chains import create_qa_with_structure_chain
from langchain.prompts.prompt import PromptTemplate

# CRIAR UM .ENV COM A CHAVE OPENAI_APIKEY=<sua_chave>
load_dotenv()
openai_key = os.environ.get('OPENAI_APIKEY', None)

class ResponseSchema(BaseModel):
    """Resposta para a pergunta: Qual a chance deste passageiro sobreviver?"""

    survived: int = Field(..., description="1 para sim e 0 para nao")
    explanation: str = Field(..., description="Uma breve explicação do motivo da resposta")


template = """
    Você foi encarregado de desenvolver um modelo de inferência para prever se um passageiro do Titanic sobreviveria ou não com base em seus atributos pessoais.
    Esta é uma tarefa importante que requer a capacidade de analisar informações contextuais e tomar decisões com base em padrões observados nos dados.
    
    Leve em consideração os seguintes dados históricos para a tomada de decisão:
    {context}

    Agora, suponha que exista no Titanic um novo passageiro com os seguintes atributos:
    Pclass: {Pclass}
    Name: {Name}
    Sex: {Sex}
    Age: {Age}
    SibSp: {SibSp}
    Parch: {Parch}
    Ticket: {Ticket}
    Fare: {Fare}
    Cabin: {Cabin}
    Embarked: {Embarked}

    Qual a chance deste passageiro sobreviver?
"""

prompt = PromptTemplate(input_variables=["history", "input"], template=template)

llm = ChatOpenAI(
    temperature=0,
    openai_api_key=openai_key,
    model_name='gpt-3.5-turbo-1106'
)

chain = create_qa_with_structure_chain(
    prompt=prompt,
    llm=llm,
    schema=ResponseSchema,
    output_parser="pydantic"
)

train_df = pd.read_csv('train.csv')

test_df = pd.read_csv('test.csv')
row = test_df.iloc[4]

result = chain.predict(
    context=train_df.sample(n=20).to_string(),
    Pclass=row['Pclass'],
    Name=row['Name'],
    Sex=row['Sex'],
    Age=row['Age'],
    SibSp=row['SibSp'],
    Parch=row['Parch'],
    Ticket=row['Ticket'],
    Fare=row['Fare'],
    Cabin=row['Cabin'],
    Embarked=row['Embarked']
)

print("======================================================")
print(row)
print('-----------')
print(result.survived)
print(result.explanation)
print("======================================================")
