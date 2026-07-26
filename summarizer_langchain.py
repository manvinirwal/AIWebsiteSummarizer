  
# summarizer, rebuilt the LangChain way.
# Run:  python summarizer_langchain.py
import os
from langchain_openai import ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from scraper import fetch_website_contents   # reuse Class 1's scraper

load_dotenv()

# ① a reusable prompt with a {website} blank
prompt = ChatPromptTemplate.from_template(
    "Give a short, friendly summary of this website:\n\n{website}"
)


# ② the same model from Class 1, wrapped for LangChain
# model = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

#ChatOpenAI is specifically designed for OpenAI-compatible chat endpoints.
#Since Groq exposes an OpenAI-compatible API, you can point ChatOpenAI to Groq instead of OpenAI.
model = ChatOpenAI(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
    temperature=0.3
)

# ow we had to learn groq specifics to get the model to work, but now we can use it just like any other OpenAI model in LangChain.
#from langchain_groq import ChatGroq

#model = ChatGroq(
 #   model="llama-3.3-70b-versatile",
 #   api_key=os.getenv("GROQ_API_KEY"),
  #  temperature=0.3
#)
# ③ the package-opener: hands back plain text
parser = StrOutputParser()

# ④ snap them together  (| means "then")
chain = prompt | model | parser


def summarize(url):
    # ⑤ run it; the dict fills the {website} blank by name
    return chain.invoke({"website": fetch_website_contents(url)})


if __name__ == "__main__":
    print(summarize("https://anthropic.com"))
