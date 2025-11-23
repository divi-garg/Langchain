from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

load_dotenv()

def main():
    print("Hello from LangChain + Ollama!")

    information = """
    Bharat Chauhan is a Delhi-based singer-songwriter known for his melancholically psychedelic music.
    Chauhan writes songs in Hindi, Urdu and Punjabi. His music generally has a heavy emphasis on the lyrics.
    His latest album 'Qurbat' is a mix of acoustic, heavy/trippy, rock, and psychedelic music.
    """

    summary_template = """
    Given the information: {information} about the person, create:
    1. A short summary
    2. Two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
    )

    # 🔥 Local Llama3 through Ollama
    llm = ChatOllama(
        model="llama3",  # or "llama3.1" if you downloaded that version later
        temperature=0
    )

    chain = summary_prompt_template | llm

    response = chain.invoke({"information": information})
    print(response.content)

if __name__ == "__main__":
    main()
