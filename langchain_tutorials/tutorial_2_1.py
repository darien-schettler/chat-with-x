import argparse
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from misc.utils import retrieve_local_api_keys, print_colored_output


def chat_completion_script(model_name="gpt-3.5-turbo", temperature=0, input_clr="blue", response_clr="green", use_color=True):
    """ Generate a chat completion using LangChain ChatOpenAI

    Args:
        model_name (str): Model name for the LLM (default: gpt-3.5-turbo) to be used in the ChatOpenAI instance
        temperature (float): Temperature for randomness in output. Higher values will result in more random output
        input_clr (str): Color for user input text
        response_clr (str): Color for AI response text

    Returns:
        None; prints the chat completion to the console and interacts with the user
    """

    # Load the API keys from the .env file
    retrieve_local_api_keys()

    # Initialize the ChatOpenAI instance with the specified parameters
    chat = ChatOpenAI(model_name=model_name, temperature=temperature)

    # Continue the conversation loop until the user decides to exit
    while True:
        input_text = input("\nEnter your message (or press enter to exit):\n")
        if not input_text: break

        messages = [
            SystemMessage(content="You are a helpful assistant that translates English to French."),
            HumanMessage(content=input_text)
        ]
        response = chat(messages)

        print_colored_output(input_text, response.content, input_clr, response_clr, full_color=use_color)


def main():
    parser = argparse.ArgumentParser(description="Generate a chat completion using LangChain ChatOpenAI")
    parser.add_argument("-n", "--model_name", type=str, default="gpt-3.5-turbo", help="Model name for the ChatLLM")
    parser.add_argument("-t", "--temperature", type=float, default=0, help="Temperature for randomness in output")
    parser.add_argument("--input_clr", type=str, default="blue", help="Color for user input text")
    parser.add_argument("--response_clr", type=str, default="green", help="Color for AI response text")
    parser.add_argument("-c", "--use_color", action="store_false", help="Color only to titles")
    args = parser.parse_args()

    # Generate the chat completion based on the user inputs
    chat_completion_script(**args.__dict__)


if __name__ == "__main__":
    main()
