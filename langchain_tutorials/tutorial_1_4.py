import argparse
from langchain import OpenAI, ConversationChain
from misc.utils import retrieve_local_api_keys
from colorama import init, Fore


def conversation_script(conversation_inputs, model_name="text-davinci-003", temperature=0, verbose=True):
    # Load the API keys from the .env file
    retrieve_local_api_keys()

    # Initialize the LLM with the specified parameters
    llm = OpenAI(model_name=model_name, temperature=temperature)

    # Initialize the ConversationChain
    conversation = ConversationChain(llm=llm, verbose=verbose)

    # Iterate over conversation inputs and run the ConversationChain
    for i, input_text in enumerate(conversation_inputs):
        response = conversation.predict(input=input_text)
        print(f"{Fore.BLUE}Input {i + 1:2d}         :{Fore.RESET} '{input_text}'")
        print(f"{Fore.GREEN}Response {i + 1:2d}      :{Fore.RESET} '{response.strip()}'\n")


def main():
    parser = argparse.ArgumentParser(description="Run a conversation using a LangChain ConversationChain")
    parser.add_argument("conversation_inputs", nargs="+", type=str,
                        help="List of conversation inputs to be processed by the ConversationChain")
    parser.add_argument("-n", "--model_name", type=str, default="text-davinci-003", help="Model name for the LLM")
    parser.add_argument("-t", "--temperature", type=float, default=0, help="Temperature for randomness in output")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    # Run the conversation script with the provided arguments
    conversation_script(args.conversation_inputs, args.model_name, args.temperature, args.verbose)


if __name__ == "__main__":
    main()
