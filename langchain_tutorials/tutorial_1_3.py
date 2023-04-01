import argparse
import os
from langchain.agents import load_tools, initialize_agent
from langchain.llms import OpenAI
from misc.utils import retrieve_local_api_keys
from colorama import init, Fore


def agent_script(query,
                 model_name="text-davinci-003",
                 temperature=0.0,
                 agent_type="zero-shot-react-description",
                 verbose=True):

    # Load the API keys from the .env file
    retrieve_local_api_keys()

    # Initialize the LLM with the specified parameters
    llm = OpenAI(model_name=model_name, temperature=temperature)

    # Load the required tools
    tools = load_tools(["google-serper", "llm-math"], llm=llm)

    # Initialize the agent
    agent = initialize_agent(tools, llm, agent=agent_type, verbose=verbose)

    # Run the agent with the given query
    response = agent.run(query)

    # Print the response
    print(f"\n{Fore.BLUE}Query    :{Fore.RESET} '{query}'")
    print(f"{Fore.GREEN}Response : {Fore.RESET} '{response.strip()}'\n")


def main():
    parser = argparse.ArgumentParser(description="Run a query using a LangChain agent")
    parser.add_argument("query", type=str, help="The query to be run by the agent")
    parser.add_argument("-n", "--model_name", type=str, default="text-davinci-003", help="Model name for the LLM")
    parser.add_argument("-t", "--temperature", type=float, default=0.0, help="Temperature for randomness in output")
    parser.add_argument("-a", "--agent_type", type=str, default="zero-shot-react-description",
                        help="Type of agent to use")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    # Run the agent script with the provided arguments
    agent_script(**args.__dict__)


# Example Usage:
# python3 -m langchain_tutorials.tutorial_1_3 \
# "List all AI models that were released in 2023 as well as who released them and the date they were released:" -v
if __name__ == "__main__":
    main()
