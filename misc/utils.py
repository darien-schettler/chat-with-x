from dotenv import load_dotenv
from colorama import init, Fore


def print_colored_output(input_text, response_text, input_color="blue", response_color="green", full_color=False):
    """ Print the input and response text with the specified colors

    Args:
        input_text (str): The input text
        response_text (str): The response text
        input_color (str): The color to use for the input text
        response_color (str): The color to use for the response text
        full_color (bool): If True, color the entire string, otherwise color only the first part

    Returns:
        None; prints the input and response text with the specified colors
    """

    input_color_code = getattr(Fore, input_color.upper(), Fore.RESET)
    response_color_code = getattr(Fore, response_color.upper(), Fore.RESET)

    if full_color:
        print(f"{input_color_code}Input         : '{input_text}'{Fore.RESET}")
        print(f"{response_color_code}Response      : '{response_text.strip()}'{Fore.RESET}\n")
    else:
        print(f"{input_color_code}Input         :{Fore.RESET} '{input_text}'")
        print(f"{response_color_code}Response      :{Fore.RESET} '{response_text.strip()}'\n")


def retrieve_local_api_keys():
    """ Load the API keys from the .env file using dotenv """
    load_dotenv()
