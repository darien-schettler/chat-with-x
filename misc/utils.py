from dotenv import load_dotenv
from colorama import init, Fore


def print_colored_output(input_text, response_text,
                         system_message=None,
                         input_color="blue",
                         response_color="green",
                         system_color="magenta",
                         full_color=False):
    """ Print the input and response text with the specified colors

    Args:
        input_text (str): The input text
        response_text (str): The response text
        system_message (str, optional): The system message text
        input_color (str, optional): The color to use for the input text
        response_color (str, optional): The color to use for the response text
        system_color (str, optional): The color to use for the system message text
        full_color (bool, optional): If True, color the entire string, otherwise color only the first part

    Returns:
        None; prints the input and response text with the specified colors
    """

    input_color_code = getattr(Fore, input_color.upper(), Fore.RESET)
    response_color_code = getattr(Fore, response_color.upper(), Fore.RESET)
    system_color_code = getattr(Fore, system_color.upper(), Fore.RESET)

    if full_color:
        if system_message:
            print(f"\n{system_color_code}System Message: '{system_message}'{Fore.RESET}")
        print(f"{input_color_code}Input         : '{input_text}'{Fore.RESET}")
        print(f"{response_color_code}Response      : '{response_text.strip()}'{Fore.RESET}\n")
    else:
        if system_message:
            print(f"\n{system_color_code}System Message:{Fore.RESET} '{system_message}'")
        print(f"{input_color_code}Input         :{Fore.RESET} '{input_text}'")
        print(f"{response_color_code}Response      :{Fore.RESET} '{response_text.strip()}'\n")


def retrieve_local_api_keys():
    """ Load the API keys from the .env file using dotenv """
    load_dotenv()
