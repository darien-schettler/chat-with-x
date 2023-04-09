from threading import Lock
import gradio as gr

from chat_with_x.utils.dataclasses import InstructionDescriptions, TaskDescriptions
from chat_with_x.processing.query_data import get_chain
from misc.utils import load_pickle_file, retrieve_local_api_keys, remove_api_keys

PKL_PATH = "idem_vs.pkl"
_instruction_keys = [x for x in InstructionDescriptions.__dict__.keys() if not x.startswith("_")]
_task_keys = [x for x in TaskDescriptions.__dict__.keys() if not x.startswith("_")]


def _live_update(token: str):
    chatbot.append("ai", token)


# def _get_chain_from_pkl(api_key=None, pkl_path="idem_vs.pkl", ):
#     """ Authenticate. Load the vectorstore from a pickle file. Remove authentication."""
#     # auth from passed key
#     if api_key is not None:
#         os.environ["OPENAI_API_KEY"] = api_key
#
#         os.environ["OPENAI_API_KEY"] = ""
#     # try auth from env file if local run
#     else:
#         retrieve_local_api_keys()
#         chain = get_chain(load_pickle_file(pkl_path))
#         # remove_api_keys()
#     return chain


class ChatWrapper:
    def __init__(self):
        self.lock = Lock()

    def __call__(self, input_text, document_description, task_type, instruction,
                 history=None, task_type_txt=None,
                 instruction_txt=None, api_key=None, temp=0.7):
        def __live_update_callback(token: str):
            nonlocal output
            output += token
            _live_update(token)

        # Authenticate.
        retrieve_local_api_keys()

        output = ""
        with self.lock:
            history = history or []
            chain = get_chain(load_pickle_file(PKL_PATH), document_description,
                              task_type=task_type, instruction=instruction, model_temperature=temp)
            output = chain({"question": input_text, "chat_history": history})["answer"]
            history.append((input_text, output))

        # Clean up
        # remove_api_keys()

        return history, history


chat = ChatWrapper()

block = gr.Blocks(css=".gradio-container {background-color: lightgray}")
with block:
    with gr.Row():
        gr.Markdown("<h3><center>Chat-With-X (IDEM by default)</center></h3>")

        # openai_api_key_textbox = gr.Textbox(
        #     placeholder="Paste your OpenAI API key (sk-...)",
        #     show_label=False,
        #     lines=1,
        #     type="password",
        # )

        document_description = gr.Textbox(
            label="Enter a description for the document:",
            placeholder="All of the documentation for the IDEM Infrastructure-as-Code language based on SaltStack.",
            lines=1,
        )
    with gr.Row():
        task_type_dropdown = gr.Dropdown(
            label="Select Task Type or Provide Your Own",
            choices=_task_keys,
            value=_task_keys[0],
            allow_custom_value=True
        )

        instruction_dropdown = gr.Dropdown(
            label="Select Instruction or Provide Your Own",
            choices=_instruction_keys,
            value=_instruction_keys[0],
            allow_custom_value=True
        )

    chatbot = gr.Chatbot()

    with gr.Row():
        message = gr.Textbox(
            label="What's your question?",
            placeholder="Ask questions about the document",
            lines=1,
        )
        submit = gr.Button(value="Send", variant="secondary").style(full_width=False)

    gr.Examples(
        examples=[
            "Create a highly available collection of virtual machines in Azure",
            "How to delete resources given an SLS file?",
            "Modify an existing SLS file to make it more secure",
        ],
        inputs=message,
    )

    gr.HTML("Demo Application Showing LangChain Chains in Action")
    state = gr.State()
    # agent_state = gr.State()

    submit.click(
        chat,
        inputs=[
            message, document_description, task_type_dropdown, instruction_dropdown, state
        ],
        outputs=[
            chatbot, state
        ])

    message.submit(
        chat,
        inputs=[
            message, document_description, task_type_dropdown, instruction_dropdown, state
        ],
        outputs=[
            chatbot, state
        ])

    # openai_api_key_textbox.change(
    #     _get_chain_from_pkl,
    #     inputs=[openai_api_key_textbox],
    #     outputs=[agent_state],
    # )

block.launch(debug=True)
