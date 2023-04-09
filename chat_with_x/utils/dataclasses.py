from dataclasses import dataclass


class BaseObjectDescriptions:
    """ A base class for dataclasses containing descriptions for each option """

    def get_all_option_keys(self) -> list:
        return list(self.__dict__.keys())

    def get_all_option_values(self) -> list:
        return list(self.__dict__.values())

    def print_all_options(self) -> None:
        print("\n\n==== HERE ARE ALL THE OPTIONS AND THEIR CORRESPONDING DESCRIPTIONS ====\n")
        for option, desc in self.__dict__.items():
            print(f"{option:^25}  ---->  '{desc}'")


@dataclass
class TaskDescriptions(BaseObjectDescriptions):
    """ A dataclass containing the task options and the descriptions for each """
    coding: str = "providing coding assistance related to the {one_line_desc}."
    creative_writing: str = "providing creative writing suggestions related to the {one_line_desc}."
    research: str = "providing research assistance related to the {one_line_desc}."
    initial: str = "answering questions about the {one_line_desc}."
    proofreading: str = "proofreading and offering suggestions to improve the {one_line_desc}."
    translation: str = "translating content related to the {one_line_desc} into another language."
    brainstorming: str = "brainstorming ideas and concepts related to the {one_line_desc}."
    editing: str = "editing and revising the content of the {one_line_desc}."
    fact_checking: str = "fact-checking information found in the {one_line_desc}."
    summarizing: str = "summarizing the main points of the {one_line_desc}."
    design: str = "providing design advice and suggestions for the {one_line_desc}."
    tutoring: str = "providing tutoring on topics related to the {one_line_desc}."
    data_analysis: str = "performing data analysis on data found in the {one_line_desc}."
    recipe_suggestions: str = "suggesting recipes and cooking tips related to the {one_line_desc}."
    recommendations: str = "providing recommendations related to the {one_line_desc}."
    exercise_planning: str = "creating exercise plans and offering fitness advice based on the {one_line_desc}."
    travel_planning: str = "planning travel itineraries and offering suggestions based on the {one_line_desc}."
    finance: str = "offering financial advice and insights related to the {one_line_desc}."
    legal: str = "providing legal guidance and information based on the {one_line_desc}."
    relationship_advice: str = "giving relationship advice and insights based on the {one_line_desc}."
    music: str = "suggesting music compositions and ideas related to the {one_line_desc}."
    poetry: str = "crafting poetry and offering poetic advice based on the {one_line_desc}."
    storytelling: str = "creating stories and offering storytelling guidance based on the {one_line_desc}."
    product_ideas: str = "suggesting product ideas and innovations related to the {one_line_desc}."
    math: str = "solving math problems and providing mathematical insights related to the {one_line_desc}."
    career_advice: str = "offering career guidance and suggestions based on the {one_line_desc}."
    language_learning: str = "providing language learning tips and resources related to the {one_line_desc}."
    art: str = "offering artistic guidance and suggestions related to the {one_line_desc}."
    health_advice: str = "providing health advice and recommendations based on the {one_line_desc}."
    nutrition: str = "giving nutrition tips and information related to the {one_line_desc}."
    parenting: str = "offering parenting advice and support based on the {one_line_desc}."
    gaming: str = "suggesting gaming strategies and tips related to the {one_line_desc}."
    event_planning: str = "planning events and offering event management advice based on the {one_line_desc}."
    diy: str = "providing do-it-yourself tips and project ideas related to the {one_line_desc}."
    gardening: str = "offering gardening advice and plant care tips based on the {one_line_desc}."
    motivation: str = "providing motivational tips and insights related to the {one_line_desc}."
    self_improvement: str = "offering self-improvement guidance and advice based on the {one_line_desc}."
    fashion: str = "providing fashion tips and suggestions related to the {one_line_desc}."
    personal_finance: str = "offering personal finance advice and insights based on the {one_line_desc}."
    home_organization: str = "giving home organization tips and ideas based on the {one_line_desc}."


@dataclass
class InstructionDescriptions(BaseObjectDescriptions):
    """ A dataclass containing the instructions options and descriptions of each """
    conversational: str = "Provide a conversational answer."
    code_output: str = "Provide code outputs and explanations. In cases where the code may be unclear, ensure you " \
                       "include adequate comments and/or docstrings."
    creative_suggestions: str = "Offer creative and imaginative suggestions and advice that expand on ideas."
    research_assistance: str = "Give thoughtful and precise research assistance and insights. Try to be factual and " \
                               "offer helpful context and descriptions to educate with your answers."
    proofreading: str = "Review the content for grammar, punctuation, and spelling errors. Suggest changes to improve" \
                        " readability and clarity."
    translation: str = "Translate the text accurately while preserving the original meaning and tone."
    brainstorming: str = "Generate a variety of ideas and concepts related to the topic. Encourage exploration and " \
                         "creative thinking."
    editing: str = "Revise the content to improve its structure, flow, and overall quality. Suggest changes that " \
                   "enhance the clarity and impact of the text."
    fact_checking: str = "Verify the accuracy of the information provided. Correct any inaccuracies and provide " \
                         "sources when possible."
    summarizing: str = "Provide a concise summary of the main points, highlighting the most important ideas and " \
                       "details."
    design_advice: str = "Suggest design improvements and creative concepts. Offer insights on aesthetics, " \
                         "usability, and functionality."
    tutoring: str = "Explain concepts clearly and patiently. Use examples and analogies to help with understanding."
    data_analysis: str = "Analyze data to find patterns, trends, and insights. Explain your findings in a clear and " \
                         "understandable manner."
    recipe_suggestions: str = "Recommend recipes and cooking tips. Offer creative ideas for ingredient substitutions " \
                              "or variations."
    recommendations: str = "Provide personalized recommendations based on the given information. Consider the user's " \
                           "preferences and needs."
    exercise_planning: str = "Create exercise plans tailored to individual goals and fitness levels. Offer advice on " \
                             "proper form and technique."
    travel_planning: str = "Plan travel itineraries that cater to individual preferences and interests. Suggest " \
                           "activities, accommodations, and dining options."
    finance: str = "Offer financial advice based on sound principles and current information. Help users make " \
                   "informed decisions about their finances."
    legal: str = "Provide accurate legal information and guidance. Clarify complex concepts and help users " \
                 "understand their options."
    relationship_advice: str = "Give empathetic and non-judgmental relationship advice. Offer insights and " \
                               "suggestions for improving communication and resolving conflicts."
    music: str = "Suggest music compositions and ideas that match the desired style and mood. Offer tips for " \
                 "improving musical skills and creativity."
    poetry: str = "Craft poetry that evokes emotion and captures the essence of the subject. Offer constructive " \
                  "feedback and suggestions for improvement."
    storytelling: str = "Create engaging stories with well-developed characters and plotlines. Offer guidance on " \
                        "storytelling techniques and narrative structure."
    product_ideas: str = "Propose innovative product ideas and concepts. Consider market needs, feasibility, " \
                         "and potential impact."
    math: str = "Solve math problems accurately and efficiently. Explain the solution process and provide " \
                "insights on mathematical concepts."
    career_advice: str = "Offer career guidance based on individual skills, interests, and goals. Suggest strategies " \
                         "for professional growth and development."
    language_learning: str = "Provide language learning tips and resources. Offer guidance on grammar, vocabulary, " \
                             "and pronunciation."
    art: str = "Offer artistic guidance and suggestions. Provide feedback on technique, composition, and style."
    health_advice: str = "Provide health advice based on current knowledge and best practices. Offer insights on " \
                         "prevention, treatment, and overall wellness."
    nutrition: str = "Give nutrition tips and information tailored to individual needs and goals. Suggest dietary " \
                     "changes and meal planning strategies."
    parenting: str = "Offer parenting advice based on empathy and understanding. Provide suggestions for addressing " \
                     "challenges and fostering healthy relationships."
    gaming: str = "Suggest gaming strategies and tips to improve performance and enjoyment. Offer insights on game " \
                  "mechanics, tactics, and techniques."
    event_planning: str = "Plan events that cater to the desired theme and atmosphere. Offer advice on logistics, " \
                          "entertainment, and budget management."
    diy: str = "Provide do-it-yourself tips and project ideas. Offer guidance on tools, materials, " \
               "and techniques for successful completion."
    gardening: str = "Offer gardening advice and plant care tips. Suggest strategies for healthy growth, " \
                     "pest control, and landscape design."
    motivation: str = "Provide motivational tips and insights. Offer encouragement and strategies for overcoming " \
                      "obstacles and achieving goals."
    self_improvement: str = "Offer self-improvement guidance and advice. Suggest techniques for personal growth, " \
                            "habit formation, and goal-setting."
    fashion: str = "Provide fashion tips and suggestions. Offer insights on current trends, personal style, " \
                   "and wardrobe essentials."
    personal_finance: str = "Offer personal finance advice and insights. Help users make informed decisions about " \
                            "budgeting, saving, and investing."
    home_organization: str = "Give home organization tips and ideas. Offer guidance on decluttering, " \
                             "storage solutions, and efficient space utilization."
