from typing import Set, List

from backend.core import run_llm, summary_llm
import streamlit as st
from streamlit_chat import message

st.header("JUNIPER NETWORKS : Juniper BNG CUPS Document ChatBot")

# prompt = st.text_input("Prompt", placeholder="Enter your prompt here..")
prompt = st.chat_input( placeholder="Enter your message here...") 
# message(message, 
#             is_user=False, 
#             avatar_style="adventurer", # change this for different user icon
#             seed=123, # or the seed for different user icons
# )
count = 0
if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []

if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []



def create_sources_string(source_urls: List[str]) -> str :
    if not source_urls:
        return ""
    # sources_list = list(source_urls)
    # sources_list.sort()
    sources_string = "sources:\n"
    for i,source in enumerate(source_urls):
        sources_string += f"{i+1}. {source}\n"
    return sources_string

if prompt:
    with st.spinner("Generating response.."):
        # import time
        # time.sleep(3)
        # if "summary" or "summarize" not in prompt:
        #     generated_response = run_llm(query=prompt, chat_history=st.session_state["chat_history"])
        # else:
        #     generated_response = summary_llm(query=prompt, chat_history=st.session_state["chat_history"])
        generated_response, all_source = run_llm(query=prompt, chat_history=st.session_state["chat_history"])
        # print(generated_response)
        print(all_source)
        # sources = set([doc.metadata["source"] for doc in generated_response["source_documents"]])
        # Extracting sources and converting to a list
        # Creating an ordered set to ensure uniqueness and maintaining order
        sources = list(dict.fromkeys([doc.metadata["source"] for doc in generated_response["source_documents"]]))
        all_sources = create_sources_string(list(dict.fromkeys((all_source))))
        formatted_response = (f"{generated_response['answer']} \n\n  {create_sources_string(sources)}")
        
        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answers_history"].append(formatted_response)
        st.session_state["chat_history"].append((prompt, generated_response["answer"]))
        


if st.session_state["chat_answers_history"]:
    for i, j in zip(
        st.session_state["chat_answers_history"],
        st.session_state["user_prompt_history"],
    ):
        message(j, is_user=True, key=f"user_message_{count}")
        message(i, key=f"generated_response_{count}")
        
        if st.button(label="All Sources",help="Displays all sources related to the query", key=f"sources_button_{count}"):
            st.write(all_sources)
            print(all_sources)
            # message(all_source)
        count += 1

        


# def callback():
#   st.write("You clicked the button!")

# st.button("Click me!", on_click=callback)

# import streamlit as st

# # Create a simple chat application
# st.header("Chat Application")

# if not "messages" in st.session_state:
#     st.session_state.messages = []

# user_input = st.chat_input()

# if user_input:
#     st.session_state.messages.append({"role": "user", "content": user_input})

# # Add a button that prints a message to the console when clicked
# if st.button("Click me!"):
#     print("Button clicked!")
#     print(st.session_state.messages)


# import streamlit as st

# # Create a simple application with a button
# st.header("Simple Application")

# if 'counter' not in st.session_state:
#     st.session_state['counter'] = 0

# # Add a button that updates the state of the application when clicked
# if st.button("Click me!"):
#     st.session_state.counter += 1

# # Display the current value of the counter
# st.write("Counter:", st.session_state.counter)


