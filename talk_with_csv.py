from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.agents import create_pandas_dataframe_agent
from langchain.callbacks import get_openai_callback
import pandas as pd
from dotenv import load_dotenv 
import json
import openai
import streamlit as st
load_dotenv()

total_token = 0
total_cost = 0

openai.api_base = "https://oai.hconeai.com/v1"
llm_davinci = OpenAI(
    temperature=0.9,
    headers={
      "Helicone-Auth": "Bearer helicone api key",
        "Helicone-User-Id": "user-id"
    }
  )
llm_gpt = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.9,
    headers={
      "Helicone-Auth": "Bearer sk-oxyq5qy-gm4eqny-s7c3i3q-wkn7d7q",
        "Helicone-User-Id": "deepak.jaiswal@gmail.com"
    }
  )
def csv_tool(filename : str):

    df = pd.read_csv(filename)
    return create_pandas_dataframe_agent(llm=llm_davinci, df=df, verbose=True)

def ask_agent(agent, query):
    global total_token
    global total_cost
    with get_openai_callback() as cb:   
        """
        Query an agent and return the response as a string.

        Args:
            agent: The agent to query.
            query: The query to ask the agent.

        Returns:
            The response from the agent as a string.
        """
        # Prepare the prompt with query guidelines and formatting
        prompt = (
            """
            Let's decode the way to respond to the queries. The responses depend on the type of information requested in the query. 

            1. If the query requires a table, format your answer like this:
               {"table": {"columns": ["column1", "column2", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}}

            2. For a bar chart, respond like this:
               {"bar": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}

            3. If a line chart is more appropriate, your reply should look like this:
               {"line": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}

            Note: We only accommodate two types of charts: "bar" and "line".

            4. For a plain question that doesn't need a chart or table, your response should be:
               {"answer": "Your answer goes here"}

            For example:
               {"answer": "The Product with the highest Orders is '15143Exfo'"}

            5. If the answer is not known or available, respond with:
               {"answer": "I do not know."}

            Return all output as a string. Remember to encase all strings in the "columns" list and data list in double quotes. 
            For example: {"columns": ["Products", "Orders"], "data": [["51993Masc", 191], ["49631Foun", 152]]}

            Now, let's tackle the query step by step. Here's the query for you to work on: 
            """
            + query
         )

        # Run the prompt through the agent and capture the response.
        response = agent.run(prompt)

        # Return the response converted to a string.
        total_cost = total_cost + cb.total_cost
        total_token = total_token + cb.total_tokens
        return str(response)

def decode_response(response: str) -> dict:
    """This function converts the string response from the model to a dictionary object.

    Args:
        response (str): response from the model

    Returns:
        dict: dictionary with response data
    """
    return json.loads(response)

def write_answer(response_dict: dict):
    """
    Write a response from an agent to a Streamlit app.

    Args:
        response_dict: The response from the agent.

    Returns:
        None.
    """

    # Check if the response is an answer.
    if "answer" in response_dict:
        st.write(response_dict["answer"])

    # Check if the response is a bar chart.
    if "bar" in response_dict:
        data = response_dict["bar"]
        try:
            df_data = {
                    col: [x[i] if isinstance(x, list) else x for x in data['data']]
                    for i, col in enumerate(data['columns'])
                }       
            df = pd.DataFrame(df_data)
            df.set_index("Products", inplace=True)
            st.bar_chart(df)
        except ValueError:
            print(f"Couldn't create DataFrame from data: {data}")

    # Check if the response is a line chart.
    if "line" in response_dict:
        data = response_dict["line"]
        try:
            df_data = {col: [x[i] for x in data['data']] for i, col in enumerate(data['columns'])}
            df = pd.DataFrame(df_data)
            df.set_index("Products", inplace=True)
            st.line_chart(df)
        except ValueError:
            print(f"Couldn't create DataFrame from data: {data}")


    # Check if the response is a table.
    if "table" in response_dict:
        data = response_dict["table"]
        df = pd.DataFrame(data["data"], columns=data["columns"])
        st.table(df)
st.set_page_config(page_title="👨‍💻 Talk with your CSV")
st.title("👨‍💻 Talk with your CSV")

st.write("Please upload your CSV file below.")

data = st.file_uploader("Upload a CSV" , type="csv")

query = st.text_area("Send a Message")

if st.button("Submit Query", type="primary"):
    # Create an agent from the CSV file.
    agent = csv_tool(data)

    # Query the agent.
    response = ask_agent(agent=agent, query=query)

    # Decode the response.
    decoded_response = decode_response(response)

    # Write the response to the Streamlit app.
    write_answer(decoded_response)

st.write(f"Total Token {total_token}")
st.write(f"Total Cost {total_cost}")