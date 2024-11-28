# langchain_analysis.py

import os
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI  # Updated import
from langchain.chains import LLMChain
from dotenv import load_dotenv

def chunk_text(text, chunk_size=8000, chunk_overlap=500):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

def analyze_chunks(chunks, api_key, output_file="analysis_results.json", model_name="gpt-4"):
    if os.path.exists(output_file):
        print(f"Loading existing results from {output_file}")
        try:
            with open(output_file, "r") as file:
                return json.load(file)
        except json.JSONDecodeError as e:
            print(f"Error loading JSON: {e}")
            return []

    analysis_template = """
    You are a financial expert specializing in stock analysis. Based on the following transcript segment, provide a structured JSON response containing:

    {
        "key_financial_insights": [
            "Insight 1",
            "Insight 2",
            ...
        ],
        "stock_or_sector_recommendations": [
            "Recommendation 1",
            "Recommendation 2",
            ...
        ],
        "actions_to_consider": {
            "Buy": ["Ticker1", "Ticker2", ...],
            "Hold": ["Ticker3", "Ticker4", ...],
            "Sell": ["Ticker5", "Ticker6", ...]
        }
    }

    **Important:** 
    - Ensure the JSON is valid.
    - Under each action (Buy, Hold, Sell), list only the ticker symbols separated by commas.
    - Do not include descriptive sentences or reasons in the Buy/Hold/Sell lists.
    - Provide reasons for each recommendation in the "stock_or_sector_recommendations" section.

    Transcript Segment:
    {chunk}
    """

    analysis_prompt_template = PromptTemplate(
        input_variables=["chunk"], template=analysis_template
    )

    llm = ChatOpenAI(
        openai_api_key=api_key,
        temperature=0.2,
        model_name=model_name,
        max_tokens=1500
    )

    chain = LLMChain(llm=llm, prompt=analysis_prompt_template)

    analysis_results = []
    for idx, chunk in enumerate(chunks, 1):
        print(f"Analyzing chunk {idx}/{len(chunks)}: {chunk[:100]}...")
        try:
            result = chain.run(chunk)
            parsed_result = json.loads(result)
            analysis_results.append(parsed_result)
        except json.JSONDecodeError as e:
            print(f"JSON parsing error in chunk {idx}: {e}")
            analysis_results.append({"error": f"JSON parsing error: {e}"})
        except Exception as e:
            print(f"Error processing chunk {idx}: {e}")
            analysis_results.append({"error": f"Error: {e}"})

    try:
        with open(output_file, "w") as file:
            json.dump(analysis_results, file, indent=4)
        print(f"Analysis results saved to {output_file}")
    except Exception as e:
        print(f"Error saving results: {e}")

    return analysis_results
