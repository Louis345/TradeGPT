# main.py

import sys
import json
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_analysis import chunk_text, analyze_chunks
from dotenv import load_dotenv
import os
import subprocess

def get_video_id(url_or_id):
    """
    Extracts the YouTube video ID from a URL or returns it directly if already provided.
    """
    if "youtube.com" in url_or_id or "youtu.be" in url_or_id:
        parsed_url = urlparse(url_or_id)
        if parsed_url.netloc == "youtu.be":
            return parsed_url.path[1:]  # Extract video ID from short URL
        elif "v" in parse_qs(parsed_url.query):
            return parse_qs(parsed_url.query)["v"][0]  # Extract video ID from long URL
    return url_or_id  # Assume it's already the video ID

def save_analysis_results(analysis_results, video_id):
    """
    Saves the analysis results to a JSON file and a human-readable summary file.
    """
    analysis_file = f"{video_id}_analysis.json"
    summary_file = f"{video_id}_summary.txt"
    try:
        with open(analysis_file, "w") as file:
            json.dump(analysis_results, file, indent=4)
        print(f"Analysis saved to {analysis_file}")
        
        # Create a human-readable summary
        with open(summary_file, "w") as file:
            for idx, chunk in enumerate(analysis_results, 1):
                if "error" in chunk:
                    file.write(f"Chunk {idx} Error: {chunk['error']}\n\n")
                    continue
                file.write(f"Chunk {idx}:\n")
                file.write("1. Key Financial Insights:\n")
                for insight in chunk["key_financial_insights"]:
                    file.write(f"   - {insight}\n")
                file.write("\n2. Stock or Sector Recommendations:\n")
                for rec in chunk["stock_or_sector_recommendations"]:
                    file.write(f"   - {rec}\n")
                file.write("\n3. Actions to Consider:\n")
                for action, tickers in chunk["actions_to_consider"].items():
                    tickers_formatted = ", ".join(tickers)
                    file.write(f"   - {action}: {tickers_formatted}\n")
                file.write("\n\n")
        print(f"Summary saved to {summary_file}")
    except Exception as e:
        print(f"Error saving analysis results: {e}")

def main():
    # Load environment variables from .env file
    load_dotenv()

    # Get URL or video ID from the command line argument
    if len(sys.argv) < 2:
        print("Usage: python main.py <YouTube URL or Video ID>")
        sys.exit(1)

    url_or_id = sys.argv[1]
    video_id = get_video_id(url_or_id)

    # Step 1: Get the transcript
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        # Save transcript to JSON file
        transcript_file = f"{video_id}_transcript.json"
        with open(transcript_file, "w") as file:
            json.dump(transcript, file, indent=4)
        print(f"Transcript saved to {transcript_file}")
        
        # Convert transcript to plain text
        plain_text = " ".join([entry["text"] for entry in transcript])

        # Step 2: Chunk the plain text
        print("Chunking transcript...")
        chunks = chunk_text(plain_text)
        print(f"Total chunks created: {len(chunks)}")

        # Step 3: Analyze chunks with LangChain
        print("Analyzing transcript...")
        api_key = os.getenv("OPENAI_API_KEY")  # Load API key from environment variable
        desired_model = "gpt-4"  # Corrected model name
        analysis_output_file = f"{video_id}_analysis.json"  # Unique output file
        analysis_results = analyze_chunks(chunks, api_key, output_file=analysis_output_file, model_name=desired_model)

        # Step 4: Save analysis results and create summary
        save_analysis_results(analysis_results, video_id)

        # Step 5: Call stock_analysis.py with the analysis file
        print("Running stock_analysis.py...")
        try:
            subprocess.run(['python', 'stock_analysis.py', analysis_output_file], check=True)
            print("stock_analysis.py executed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error executing stock_analysis.py: {e}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
