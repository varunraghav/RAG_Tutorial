import os
from dotenv import load_dotenv
from autogen import ConversableAgent
import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from FlightAvailBtwCities import get_flight_availability
from airport_on_time_performance import process_airport_query  # Import airport module


#from chromadb.PersistentClient import PersistentClient


#from langchain.indexes import VectorStoreIndex
#from langchain.indexes.storage import StorageContext

load_dotenv()

def classify_query_with_llm(user_input):
    classification_prompt = f"""
    Your Task: 
    Determine if the following query is about travel information on rules and regulations that can be answered from the RAG document or 
    if they are about flight enquires that could be answered through API call 
    or if they are neither and cannot be answered by the chatbot.

    User Query: "{user_input}"

    Guidelines:
    1. The Rag document has information about luggage policy, including check-in and carry on bags, lost, damaged or delayed luggage.
        They also contain information about carrying pets, sports equipments, electronic equipments, and guide to travelling with children.
        If you feel query is related to the above topics classify it as 'Rag-Related'
    2. The API functionalitity can gather information about airport delays, on-time performance, or airport codes.
        It can answer the following questions:
        1. Flights between 2 cities.
        2. Non-stop flight between 2 locations
        3. Fastest or slowest flight between two locations.
        4. Flights between 2 cities with any additional filter.
        5. What are the cheapest flights from a particular location with or without additional filters
        6. What are the cheapest flight between 2 cities
        7. How is the performance of a particular aiport
        If you feel query is related to the above topics and questions under API functionality classify it as 'API-Related'
    3. Otherwise, classify it as 'General'.

    Response Format:
    - Rag-Related
    - API-Related
    - General
    """
    classification_reply = classifier_agent.generate_reply(
        messages=[{"content": classification_prompt, "role": "user"}]
    )
    print(classification_reply)
    return classification_reply.strip().lower()

def classify_api_query_with_llm(user_input):
    classification_prompt = f"""
    Your Task: 
    Determine if the following query is about the below given topics. If you determine are about the given below topics, 
    return their topic number.
    Else return "Data Not Available"

    User Query: "{user_input}"

    Topics:
    1. The Rag document has information about luggage policy, including check-in and carry on bags, lost, damaged or delayed luggage.
        They also contain information about carrying pets, sports equipments, electronic equipments, and guide to travelling with children.
        If you feel query is related to the above topics classify it as 'Rag-Related'
    2. The API functionalitity can gather information about airport delays, on-time performance, or airport codes.
        It can answer the following questions:
        1. Flights between 2 cities.
        2. Non-stop flight between 2 locations
        3. Fastest or slowest flight between two locations.
        4. Flights between 2 cities with any additional filter.
        5. What are the cheapest flights from a particular location with or without additional filters
        6. What are the cheapest flight between 2 cities
        7. How is the performance of a particular aiport
        If you feel query is related to the above topics and questions under API functionality classify it as 'API-Related'
    3. Otherwise, classify it as 'General'.

    Response Format:
    - 
    - API-Related
    - General
    """
    classification_reply = classifier_agent.generate_reply(
        messages=[{"content": classification_prompt, "role": "user"}]
    )
    print(classification_reply)
    return classification_reply.strip().lower()

def handle_flight_query_with_llm(user_input):
    """
    Gather flight details (origin, destination, date) from the user, use LLM to determine city/airport codes
    for origin and destination, and then fetch flight availability.
    """

    # Step 1: Ask the user for origin, destination, and date
    origin_input = input("Please provide the origin city or airport name: ").strip()
    destination_input = input("Please provide the destination city or airport name: ").strip()
    travel_date = input("Please provide the travel date (YYYY-MM-DD): ").strip()

    # Step 2: Use LLM to analyze and determine the city or airport codes for origin and destination
    llm_prompt = f"""
    Your Task: Convert the city or airport name provided by the user into its respective airport code.
    
    Origin: "{origin_input}"
    Destination: "{destination_input}"

    Response Format:
    origin_code: <airport code or MISSING>
    destination_code: <airport code or MISSING>
    """
    llm_reply = classifier_agent.generate_reply(messages=[{"content": llm_prompt, "role": "user"}])
    print(f"LLM Analysis:\n{llm_reply}")

    # Parse LLM response
    extracted_codes = {"origin_code": "MISSING", "destination_code": "MISSING"}
    for line in llm_reply.splitlines():
        if "origin_code:" in line:
            extracted_codes["origin_code"] = line.split("origin_code:")[1].strip()
        elif "destination_code:" in line:
            extracted_codes["destination_code"] = line.split("destination_code:")[1].strip()

    # Step 3: Ask for any missing details based on LLM output
    origin_code = extracted_codes["origin_code"]
    if origin_code == "MISSING":
        origin_code = input("The origin code is missing. Please provide the origin airport code: ").strip().upper()

    destination_code = extracted_codes["destination_code"]
    if destination_code == "MISSING":
        destination_code = input("The destination code is missing. Please provide the destination airport code: ").strip().upper()

    # Debugging: Show final parsed and user-filled details
    print(f"Final Details -> Origin Code: {origin_code}, Destination Code: {destination_code}, Date: {travel_date}")

    # Step 4: Validate all fields before making the API call
    if not origin_code or not destination_code or not travel_date:
        return "Missing required details for the flight query. Please provide all details."

    # Step 5: Fetch flight availability
    flight_info = get_flight_availability(origin_code, destination_code, travel_date)
    if "error" in flight_info:
        return f"Error fetching flight information: {flight_info['error']}"

    # Step 6: Format and return the response
    response = "Flight Availability:\n"
    for flight in flight_info:
        response += (f"Duration: {flight['duration']}, Stops: {flight['number_of_stops']}, "
                     f"Departure: {flight['departure_time']}, Arrival: {flight['arrival_time']}\n")
    return response



def handle_airport_query(user_input):
    return process_airport_query(user_input)  # Call airport agent to handle query


from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.postprocessor import KeywordNodePostprocessor

# Initialize the index
def initialize_index():
    db_path = "./my_project_chroma_db"
    os.makedirs(db_path, exist_ok=True)

    # Create or load the PersistentClient
    db = chromadb.PersistentClient(path=db_path)
    chroma_collection = db.get_or_create_collection("my-docs-collection")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    if chroma_collection.count() > 0:
        print("Loading existing index...")
        return VectorStoreIndex.from_vector_store(
            vector_store, storage_context=storage_context
        )
    else:
        print("Creating new index...")
        documents = []
        base_path = "./TravelInfo"
        for company_folder in os.listdir(base_path):
            company_path = os.path.join(base_path, company_folder)
            if os.path.isdir(company_path):
                company_name = company_folder
                loader = SimpleDirectoryReader(company_path)
                for doc in loader.load_data():
                    doc.metadata["company"] = company_name
                    documents.append(doc)

        return VectorStoreIndex.from_documents(
            documents, storage_context=storage_context
        )

# Initialize index
index = initialize_index()


# Classify query into specific, generic, or comparative
def classify_query(user_input):
    companies = ["United", "Delta", "American"]
    mentioned_companies = [c for c in companies if c.lower() in user_input.lower()]

    if len(mentioned_companies) == 1:
        return "specific", mentioned_companies[0]
    elif len(mentioned_companies) > 1:
        return "comparison", mentioned_companies
    else:
        return "generic", None

# Set up the retriever and postprocessor
def setup_query_engine(company_name=None):
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=10  # Adjust as needed
    )

    if company_name:
        postprocessor = KeywordNodePostprocessor(
            required_keywords=[company_name]  # Filter by company metadata
        )
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            node_postprocessors=[postprocessor]
        )
    else:
        # No filtering if no company name is provided
        query_engine = RetrieverQueryEngine(retriever=retriever)

    return query_engine

# Update the create_prompt function
def create_prompt(user_input):
    query_type, company_or_companies = classify_query(user_input)

    # Configure the query engine based on query type
    if query_type == "specific":
        print(f"Fetching documents for {company_or_companies}...")
        query_engine = setup_query_engine(company_name=company_or_companies)
        result = query_engine.query(user_input)
    elif query_type == "comparison":
        print(f"Fetching documents for comparison: {', '.join(company_or_companies)}...")
        query_engine = setup_query_engine()  # No filter for multiple companies
        result = query_engine.query(user_input)
    else:
        print("Fetching documents for generic query...")
        query_engine = setup_query_engine()  # No filter for generic queries
        result = query_engine.query(user_input)

    # Generate a prompt with the retrieved context
    prompt = f"""
    Your Task: Provide a concise and informative response to the user's query, drawing on the provided context.

    Context: {result}

    User Query: {user_input}

    Guidelines:
    1. Relevance: Focus directly on the user's question.
    2. Conciseness: Avoid unnecessary details.
    3. Accuracy: Ensure factual correctness.
    4. Clarity: Use clear language.
    5. Contextual Awareness: Use general knowledge if context is insufficient.
    6. Honesty: State if you lack information.

    Response Format:
    - Direct answer
    - Brief explanation (if necessary)
    - Citation (if relevant)
    - Conclusion
    """

    return prompt


llm_config = {
    "config_list": [
        {
            "model": "gpt-4o-mini",
            "api_key": os.getenv("OPENAI_API_KEY")
        }
    ]
}

classifier_agent = ConversableAgent(
    name="classifierbot",
    system_message="""You are an intelligent text classifier agent. 
                    Your task is to classify the input query into categories.
                    Strictly follow the instructions of the user message""",
    llm_config=llm_config,
    code_execution_config=False,
    human_input_mode="NEVER",
)

rag_agent = ConversableAgent(
    name="RAGbot",
    system_message="""You are a helpful customer service agent.
                        Your task is to answer customer's question based on the prompt""",
    llm_config=llm_config,
    code_execution_config=False,
    human_input_mode="NEVER",
)




def main():
    print("Welcome to RAGbot! Type 'exit', 'quit', or 'bye' to end the conversation.")
    while True:
        user_input = input("\nUser: ")

        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Goodbye! Have a great day!")
            break

        query_type = classify_query_with_llm(user_input)
        print(f"the query type is {query_type}")
        if query_type == "- rag-related":
            prompt = create_prompt(user_input)
            reply = rag_agent.generate_reply(messages=[{"content": prompt, "role": "user"}])
        elif query_type == "- API-Related":
            reply = handle_airport_query(user_input)
            reply = handle_flight_query_with_llm(user_input)
        else:
           print("This is beyong the scope of the project")

if __name__ == "__main__":
    main()