import re
import time
import spacy
from datetime import datetime
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from transformers import GPT2LMHeadModel, GPT2Tokenizer

try:
    spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")

nlp = spacy.load("en_core_web_sm")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model_gpt = GPT2LMHeadModel.from_pretrained("gpt2")

month_normalization = {
    "January": "Jan",
    "February": "Feb",
    "March": "Mar",
    "April": "Apr",
    "May": "May",
    "June": "Jun",
    "July": "Jul",
    "August": "Aug",
    "September": "Sep",
    "October": "Oct",
    "November": "Nov",
    "December": "Dec"
}

def normalize_ip(ip):
    if ip == "::1":
        return "127.0.0.1"
    return ip

def parse_date(text):
    patterns = [
        '%B %dth %Y',
        '%B %d %Y',
        '%B %d',
        '%d %B %Y',
        '%d %B'
    ]

    for pattern in patterns:
        try:
            date_obj = datetime.strptime(text, pattern)
            day_of_month = str(date_obj.day)
            month_of_year = month_normalization.get(date_obj.strftime('%B'))
            year = str(date_obj.year) if '%Y' in pattern else None
            return day_of_month, month_of_year, year
        except ValueError:
            continue

    return None, None, None

def extract_entities(prompt):
    entities = {"day_of_month": None, "month_of_year": None, "year": None, "ip_address": None, "visited_page": None}
    ip_match = re.search(r'\b(\d{1,3}\.){3}\d{1,3}\b|\b(::1|([a-fA-F0-9]{1,4}:){1,7}:?)\b', prompt)
    if ip_match:
        entities["ip_address"] = normalize_ip(ip_match.group())
    doc = nlp(prompt)
    for ent in doc.ents:
        if ent.label_ == "DATE":
            normalized_date_text = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', ent.text)
            try:
                date_obj = datetime.strptime(normalized_date_text, '%d %B %Y')
                entities["day_of_month"] = str(date_obj.day)
                entities["month_of_year"] = month_normalization.get(date_obj.strftime('%B'))
                entities["year"] = str(date_obj.year)
            except ValueError:
                try:
                    date_obj = datetime.strptime(normalized_date_text, '%B %Y')
                    entities["month_of_year"] = month_normalization.get(date_obj.strftime('%B'))
                    entities["year"] = str(date_obj.year)
                except ValueError:
                    try:
                        date_obj = datetime.strptime(normalized_date_text, '%B %d')
                        entities["day_of_month"] = str(date_obj.day)
                        entities["month_of_year"] = month_normalization.get(date_obj.strftime('%B'))
                    except ValueError:
                        if any(month in normalized_date_text for month in month_normalization.keys()):
                            entities["month_of_year"] = month_normalization.get(normalized_date_text.capitalize(), None)
    if not entities["month_of_year"] or not entities["year"]:
        for token in doc:
            if token.text.capitalize() in month_normalization and not entities["month_of_year"]:
                entities["month_of_year"] = month_normalization.get(token.text.capitalize())
            if token.like_num and len(token.text) == 4 and not entities["year"]:
                entities["year"] = token.text
    visited_page_match = re.search(r'page:? ([\w/\.]+)', prompt.lower())
    if visited_page_match:
        entities["visited_page"] = visited_page_match.group(1)
    return entities

def generate_response(retrieved_logs, entities):
    filtered_logs = []
    for log in retrieved_logs:
        match = True
        log_ip_normalized = normalize_ip(log['metadata']['ip_address'])
        if entities["day_of_month"] and entities["day_of_month"] != log['metadata']['day_of_month']:
            match = False
        if entities["month_of_year"] and entities["month_of_year"] != log['metadata']['month_of_year']:
            match = False
        if entities["year"] and entities["year"] != log['metadata']['year']:
            match = False
        if entities["ip_address"] and entities["ip_address"] != log_ip_normalized:
            match = False
        if entities["visited_page"] and entities["visited_page"] not in log['metadata']['visited_page']:
            match = False
        if match:
            filtered_logs.append(log)
    if not filtered_logs:
        return "No matching logs found.", filtered_logs
    summary = summarize_logs(filtered_logs)
    context = "\n".join(
        [
            f"{log['metadata']['ip_address']} visited {log['metadata']['visited_page']} on {log['metadata']['day_of_month']} {log['metadata']['month_of_year']} {log['metadata']['year']}."
            for log in filtered_logs[:10]]
    )
    response = f"Summary of activities:\n{summary}\n\nDetailed log information:\n{context}"
    return response, filtered_logs

def summarize_logs(filtered_logs):
    summary = {}
    for log in filtered_logs:
        page = log['metadata']['visited_page']
        summary[page] = summary.get(page, 0) + 1
    summary_text = "\n".join([f"{page}: {count} unique visits" for page, count in summary.items()])
    return summary_text

def generate_generative_response(context):
    input_ids = tokenizer.encode(context, return_tensors='pt')
    max_new_tokens = 100 if input_ids.shape[1] < 200 else 50
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    output = model_gpt.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        num_return_sequences=1,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
        repetition_penalty=1.2,
        no_repeat_ngram_size=3
    )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

def getAnswer(prompt):
    start_time = time.time()
    entities = extract_entities(prompt)
    new_prompt = f"IP address: {entities['ip_address']}, Day of month: {entities['day_of_month']}, Month of year: {entities['month_of_year']}, Year: {entities['year']}, Visited page: {entities['visited_page']}"
    query_vector = model.encode(new_prompt).astype(float).tolist()
    retrieved_logs = index.query(vector=query_vector, top_k=20, include_metadata=True)['matches']
    relevant_logs = calculate_relevant_logs(entities)
    context, filtered_logs = generate_response(retrieved_logs, entities)
    if context != "No matching logs found.":
        final_response = generate_generative_response(context)
        print(final_response)
    else:
        print(context)
    end_time = time.time()
    time_diff = end_time - start_time
    print(f"Time taken to generate the answer: {time_diff:.2f} seconds")
    evaluate_performance(retrieved_logs, filtered_logs, relevant_logs)

def calculate_relevant_logs(entities):
    all_logs = index.query(vector=[0]*384, top_k=1000, include_metadata=True)['matches']
    relevant_logs = 0
    for log in all_logs:
        match = True
        log_ip_normalized = normalize_ip(log['metadata']['ip_address'])
        if entities["day_of_month"] and entities["day_of_month"] != log['metadata']['day_of_month']:
            match = False
        if entities["month_of_year"] and entities["month_of_year"] != log['metadata']['month_of_year']:
            match = False
        if entities["year"] and entities["year"] != log['metadata']['year']:
            match = False
        if entities["ip_address"] and entities["ip_address"] != log_ip_normalized:
            match = False
        if entities["visited_page"] and entities["visited_page"] not in log['metadata']['visited_page']:
            match = False
        if match:
            relevant_logs += 1
    return relevant_logs

def evaluate_performance(retrieved_logs, filtered_logs, relevant_logs):
    total_logs = len(retrieved_logs)
    matched_logs = len(filtered_logs)
    precision = matched_logs / total_logs if total_logs > 0 else 0
    recall = matched_logs / relevant_logs if relevant_logs > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    print(
        f"Performance Evaluation:\n- Total logs retrieved: {total_logs}\n- Matched logs: {matched_logs}\n- Relevant logs: {relevant_logs}\n- Precision: {precision:.2f}\n- Recall: {recall:.2f}\n- F1 Score: {f1_score:.2f}"
    )

if __name__ == "__main__":
    log_file_path = r"C:\xampp\apache\logs\access.log"
    pinecone = Pinecone(api_key="bda168eb-efa4-40ce-9478-e044821f98dc")
    if "log-index" not in pinecone.list_indexes():
        try:
            pinecone.create_index("log-index", dimension=384, metric="cosine")
        except:
            print("Index already exists, proceeding with existing index.")
    index = pinecone.Index("log-index")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    try:
        with open(log_file_path, 'r') as log_file:
            log_contents = log_file.readlines()
            for i, line in enumerate(log_contents):
                try:
                    parts = line.split(" ")
                    if len(parts) > 6:
                        ip_address = parts[0]
                        date_parts = parts[3].split("/")
                        day_of_month = date_parts[0].split("[")[1]
                        month_of_year = date_parts[1]
                        year = date_parts[2].split(":")[0]
                        visited_page = line.split("\"GET /")[1].split(" HTTP")[0]
                        vector = model.encode(
                            f"IP address: {ip_address}, Day of month: {day_of_month}, Month of year: {month_of_year}, Year: {year}, Visited page: {visited_page}"
                        ).astype(float).tolist()
                        metadata = {
                            "ip_address": ip_address,
                            "day_of_month": day_of_month,
                            "month_of_year": month_of_year,
                            "year": year,
                            "visited_page": visited_page
                        }
                        index.upsert([(str(i), vector, metadata)])
                        time.sleep(0.001)
                except IndexError:
                    print(f"Skipping malformed line: {line}")
                except Exception as e:
                    print(f"An error occurred while processing line {i}: {e}")
    except FileNotFoundError:
        print(f"File not found: {log_file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
    while True:
        prompt = input("Ask me anything: ")
        getAnswer(prompt)
