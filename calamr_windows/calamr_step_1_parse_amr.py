# # calamr_step_1_parse_amr.py
# import os
# import json
# import time
# import argparse
# from amrlib.models.parse_xfm.inference import Inference
#
# # Disable TensorFlow backend for HuggingFace
# os.environ["TRANSFORMERS_NO_TF"] = "1"
#
# # Paths
# model_path = r"D:\assignments\independent_study\project_CALAMR\project_CALAMR\CALAMR\models\model_parse_xfm_bart_large-v0_1_0\model_parse_xfm_bart_large-v0_1_0"
# input_path = "corpus/input_docs.json"
# output_path = "corpus/parsed_amrs.json"
# os.makedirs("corpus", exist_ok=True)
#
# # Parse arguments
# parser_arg = argparse.ArgumentParser()
# parser_arg.add_argument("--start", type=int, default=0)
# parser_arg.add_argument("--end", type=int, default=None)
# args = parser_arg.parse_args()
#
# # Load input
# with open(input_path, "r", encoding="utf-8") as f:
#     input_data = json.load(f)
#
# # Slice range if needed
# input_data = input_data[args.start:args.end] if args.end else input_data[args.start:]
#
# # Load model (faster config)
# parser = Inference(model_dir=model_path, batch_size=8, num_beams=1)
#
# output_data = []
#
# print(f"[•] Parsing {len(input_data)} documents (batch_size=8, beams=1)...")
#
# for entry in input_data:
#     doc_id = entry["id"]
#     body = entry["body"][:2000].strip()  # Truncate to ~2000 characters
#     summary = entry["summary"][:1000].strip()  # Truncate summary
#
#     if not body or not summary:
#         print(f"[!] Skipping empty {doc_id}")
#         continue
#
#     try:
#         start_time = time.time()
#         body_amr = parser.parse_sents([body])[0]
#         summary_amr = parser.parse_sents([summary])[0]
#
#         output_data.append({
#             "id": doc_id,
#             "body": body,
#             "summary": summary,
#             "body_amr": body_amr.strip(),
#             "summary_amr": summary_amr.strip()
#         })
#
#         duration = time.time() - start_time
#         print(f"[✓] Parsed {doc_id} in {duration:.2f}s")
#
#     except Exception as e:
#         print(f"[!] Failed parsing {doc_id}: {e}")
#
# # Save parsed AMRs
# with open(output_path, "w", encoding="utf-8") as f:
#     json.dump(output_data, f, indent=2)
#
# print(f"[🎯] Saved {len(output_data)} AMRs to {output_path}")


# calamr_step_1_parse_amr.py
import os
import json
import time
import argparse
from amrlib.models.parse_xfm.inference import Inference
from pathlib import Path

# Disable TensorFlow backend for HuggingFace
os.environ["TRANSFORMERS_NO_TF"] = "1"

# Paths
model_path = r"D:\assignments\independent_study\project_CALAMR\project_CALAMR\project_CALAMR\CALAMR\models\model_parse_xfm_bart_large-v0_1_0\model_parse_xfm_bart_large-v0_1_0"
input_path = "corpus/input_docs.json"
output_path = "corpus/parsed_amrs.json"
os.makedirs("corpus", exist_ok=True)

# Parse arguments
parser_arg = argparse.ArgumentParser()
parser_arg.add_argument("--start", type=int, default=0)
parser_arg.add_argument("--end", type=int, default=None)
args = parser_arg.parse_args()

# Load input
with open(input_path, "r", encoding="utf-8") as f:
    input_data = json.load(f)

# Slice range if needed
input_data = input_data[args.start:args.end] if args.end else input_data[args.start:]

# Load model
parser = Inference(model_dir=model_path, batch_size=8, num_beams=1)

print(f"[•] Loaded {len(input_data)} documents from {input_path}")
print(f"[•] Using parser model: {model_path}")
start_time = time.time()

output_data = []

for entry in input_data:
    doc_id = entry["id"]
    body = entry["body"][:2000].strip()
    summary = entry["summary"][:1000].strip()

    if not body or not summary:
        print(f"[!] Skipping empty {doc_id}")
        continue

    try:
        doc_start = time.time()
        body_amr = parser.parse_sents([body])[0]
        summary_amr = parser.parse_sents([summary])[0]

        output_data.append({
            "id": doc_id,
            "body": body,
            "summary": summary,
            "body_amr": body_amr.strip(),
            "summary_amr": summary_amr.strip()
        })

        duration = time.time() - doc_start
        print(f"[✓] Parsed {doc_id} in {duration:.2f}s")

    except Exception as e:
        print(f"[!] Failed parsing {doc_id}: {e}")

# Save parsed AMRs
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=2)

total_time = time.time() - start_time
output_size = Path(output_path).stat().st_size / 1024

print(f"[🎯] Saved {len(output_data)} AMRs to {output_path} ({output_size:.1f} KB)")
print(f"[⏱️] Total pipeline time: {total_time:.2f}s")
