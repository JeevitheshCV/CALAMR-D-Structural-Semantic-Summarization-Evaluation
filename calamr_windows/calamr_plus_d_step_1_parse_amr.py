# # calamr_plus_d_step_1_parse_amr.py
# import os
# import json
# from amrlib.models.parse_xfm.inference import Inference
#
# # Load parser
# os.environ["TRANSFORMERS_NO_TF"] = "1"
# model_path = r"D:\ui_project\project_CALAMR\project_CALAMR\CALAMR\models\model_parse_xfm_bart_large-v0_1_0\model_parse_xfm_bart_large-v0_1_0"
# parser = Inference(model_dir=model_path, batch_size=4, num_beams=4)
#
# import torch
# print(f"[✓] Using device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
#
# # Paths
# input_path = os.path.join("corpus", "structured_docs.json")
# output_path = os.path.join("corpus", "parsed_amrs_structured.json")
# os.makedirs("corpus", exist_ok=True)
#
# # Load input
# with open(input_path, "r", encoding="utf-8") as f:
#     structured_data = json.load(f)
#
# output_data = []
#
# for doc in structured_data:
#     doc_id = doc["id"]
#     body_sents = doc["body_sents"]
#     summary_sents = doc["summary_sents"]
#
#     print(f"[•] Parsing {doc_id} ({len(body_sents)} body, {len(summary_sents)} summary)")
#
#     try:
#         # Parse all body and summary sentences
#         body_amrs = parser.parse_sents(body_sents)
#         summary_amrs = parser.parse_sents(summary_sents)
#
#         output_data.append({
#             "id": doc_id,
#             "body_sents": body_sents,
#             "summary_sents": summary_sents,
#             "body_amrs": body_amrs,
#             "summary_amrs": summary_amrs
#         })
#
#         print(f"[✓] Parsed {doc_id} ({len(body_amrs)} + {len(summary_amrs)} AMRs)")
#
#     except Exception as e:
#         print(f"[!] Failed parsing {doc_id}: {e}")
#
# # Save output
# with open(output_path, "w", encoding="utf-8") as f:
#     json.dump(output_data, f, indent=2)
#
# print(f"[🎯] Sentence-level AMRs saved to {output_path}")
# calamr_plus_d_step_1_parse_amr.py
# import os
# import json
# import time
# import argparse
# from amrlib.models.parse_xfm.inference import Inference
#
# # Avoid TensorFlow interference
# os.environ["TRANSFORMERS_NO_TF"] = "1"
#
# # Paths
# model_path = r"D:\assignments\independent_study\project_CALAMR\project_CALAMR\CALAMR\models\model_parse_xfm_bart_large-v0_1_0\model_parse_xfm_bart_large-v0_1_0"
# input_path = "corpus/structured_docs.json"
# output_path = "corpus/parsed_amrs_structured.json"
# os.makedirs("corpus", exist_ok=True)
#
# # CLI args for parallelization
# parser_arg = argparse.ArgumentParser()
# parser_arg.add_argument("--start", type=int, default=0)
# parser_arg.add_argument("--end", type=int, default=None)
# args = parser_arg.parse_args()
#
# # Load structured sentence-level input
# with open(input_path, "r", encoding="utf-8") as f:
#     all_data = json.load(f)
#
# # Slice for chunked processing
# data = all_data[args.start:args.end] if args.end else all_data[args.start:]
#
# # Load parser model (faster config)
# parser = Inference(model_dir=model_path, batch_size=8, num_beams=1)
#
# output_data = []
#
# print(f"[•] Parsing {len(data)} structured documents (batch_size=8, beams=1)...")
#
# for entry in data:
#     doc_id = entry["id"]
#     body_sents = [s[:300] for s in entry["body_sents"]]     # Truncate long sents
#     summary_sents = [s[:300] for s in entry["summary_sents"]]
#
#     try:
#         start_time = time.time()
#         body_amrs = parser.parse_sents(body_sents)
#         summary_amrs = parser.parse_sents(summary_sents)
#
#         output_data.append({
#             "id": doc_id,
#             "body_sents": body_sents,
#             "summary_sents": summary_sents,
#             "body_amrs": body_amrs,
#             "summary_amrs": summary_amrs
#         })
#
#         duration = time.time() - start_time
#         print(f"[✓] Parsed {doc_id} in {duration:.2f}s")
#
#     except Exception as e:
#         print(f"[!] Failed {doc_id}: {e}")
#
# # Save results
# with open(output_path, "w", encoding="utf-8") as f:
#     json.dump(output_data, f, indent=2)
#
# print(f"[🎯] Saved parsed AMRs to {output_path}")





# calamr_plus_d_step_1_parse_amr.py
import os
import json
import time
import argparse
from pathlib import Path
from amrlib.models.parse_xfm.inference import Inference

# Avoid TensorFlow interference
os.environ["TRANSFORMERS_NO_TF"] = "1"

# Paths
model_path = r"D:\assignments\independent_study\project_CALAMR\project_CALAMR\project_CALAMR\CALAMR\models\model_parse_xfm_bart_large-v0_1_0\model_parse_xfm_bart_large-v0_1_0"
input_path = "corpus/structured_docs.json"
output_path = "corpus/parsed_amrs_structured.json"
os.makedirs("corpus", exist_ok=True)

# CLI args
parser_arg = argparse.ArgumentParser()
parser_arg.add_argument("--start", type=int, default=0)
parser_arg.add_argument("--end", type=int, default=None)
args = parser_arg.parse_args()

# Load structured data
with open(input_path, "r", encoding="utf-8") as f:
    all_data = json.load(f)

# Slice range
data = all_data[args.start:args.end] if args.end else all_data[args.start:]

# Load parser model
parser = Inference(model_dir=model_path, batch_size=8, num_beams=1)

print(f"[•] Loaded {len(data)} structured documents from {input_path}")
print(f"[•] Using parser model: {model_path}")
start_time = time.time()

output_data = []

for entry in data:
    doc_id = entry["id"]
    body_sents = [s[:300] for s in entry["body_sents"]]
    summary_sents = [s[:300] for s in entry["summary_sents"]]

    try:
        doc_start = time.time()
        body_amrs = parser.parse_sents(body_sents)
        summary_amrs = parser.parse_sents(summary_sents)

        output_data.append({
            "id": doc_id,
            "body_sents": body_sents,
            "summary_sents": summary_sents,
            "body_amrs": body_amrs,
            "summary_amrs": summary_amrs
        })

        duration = time.time() - doc_start
        print(f"[✓] Parsed {doc_id} in {duration:.2f}s")

    except Exception as e:
        print(f"[!] Failed {doc_id}: {e}")

# Save output
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=2)

total_time = time.time() - start_time
output_size = Path(output_path).stat().st_size / 1024

print(f"[🎯] Parsed AMRs saved to {output_path} ({output_size:.1f} KB)")
print(f"[⏱️] Total time: {total_time:.2f}s")
