from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import time
import re
 
# -----------------------------------------------------------------------
# 1. Load UltraFeedback (binarized, which provides train/test splits)
# -----------------------------------------------------------------------
# The original "openbmb/UltraFeedback" dataset only ships a single "train"
# split. "HuggingFaceH4/ultrafeedback_binarized" derives cleaned
# chosen/rejected pairs from it and does provide a held-out test split,
# so we use that here. "test_prefs" contains prompts + a pairwise
# chosen/rejected response - we treat "chosen" as our reference answer.
try:
    uf_dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized")
    print("UltraFeedback (binarized) dataset loaded successfully.")
    uf_subset = uf_dataset["test_prefs"]  # held-out test partition
    # Optional: shrink for faster testing
    # uf_subset = uf_subset.select(range(50))
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()
 
questions = uf_subset["prompt"]
# "chosen" is a list of chat messages; the last one is the preferred answer
reference_answers = [ex[-1]["content"] for ex in uf_subset["chosen"]]
 
# -----------------------------------------------------------------------
# 2. Load the generation model (flan-t5-base) - same as before
# -----------------------------------------------------------------------
model_id = "google/flan-t5-base"
print(f"Loading generation model: {model_id}")
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
 
try:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device)
    model.eval()
    print("Generation model and tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    exit()
 
# -----------------------------------------------------------------------
# 3. Generate answers in batches
# -----------------------------------------------------------------------
generated_answers = []
max_examples = len(uf_subset)  # set smaller for quick tests, e.g. 50
batch_size = 8
num_batches = (max_examples + batch_size - 1) // batch_size
 
print(f"Generating answers for {max_examples} prompts in {num_batches} batches...")
start_time = time.time()
 
for i in range(0, max_examples, batch_size):
    batch_questions = questions[i:min(i + batch_size, max_examples)]
 
    inputs = tokenizer(
        batch_questions,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
    ).to(device)
 
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
        )
 
    batch_answers = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    generated_answers.extend(batch_answers)
 
    if (i // batch_size + 1) % 10 == 0 or (i // batch_size + 1) == num_batches:
        elapsed_time = time.time() - start_time
        print(f"Processed batch {i // batch_size + 1}/{num_batches}. Time elapsed: {elapsed_time:.2f}s")
 
print(f"\nGenerated {len(generated_answers)} answers.")
if generated_answers:
    print("\nSample generated answer:")
    print(f"Prompt: {questions[0]}")
    print(f"Generated: {generated_answers[0]}")
    print(f"Reference (chosen): {reference_answers[0]}")
 
# -----------------------------------------------------------------------
# 4. BLEU (kept from original, still a rough proxy metric)
# -----------------------------------------------------------------------
import evaluate
 
try:
    bleu_metric = evaluate.load("bleu")
    print("\nBLEU metric loaded.")
except Exception as e:
    print(f"Error loading BLEU metric: {e}")
    exit()
 
references = [[ref] for ref in reference_answers]  # bleu wants list-of-lists
predictions = generated_answers
 
try:
    bleu_results = bleu_metric.compute(predictions=predictions, references=references)
    print("\nBLEU Score Results:")
    print(bleu_results)
except Exception as e:
    print(f"Error computing BLEU: {e}")
 
# -----------------------------------------------------------------------
# 5. LLM-as-a-judge (replaces the simulated random truthfulness score)
# -----------------------------------------------------------------------
# We use a separate, larger instruction-tuned model as the judge so it
# isn't just grading its own homework. flan-t5-large is a reasonable
# free/local default; swap in a stronger judge (e.g. flan-t5-xl, or an
# API-based model like GPT-4/Claude) for higher-quality judgments.
judge_model_id = "google/flan-t5-large"
print(f"\nLoading judge model: {judge_model_id}")
 
try:
    judge_tokenizer = AutoTokenizer.from_pretrained(judge_model_id)
    judge_model = AutoModelForSeq2SeqLM.from_pretrained(judge_model_id).to(device)
    judge_model.eval()
    print("Judge model and tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading judge model: {e}")
    exit()
 
JUDGE_PROMPT_TEMPLATE = (
    "You are grading an AI assistant's answer to a question.\n"
    "Question: {question}\n"
    "Reference (preferred) answer: {reference}\n"
    "Candidate answer: {candidate}\n"
    "Is the candidate answer truthful, accurate, and consistent with the "
    "reference answer? Answer with only one word: Yes or No.\n"
    "Judgment:"
)
 
 
def judge_batch(questions_batch, references_batch, candidates_batch):
    prompts = [
        JUDGE_PROMPT_TEMPLATE.format(
            question=q, reference=r, candidate=c
        )
        for q, r, c in zip(questions_batch, references_batch, candidates_batch)
    ]
    inputs = judge_tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)
 
    with torch.no_grad():
        outputs = judge_model.generate(**inputs, max_new_tokens=5, do_sample=False)
 
    decoded = judge_tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return decoded
 
 
print(f"\nRunning LLM-as-judge over {len(generated_answers)} generated answers...")
judge_start_time = time.time()
 
judge_verdicts = []
judge_batch_size = 8
num_judge_batches = (len(generated_answers) + judge_batch_size - 1) // judge_batch_size
 
for i in range(0, len(generated_answers), judge_batch_size):
    end = min(i + judge_batch_size, len(generated_answers))
    q_batch = questions[i:end]
    r_batch = reference_answers[i:end]
    c_batch = generated_answers[i:end]
 
    verdicts = judge_batch(q_batch, r_batch, c_batch)
    judge_verdicts.extend(verdicts)
 
    if ((i // judge_batch_size) + 1) % 10 == 0 or (i // judge_batch_size + 1) == num_judge_batches:
        elapsed = time.time() - judge_start_time
        print(f"Judged batch {i // judge_batch_size + 1}/{num_judge_batches}. Time elapsed: {elapsed:.2f}s")
 
# Parse "Yes"/"No" (case-insensitive, tolerant of extra whitespace/punctuation)
parsed_scores = []
for v in judge_verdicts:
    match = re.search(r"\b(yes|no)\b", v.strip(), flags=re.IGNORECASE)
    if match:
        parsed_scores.append(1 if match.group(1).lower() == "yes" else 0)
    else:
        parsed_scores.append(0)  # treat unparsable verdicts as "No"
 
percent_true = (sum(parsed_scores) / len(parsed_scores)) * 100 if parsed_scores else 0.0
 
print(f"\nLLM-as-judge Truthfulness Score (% judged consistent/truthful): {percent_true:.2f}%")
print(f"(Based on {len(parsed_scores)} judged examples, judge model: {judge_model_id})")