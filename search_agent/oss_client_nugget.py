from __future__ import annotations

import argparse
#import logging
import csv
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import openai
from prompts import format_query
from rich import print as rprint
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import extract_retrieved_docids_from_result

from searcher.searchers import SearcherType

#logging.basicConfig(level=logging.INFO)


class SearchToolHandler:

    def __init__(
        self,
        searcher,
        snippet_max_tokens: int | None = None,
        k: int = 5,
        include_get_document: bool = True,
    ):
        self.searcher = searcher
        self.snippet_max_tokens = snippet_max_tokens
        self.k = k
        self.include_get_document = include_get_document

        self.last_reasoning = ""

        self.tokenizer = None
        if snippet_max_tokens and snippet_max_tokens > 0:
            self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    def execute_tool(self, tool_name: str, arguments: dict):
        if tool_name == "search":
            return self._search(arguments["user_query"])
        elif tool_name == "get_document":
            return self._get_document(arguments["docid"])
        else:
            raise ValueError(f"Unknown tool: {tool_name}")

    def get_tool_definitions(self):
        tools = [
            {
                "type": "function",
                "name": "search",
                "description": self.searcher.search_description(self.k),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_query": {
                            "type": "string",
                            "description": "Query to search the local knowledge base for relevant information",
                        }
                    },
                    "required": ["user_query"],
                    "additionalProperties": False,
                },
                "strict": True,
            }
        ]

        if self.include_get_document:
            tools.append(
                {
                    "type": "function",
                    "name": "get_document",
                    "description": self.searcher.get_document_description(),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "docid": {
                                "type": "string",
                                "description": "Document ID to retrieve",
                            }
                        },
                        "required": ["docid"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                }
            )

        return tools

    def _search(self, query: str):
        if self.last_reasoning:
            setattr(self.searcher, "last_reasoning", self.last_reasoning)
        candidates = self.searcher.search(query, self.k)
        if self.snippet_max_tokens and self.snippet_max_tokens > 0 and self.tokenizer:
            for cand in candidates:
                text = cand["text"]
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                if len(tokens) > self.snippet_max_tokens:
                    truncated_tokens = tokens[: self.snippet_max_tokens]
                    cand["snippet"] = self.tokenizer.decode(
                        truncated_tokens, skip_special_tokens=True
                    )
                else:
                    cand["snippet"] = text
        else:
            for cand in candidates:
                cand["snippet"] = cand["text"]

        results = []
        for cand in candidates:
            if cand.get("score") is None:
                results.append({"docid": cand["docid"], "snippet": cand["snippet"]})
            else:
                results.append(
                    {
                        "docid": cand["docid"],
                        "score": cand["score"],
                        "snippet": cand["snippet"],
                    }
                )

        return json.dumps(results, indent=2)

    def _get_document(self, docid: str):
        result = self.searcher.get_document(docid)
        if result is None:
            return json.dumps({"error": f"Document with docid '{docid}' not found"})
        return json.dumps(result, indent=2)

nugget_answer_format = """Exact Answer: {answer}  \nConfidence: {confidence}%"""

def new_phase_tokens():
    return {
        phase: {"total_tokens": 0, "input_tokens": 0, "output_tokens": 0, "calls": 0}
        for phase in ("decomp", "nugget", "main")
    }

def add_phase_tokens(phase_tokens, phase, token_info):
    """Accumulate one LLM call's usage under the given phase (decomp / nugget / main)."""
    if phase_tokens is None:
        return
    for key in ("total_tokens", "input_tokens", "output_tokens"):
        phase_tokens[phase][key] += token_info[key]
    phase_tokens[phase]["calls"] += 1

def apply_nugget_changes(current_nuggets, changes, max_adds, max_nuggets):
    """
    Apply add/remove changes (remove indices refer to current_nuggets) and return
    (updated_nuggets, indexed_units). indexed_units is current_nuggets followed by every
    unit the model added, matching the numbering the model uses for supporting_units.
    """
    nuggets = list(current_nuggets or [])
    raw_adds = [u if isinstance(u, str) else "" for u in changes.get("add") or []]
    indexed_units = nuggets + raw_adds
    remove = {i for i in changes.get("remove") or [] if isinstance(i, int)}
    updated = [n for i, n in enumerate(nuggets) if i not in remove]
    for unit in raw_adds[:max_adds]:
        if unit and unit not in updated:
            updated.append(unit)
    if len(updated) > max_nuggets:
        updated = updated[-max_nuggets:]
    return updated, indexed_units

_ANSWER_STOPWORDS = {"the", "a", "an", "of", "and"}

def _answer_tokens(text):
    return [t for t in re.findall(r"\w+", text.lower()) if t not in _ANSWER_STOPWORDS]

def answer_supported(answer, supporting_units, indexed_units):
    """True if every content token of the answer appears in the cited supporting units."""
    cited = [
        indexed_units[i] for i in supporting_units or []
        if isinstance(i, int) and 0 <= i < len(indexed_units)
    ]
    answer_tokens = _answer_tokens(answer)
    if not cited or not answer_tokens:
        return False
    cited_tokens = set(_answer_tokens(" ".join(cited)))
    return all(t in cited_tokens for t in answer_tokens)

def nugget_update_and_check(client: openai.OpenAI, query: str, search_text: str, current_nuggets, args, phase_tokens=None):
    """
    Single LLM call that:
      1. Updates the nugget list given new search results (add/remove)
      2. Checks whether the updated nuggets are sufficient to answer the query
      3. Returns the answer if sufficient and grounded in the cited supporting units, else None

    Returns (updated_nuggets, answer_output, candidate, check, output_items, token_info).
    candidate is an answer the model proposed that failed the supporting_units check; it is
    handed to the main agent as an unverified hypothesis rather than returned as final.
    """
    tools = [
        {
            "type": "function",
            "name": "update_nuggets_and_check",
            "description": (
                "Report changes to the list of information units given new evidence, "
                "then determine whether they collectively answer the original query."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "add": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            f"At most {args.max_nugget_adds} new information units from the search results "
                            "relevant to the original query. Each is one fact of at most 25 words ending "
                            "with its source docid in square brackets. Do not repeat existing units."
                        ),
                    },
                    "remove": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": (
                            "Indices of existing units that are irrelevant or redundant."
                        ),
                    },
                    "can_answer": {
                        "type": "string",
                        "enum": ["yes", "no"],
                        "description": (
                            "Whether the current units list is sufficient to fully answer "
                            "the original query."
                        ),
                    },
                    "answer": {
                        "type": "string",
                        "description": (
                            "The exact answer to the original query derived from the units. "
                            "Only populate this field if can_answer is 'yes'."
                        ),
                    },
                    "confidence": {
                        "type": "number",
                        "description": (
                            "Your confidence in your final answer, must be an integer value between 0 and 100. "
                            "Only populate this field if can_answer is 'yes'."
                        ),
                    },
                    "supporting_units": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": (
                            "Indices of the units that contain the answer and its supporting evidence. "
                            "Existing units keep their indices; added units continue the numbering. "
                            "Only populate this field if can_answer is 'yes'."
                        ),
                    },
                },
                "required": ["add", "remove", "can_answer"],
                "additionalProperties": False,
            },
            "strict": True,
        }
    ]
    prompt = format_query(
        query,
        "NUGGET_DELTA_UPDATE_AND_CHECK_PROMPT",
        add_info={'docs': search_text, 'current_units': current_nuggets, 'max_adds': args.max_nugget_adds}
    )
    request = {
        "model": args.model,
        "max_output_tokens": args.max_tokens,
        "input": [
            {"role": "user", "content": prompt}
        ],
        'tools': tools,
        "truncation": "auto",
        "reasoning": {"effort": 'low', "summary": "detailed"},
    }
    response = None
    while response is None:
        try:
            response = client.responses.create(**request)
        except:
            response = None
    usage = response.usage
    token_info = {
        "total_tokens": usage.total_tokens,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
    }
    add_phase_tokens(phase_tokens, "nugget", token_info)
    response_dict = response.model_dump(mode="python")
    output_items = []
    updated_nuggets = current_nuggets
    answer_output = None
    candidate = None
    check = None
    for i in range(len(response_dict['output'])):
        if response_dict['output'][i]['type'] == 'mcp_call' or response_dict['output'][i]['type'] == 'function_call':
            if response_dict['output'][i]['type'] == 'mcp_call':
                response_dict['output'][i] = {
                    'id': response_dict['output'][i]['id'],
                    'call_id': response_dict['output'][i]['id'],  # function_call uses call_id
                    'arguments': response_dict['output'][i]['arguments'],
                    'name': response_dict['output'][i]['name'],
                    'type': 'function_call',
                    'status': None
                }
            parsed_args = json.loads(response_dict['output'][i]['arguments'])
            updated_nuggets, indexed_units = apply_nugget_changes(
                current_nuggets, parsed_args, args.max_nugget_adds, args.max_nuggets
            )
            if parsed_args.get("can_answer")=="yes":
                answer = parsed_args.get("answer") or None
                confidence = parsed_args.get("confidence")
                supporting_units = parsed_args.get("supporting_units") or []
                if answer and confidence is not None:
                    supported = answer_supported(answer, supporting_units, indexed_units)
                    check = {
                        "answer": answer,
                        "confidence": confidence,
                        "supporting_units": supporting_units,
                        "accepted": supported,
                    }
                    if supported:
                        answer_output = nugget_answer_format.format(answer=answer, confidence=confidence)
                    else:
                        candidate = answer
        output_items.append(response_dict['output'][i])
    return updated_nuggets, answer_output, candidate, check, output_items, token_info

# TODO: make it so it connects to the decomp_query function tool
def init_query_decomp(client: openai.OpenAI, query: str, args, phase_tokens=None):
    tools = [
        {
            "type": "function",
            "name": "decomp_query",
            "description": "Formats the extracted subqueries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subqueries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "An array of identified subqueries.",
                    },
                },
                "required": ["subqueries"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    ]
    total_tokens = 0
    input_tokens = 0
    output_tokens = 0
    decomp_outputs = []
    prompt = format_query(query, "QUERY_DECOMP_TEMPLATE")
    request = {
        "model": args.model,
        "max_output_tokens": args.max_tokens,
        "input": [
            {"role": "user", "content": prompt}
        ],
        "tools": tools,
        "truncation": "auto",
        "reasoning": {"effort": 'medium', "summary": "detailed"},
    }
    response = None
    while response is None:
        try:
            response = client.responses.create(
                **request,
            )
            usage = response.usage
            total_tokens += usage.total_tokens
            input_tokens += usage.input_tokens
            output_tokens += usage.output_tokens
            add_phase_tokens(phase_tokens, "decomp", {'total_tokens': usage.total_tokens, 'input_tokens': usage.input_tokens, 'output_tokens': usage.output_tokens})
            response_dict = response.model_dump(mode="python")
            decomp = []
            for i in range(len(response_dict['output'])):
                if response_dict['output'][i]['type'] == 'mcp_call':
                    # TODO: upstream vLLM bug outputs mcp_call but only accepts function_call as input
                    # Convert mcp_call to function_call format
                    response_dict['output'][i] = {
                        'id': response_dict['output'][i]['id'],
                        'call_id': response_dict['output'][i]['id'],  # function_call uses call_id
                        'arguments': response_dict['output'][i]['arguments'],
                        'name': response_dict['output'][i]['name'],
                        'type': 'function_call',
                        'status': None
                    }
                if 'arguments' in response_dict['output'][i]:
                    decomp = json.loads(response_dict['output'][i]['arguments'])['subqueries']
                decomp_outputs.append(response_dict['output'][i])
            return decomp, decomp_outputs, {'total_tokens': total_tokens, 'input_tokens': input_tokens, 'output_tokens': output_tokens}
        except Exception as e:
            # print(f"[ERROR ENCOUNTERED]\t{type(e).__name__} {e}")
            response = None


def run_conversation_with_tools(
    client: openai.OpenAI,
    initial_request: dict,
    tool_handler: SearchToolHandler,
    qtext: str,
    args,
    nuggets: list = [],
    max_iterations: int = 100,
    verbose: bool = True,
    phase_tokens: dict | None = None,
    nugget_checks: list | None = None,
):

    tool_usage = {}

    total_tokens = 0
    input_tokens = 0
    output_tokens = 0

    messages = initial_request["input"]

    iteration = 1

    copy_messages = messages.copy()

    while iteration <= max_iterations:
        try:
            request = initial_request.copy()
            request["input"] = messages
            response = client.responses.create(
                **request,
            )
            usage = response.usage
            total_tokens += usage.total_tokens
            input_tokens += usage.input_tokens
            output_tokens += usage.output_tokens
            add_phase_tokens(phase_tokens, "main", {'total_tokens': usage.total_tokens, 'input_tokens': usage.input_tokens, 'output_tokens': usage.output_tokens})
        except Exception as e:
            if verbose:
                print(f"Error: {e}")
                rprint(f"Request: {request}")
            iteration += 1
            continue

        response_dict = response.model_dump(mode="python")

        if len(response_dict['output']) >= 1 and response_dict['output'][-1]['type'] == 'reasoning':
            continue

        for i in range(len(response_dict['output'])):
            if response_dict['output'][i]['type'] == 'mcp_call':
                # TODO: upstream vLLM bug outputs mcp_call but only accepts function_call as input
                # Convert mcp_call to function_call format
                response_dict['output'][i] = {
                    'id': response_dict['output'][i]['id'],
                    'call_id': response_dict['output'][i]['id'],  # function_call uses call_id
                    'arguments': response_dict['output'][i]['arguments'],
                    'name': response_dict['output'][i]['name'],
                    'type': 'function_call',
                    'status': None
                }
            messages.append(response_dict['output'][i])
            copy_messages.append(response_dict['output'][i])

        reasoning_text = ""
        for item in response_dict["output"]:
            if isinstance(item, dict) and item["type"] == "reasoning":
                summary = item.get("summary")
                if isinstance(summary, list):
                    reasoning_text = "\n".join(str(s) for s in summary if str(s).strip()).strip()
                if not reasoning_text:
                    parts = item.get("content", []) or []
                    chunks = []
                    for part in parts:
                        if isinstance(part, dict) and part.get("type") in {
                            "reasoning_text",
                            "output_text",
                            "text",
                        }:
                            text_val = str(part.get("text", "")).strip()
                            if text_val:
                                chunks.append(text_val)
                    reasoning_text = "\n".join(chunks).strip()
                if reasoning_text:
                    break
        #print("reasoning text:", reasoning_text)
        tool_handler.last_reasoning = reasoning_text
        #exit()

        function_calls = [item for item in response_dict['output'] if item['type'] in ['function_call', 'mcp_call', 'custom_tool_call']]

        if not function_calls:
            if messages[-1]['type'] != 'message':
                continue
            return copy_messages, tool_usage, "completed", {'total_tokens': total_tokens, 'input_tokens': input_tokens, 'output_tokens': output_tokens}

        new_messages = messages.copy()

        for tool_call in function_calls:
            try:
                arguments = json.loads(tool_call["arguments"])
                result = tool_handler.execute_tool(tool_call["name"], arguments)
                tool_usage[tool_call["name"]] = tool_usage.get(tool_call["name"], 0) + 1
                
                if tool_call["name"]=="search":
                    # print("Query Text @ Line461:", qtext)
                    updated_nuggets, early_answer, candidate, check, nug_output, nug_token_info = nugget_update_and_check(client, qtext, result, nuggets, args, phase_tokens)
                    nuggets = updated_nuggets
                    if check is not None and nugget_checks is not None:
                        nugget_checks.append({"phase": "main", **check})
                    copy_messages.extend(nug_output)
                    total_tokens += nug_token_info['total_tokens']
                    input_tokens += nug_token_info['input_tokens']
                    output_tokens += nug_token_info['output_tokens']
                    new_output = {
                        "search output": json.loads(result),
                        "gathered nuggets": nuggets,
                    }
                    if candidate:
                        new_output["unverified candidate answer"] = candidate
                    new_messages.append(
                        {
                            "type": "function_call_output",
                            "call_id": tool_call["call_id"],
                            "output": json.dumps(new_output, indent=2),
                        }
                    )
                    copy_messages.append(
                        {
                            "type": "function_call_output",
                            "call_id": tool_call["call_id"],
                            "output": json.dumps(new_output, indent=2),
                        }
                    )
                    if early_answer:
                        copy_messages.append(
                            {
                                "type": "message",
                                "content": [{'type': 'output_text', 'text':early_answer}]
                            }
                        )
                        return copy_messages, tool_usage, "completed", {'total_tokens': total_tokens, 'input_tokens': input_tokens, 'output_tokens': output_tokens}

                else:
                    new_messages.append(
                        {
                            "type": "function_call_output",
                            "call_id": tool_call["call_id"],
                            "output": result,
                        }
                    )
                    copy_messages.append(
                        {
                            "type": "function_call_output",
                            "call_id": tool_call["call_id"],
                            "output": result,
                        }
                    )


            except Exception as e:
                error_msg = f"Error executing {tool_call['name']}: {str(e)}"
                # print(error_msg)
                new_messages.append(
                    {
                        "type": "function_call_output",
                        "call_id": tool_call["call_id"],
                        "output": error_msg,
                    }
                )
                copy_messages.append(
                    {
                        "type": "function_call_output",
                        "call_id": tool_call["call_id"],
                        "output": error_msg,
                    }
                )
        messages = new_messages
        iteration += 1

    return copy_messages, tool_usage, "incomplete", {'total_tokens': total_tokens, 'input_tokens': input_tokens, 'output_tokens': output_tokens}


def _persist_response(
    out_dir: str,
    initial_request: dict,
    messages: list,
    tool_usage: dict,
    token_counts: dict,
    status: str,
    *,
    query_id: str | None = None,
    elapsed_seconds: float = 0,
    phase_tokens: dict | None = None,
    nugget_checks: list | None = None,
):
    os.makedirs(out_dir, exist_ok=True)

    call_output_by_id: dict[str, str | dict | None] = {}
    for item in messages or []:
        if isinstance(item, dict) and item.get("type") == "function_call_output":
            call_output_by_id[item.get("call_id")] = item.get("output")

    normalized_results: list[dict] = []

    for item in messages or []:
        if not isinstance(item, dict):
            continue

        itype = item.get("type")

        if itype == "function_call":
            call_id = item.get("call_id")
            normalized_results.append(
                {
                    "type": "tool_call",
                    "tool_name": item.get("name"),
                    "arguments": item.get("arguments"),
                    "output": call_output_by_id.get(call_id),
                }
            )

        elif itype == "reasoning":
            summary = item.get("summary")
            if isinstance(summary, list) and len(summary) > 0:
                reasoning_output = summary
            else:
                reasoning_output = []
                for part in item.get("content", []) or []:
                    if isinstance(part, dict) and part.get("type") in {
                        "reasoning_text",
                        "output_text",
                        "text",
                    }:
                        text_val = str(part.get("text", "")).strip()
                        if text_val:
                            reasoning_output.append(text_val)
            normalized_results.append(
                {
                    "type": "reasoning",
                    "tool_name": None,
                    "arguments": None,
                    "output": reasoning_output,
                }
            )

        elif itype == "message":
            parts = item.get("content", []) or []
            text_chunks: list[str] = []
            for part in parts:
                if isinstance(part, dict) and part.get("type") == "output_text":
                    text_chunks.append(str(part.get("text", "")))
            text = "\n".join(chunk for chunk in text_chunks if chunk).strip()
            if text:
                normalized_results.append(
                    {
                        "type": "output_text",
                        "tool_name": None,
                        "arguments": None,
                        "output": text,
                    }
                )

    normalized_tool_counts: dict[str, int] = {}
    for tool_name, count in (tool_usage or {}).items():
        normalized_name = (
            "search"
            if isinstance(tool_name, str) and ("retrieval" in tool_name.lower())
            else tool_name
        )
        normalized_tool_counts[normalized_name] = normalized_tool_counts.get(
            normalized_name, 0
        ) + int(count or 0)

    normalized_record = {
        "metadata": {
            "model": initial_request.get("model"),
            "reasoning": initial_request.get("reasoning"),
            "output_dir": str(out_dir),
        },
        "query_id": query_id,
        "tool_call_counts": normalized_tool_counts,
        "status": status,
        "retrieved_docids": extract_retrieved_docids_from_result(normalized_results),
        "result": normalized_results,
        "total_tokens": token_counts['total_tokens'],
        "input_tokens": token_counts['input_tokens'],
        "output_tokens": token_counts['output_tokens'],
        "elapsed_seconds": elapsed_seconds,
        "phase_tokens": phase_tokens,
        "nugget_checks": nugget_checks or [],
    }

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
    filename = os.path.join(str(out_dir), f"run_{ts}.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(normalized_record, f, indent=2, default=str)

    print("Saved response to", filename, "| tool call counts:", normalized_tool_counts, "| status:", status)


def _process_tsv_dataset(
    tsv_path: str, client: openai.OpenAI, args, tool_handler: SearchToolHandler
):
    """Process a TSV file of (id \t query) pairs sequentially and persist responses."""

    dataset_path = Path(tsv_path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"TSV file not found: {tsv_path}")

    out_dir = Path(args.output_dir).expanduser().resolve()

    queries = []
    with dataset_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue
            queries.append((row[0].strip(), row[1].strip()))

    processed_ids = set()
    if out_dir.exists():
        for json_path in out_dir.glob("run_*.json"):
            try:
                with json_path.open("r", encoding="utf-8") as jf:
                    meta = json.load(jf)
                    qid_saved = meta.get("query_id")
                    if qid_saved:
                        processed_ids.add(str(qid_saved))
            except Exception:
                continue  # ignore corrupt files

    remaining = [(qid, qtext) for qid, qtext in queries if qid not in processed_ids]

    print(
        f"Processing {len(remaining)} remaining queries (skipping {len(processed_ids)}) from {dataset_path} ..."
    )

    import threading

    completed_lock = threading.Lock()
    completed_count = [0]

    def _handle_single_query(qid: str, qtext: str, pbar=None):
        """Build request, send and persist response for one query."""

        solved = None
        elapsed_seconds = 0
        start_time = time.perf_counter()
        phase_tokens = new_phase_tokens()
        nugget_checks = []
        decomp_outputs = []
        decomp_token_infos = {'total_tokens': 0, 'input_tokens': 0, 'output_tokens': 0}
        query_decomp, decomp_output, decomp_token_info = init_query_decomp(client, qtext, args, phase_tokens)

        decomp_token_infos['total_tokens'] += decomp_token_info['total_tokens']
        decomp_token_infos['input_tokens'] += decomp_token_info['input_tokens']
        decomp_token_infos['output_tokens'] += decomp_token_info['output_tokens']
        decomp_outputs.extend(decomp_output)

        # Search every subquery first, then run a single nugget call over the
        # deduplicated union of passages instead of one call per subquery.
        pooled_results = []
        seen_docids = set()
        for idx, sub_q in enumerate(query_decomp):
            sub_results = tool_handler._search(sub_q)
            decomp_outputs.append({
                'id': f'nug-{idx}-func-call',
                'call_id': f'nug-{idx}-func-call',  # function_call uses call_id
                'arguments': sub_q,
                'name': 'search',
                'type': 'function_call',
                'status': None
            })
            decomp_outputs.append({
                'id': f'nug-{idx}-func-call',
                'call_id': f'nug-{idx}-func-call',  # function_call uses call_id
                'arguments': sub_q,
                'output': sub_results,
                'name': 'search',
                'type': 'function_call_output',
                'status': None
            })
            for cand in json.loads(sub_results):
                if cand["docid"] not in seen_docids:
                    seen_docids.add(cand["docid"])
                    pooled_results.append(cand)

        all_nuggets = []
        candidate = None
        if pooled_results:
            nuggets, early_answer, candidate, check, nug_output, nug_token_info = nugget_update_and_check(
                client, qtext, json.dumps(pooled_results, indent=2), all_nuggets, args, phase_tokens
            )
            decomp_token_infos['total_tokens'] += nug_token_info['total_tokens']
            decomp_token_infos['input_tokens'] += nug_token_info['input_tokens']
            decomp_token_infos['output_tokens'] += nug_token_info['output_tokens']
            decomp_outputs.extend(nug_output)
            all_nuggets = nuggets
            if check is not None:
                nugget_checks.append({"phase": "decomp", **check})
            if early_answer:
                elapsed_seconds = time.perf_counter() - start_time
                decomp_outputs.append({
                    "type": "message",
                    "content": [{'type': 'output_text', 'text':early_answer}]
                })
                solved = True

        user_prompt = format_query(qtext, args.query_template)
        if candidate:
            # Hand a rejected early answer to the main agent as a hypothesis to verify.
            gathered = "\n".join(f"- {n}" for n in all_nuggets)
            user_prompt += (
                "\n\nInformation gathered so far:\n" + gathered
                + f"\n\nUnverified candidate answer: {candidate}\n"
                "This candidate is not yet supported by the gathered information. Verify it with the search tool before accepting it."
            )

        initial_request = {
            "model": args.model,
            "max_output_tokens": args.max_tokens,
            "input": [
                {"role": "user", "content": user_prompt}
            ],
            "tools": tool_handler.get_tool_definitions(),
            "truncation": "auto",
            "reasoning": {"effort": args.reasoning_effort, "summary": "detailed"},
        }

        if solved is not None:
            with completed_lock:
                completed_count[0] += 1
                if pbar:
                    pbar.set_postfix(completed=completed_count[0])
            _persist_response(
                out_dir, initial_request, decomp_outputs, {}, decomp_token_infos, "completed", query_id=qid,
                elapsed_seconds=elapsed_seconds, phase_tokens=phase_tokens, nugget_checks=nugget_checks
            )
        else:
            try:
                messages, tool_usage, status, token_counts = run_conversation_with_tools(
                    client, initial_request, tool_handler, qtext, args, all_nuggets, args.max_iterations, args.verbose,
                    phase_tokens=phase_tokens, nugget_checks=nugget_checks
                )
                elapsed_seconds = time.perf_counter() - start_time

                if status == "completed":
                    with completed_lock:
                        completed_count[0] += 1
                        if pbar:
                            pbar.set_postfix(completed=completed_count[0])
                decomp_outputs.extend(messages)
                messages = decomp_outputs
                token_counts['total_tokens'] += decomp_token_infos['total_tokens']
                token_counts['input_tokens'] += decomp_token_infos['input_tokens']
                token_counts['output_tokens'] += decomp_token_infos['output_tokens']
                _persist_response(
                    out_dir, initial_request, messages, tool_usage, token_counts, status, query_id=qid,
                    elapsed_seconds=elapsed_seconds, phase_tokens=phase_tokens, nugget_checks=nugget_checks
                )

            except Exception as exc:
                print(f"[Error] Query id={qid} failed: {exc}")
                sys.exit(1)

    if args.num_threads <= 1:
        with tqdm(remaining, desc="Queries", unit="query") as pbar:
            for qid, qtext in pbar:
                _handle_single_query(qid, qtext, pbar)
    else:
        with (
            ThreadPoolExecutor(max_workers=args.num_threads) as executor,
            tqdm(total=len(remaining), desc="Queries", unit="query") as pbar,
        ):
            futures = [
                executor.submit(_handle_single_query, qid, qtext, pbar)
                for qid, qtext in remaining
            ]

            for _ in as_completed(futures):
                pbar.update(1)


def main():
    parser = argparse.ArgumentParser(
        description="Call vLLM OpenAI Responses API with native function calling and local search.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--query",
        default="topics-qrels/queries.tsv",
        help="User query *or* TSV file path",
    )
    parser.add_argument(
        "--model", default="openai/gpt-oss-20b", help="Model name served by vLLM"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=10000,
        help="max_output_tokens for Responses API",
    )
    parser.add_argument(
        "--output-dir", default="runs_vllm", help="Directory to store run JSON files"
    )
    parser.add_argument(
        "--query-template",
        choices=[
            "QUERY_TEMPLATE",
            "QUERY_TEMPLATE_NO_GET_DOCUMENT",
            "QUERY_TEMPLATE_NO_GET_DOCUMENT_NUGGET",
            "QUERY_TEMPLATE_NO_GET_DOCUMENT_NO_CITATION",
        ],
        default="QUERY_TEMPLATE_NO_GET_DOCUMENT",
        help="Specify the query template to use (default: %(default)s)",
    )
    parser.add_argument(
        "--num-threads", type=int, default=1, help="Parallel threads for dataset mode"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=100,
        help="Max conversation rounds with function calls",
    )
    parser.add_argument(
        "--max-nugget-adds",
        type=int,
        default=5,
        help="Max information units kept from a single nugget update call",
    )
    parser.add_argument(
        "--max-nuggets",
        type=int,
        default=15,
        help="Max information units kept in the gathered list (oldest dropped first)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--reasoning-effort",
        default="high",
        choices=["low", "medium", "high"],
        help="Reasoning effort",
    )
    parser.add_argument(
        "--model-url", default="http://localhost:8000/v1", help="Model URL"
    )

    # Searcher selection and shared tool options --------------------------
    parser.add_argument(
        "--searcher-type",
        choices=SearcherType.get_choices(),
        required=True,
        help=f"Type of searcher to use: {', '.join(SearcherType.get_choices())}",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        help="Hugging Face token for accessing private datasets/models",
    )
    parser.add_argument(
        "--hf-home", type=str, help="Hugging Face home directory for caching"
    )

    parser.add_argument(
        "--snippet-max-tokens",
        type=int,
        default=512,
        help="Tokens per document snippet",
    )
    parser.add_argument(
        "--k", type=int, default=5, help="Top-k search results to return for each query"
    )
    parser.add_argument(
        "--get-document",
        action="store_true",
        help="Also register the get_document tool",
    )

    temp_args, _ = parser.parse_known_args()
    searcher_class = SearcherType.get_searcher_class(temp_args.searcher_type)
    searcher_class.parse_args(parser)

    args = parser.parse_args()

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = args.hf_token
    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home

    client = openai.OpenAI(
        base_url=args.model_url,
        api_key=os.environ['IDA_LLM_API_KEY'],
    )

    searcher = searcher_class(args)

    tool_handler = SearchToolHandler(
        searcher=searcher,
        snippet_max_tokens=args.snippet_max_tokens,
        k=args.k,
        include_get_document=args.get_document,
    )

    if isinstance(args.query, str):
        qstr = args.query.strip()
        if qstr.lower().endswith(".tsv"):
            potential_path = Path(qstr)
            try:
                if potential_path.is_file():
                    print("Processing TSV dataset", potential_path)
                    _process_tsv_dataset(
                        str(potential_path), client, args, tool_handler
                    )
                    return
            except OSError:
                pass

    print("Processing single query", args.query)
    qtext = args.query
    args.query = format_query(args.query, args.query_template)

    messages = [{"role": "user", "content": args.query}]

    initial_request = {
        "model": args.model,
        "max_output_tokens": args.max_tokens,
        "input": messages,
        "tools": tool_handler.get_tool_definitions(),
        "truncation": "auto",
        "reasoning": {"effort": args.reasoning_effort, "summary": "detailed"},
    }

    messages, tool_usage, status = run_conversation_with_tools(
        client, initial_request, tool_handler, args.query, args, [], args.max_iterations, args.verbose
    )

    _persist_response(
        args.output_dir, initial_request, messages, tool_usage, status, query_id=None
    )

    rprint(messages)


if __name__ == "__main__":
    main()
