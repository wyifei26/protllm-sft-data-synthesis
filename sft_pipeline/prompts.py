from __future__ import annotations

import json
from typing import Any

from .config import RunConfig

DEFAULT_STAGE1_FIELD_PROMPT = (
    "Write a concise English summary that synthesizes the curated annotation into a biologically "
    "useful description rather than a literal rewrite. Keep it faithful to the provided details, "
    "avoid database identifiers, and present the result as a coherent paragraph."
)

STAGE1_FIELD_PROMPTS: dict[str, str] = {
    "secondary": (
        "Describe the protein secondary structure in a fluent, continuous English paragraph. "
        "Do not mention any absolute residue positions or numeric ranges. Instead, summarize the "
        "composition in terms of the number and relative length of structural elements, such as how "
        "many helices are short, medium, or long, how many beta-strands or beta-sheets are present, "
        "and how many turn regions exist. Then describe how these elements are arranged relative to "
        "each other in the sequence, focusing on their overall organization and interleaving pattern, "
        "such as whether helices dominate, whether beta-elements are scattered or clustered, and how "
        "turns connect different elements. Emphasize the global folding architecture and connectivity "
        "rather than residue-level detail. Keep the response concise and written as a single coherent paragraph."
    ),
    "interpro": (
        "Describe the InterPro annotations as an integrated interpretation of the protein's likely "
        "identity, evolutionary classification, and domain architecture. Reconcile overlapping family, "
        "domain, and superfamily labels into a clean explanation of what kind of protein this is and "
        "what structural or functional modules it likely contains. Highlight whether the annotations "
        "support a single dominant module, repeated modules, or a multi-domain organization. Do not list "
        "database accessions or simply restate names one by one; instead, explain what the annotations imply "
        "in concise, natural English."
    ),
    "cath": (
        "Describe the CATH evidence as a high-level account of the protein's structural class and fold "
        "architecture. Explain whether the annotated regions are mainly alpha, mainly beta, or alpha/beta, "
        "what overall architecture or topology is suggested, and whether the entries indicate one dominant "
        "fold or several structurally distinct domains. Translate the classification into readable structural "
        "biology language instead of copying the CATH string verbatim, and avoid CATH identifiers."
    ),
    "pfam": (
        "Describe the Pfam annotations as a concise summary of the domain and family composition of the protein. "
        "Explain what kinds of modules are present, whether they are catalytic, binding, regulatory, repeat-based, "
        "or otherwise characteristic, and whether the architecture appears single-domain or multi-domain. "
        "Collapse redundant family names into a coherent description rather than listing them separately, and avoid "
        "Pfam accessions."
    ),
    "go": (
        "Describe the Gene Ontology evidence as a compact functional profile of the protein. Integrate molecular "
        "function, biological process, and cellular component annotations into a natural English paragraph that "
        "explains what the protein likely does, where it acts, and what broader process or context it belongs to. "
        "Interpret GO category labels and prefixes instead of reproducing them, and avoid GO identifiers."
    ),
    "disease": (
        "Describe the disease annotation as a concise clinical association summary. State what disorder or syndrome "
        "the protein is linked to, the major affected systems or hallmark phenotypes, and the inheritance pattern if "
        "it is provided. If multiple disease annotations are present, indicate whether they represent distinct or "
        "overlapping phenotypic associations. Compress long phenotype lists into the key medical picture, avoid "
        "disease accessions, and do not overstate causality beyond the evidence."
    ),
    "function": (
        "Describe the protein's function in concise, biologically informative English. Explain the main biochemical "
        "activity or biological role, including important substrates, products, targets, or mechanistic features when "
        "they are stated or strongly implied by standard protein terminology. Merge multiple function notes into one "
        "coherent account, avoid PubMed IDs, and do not merely paraphrase the raw text."
    ),
    "domain": (
        "Describe the domain organization of the protein as a functional architectural summary. Explain what notable "
        "domains or regions are present and what they contribute, such as catalysis, binding, localization, membrane "
        "association, regulation, or signaling. Reconcile overlapping or repetitive entries into a clean description "
        "of how the parts of the protein likely work together, and avoid residue ranges unless absolutely necessary."
    ),
    "subunit": (
        "Describe the quaternary structure and interaction behavior of the protein in a concise paragraph. Explain "
        "whether it is monomeric, self-associating, or part of heteromeric assemblies, note key binding partners or "
        "interaction classes when they are important, and mention functional consequences of assembly if the evidence "
        "supports it. Prioritize biologically meaningful interaction patterns over exhaustive partner lists, and avoid "
        "PubMed IDs."
    ),
    "catalytic": (
        "Describe the catalytic annotation as readable biochemical chemistry rather than equation fragments. Explain "
        "the overall reaction type, the main substrate and product relationships, and any evidence for alternative "
        "products, side activities, or catalytic promiscuity. Use enzyme knowledge to turn shorthand reactions into "
        "clear English, while staying faithful to the annotation and avoiding raw identifiers unless they add essential "
        "biological meaning."
    ),
}


def _format_stage1_field_prompts(fields: tuple[str, ...]) -> str:
    return "\n".join(
        f"- {field}: {STAGE1_FIELD_PROMPTS.get(field, DEFAULT_STAGE1_FIELD_PROMPT)}"
        for field in fields
    )


def build_stage1_messages(record: dict[str, Any], config: RunConfig) -> list[dict[str, str]]:
    payload = {
        "primary_accession": record["primary_accession"],
        "release_date": record["release_date"],
        "fields": {
            field: {
                "details": record[field].get("details", []),
            }
            for field in config.summary_fields
        },
    }
    field_instructions = _format_stage1_field_prompts(config.summary_fields)
    system = (
        "You are building concise English metadata summaries for protein SFT training. "
        "Use the provided annotation details as evidence, but do not merely paraphrase or concatenate them. "
        "Apply standard protein, structural biology, and functional annotation knowledge to explain what the "
        "annotations imply at a higher level while staying faithful to the evidence. Return strict JSON only. "
        "Do not invent facts. If a field has no details, write a short absence summary."
    )
    user = (
        "Write one concise but information-dense English summary for each metadata field so the results are useful "
        "for downstream protein QA synthesis.\n"
        "General requirements:\n"
        "1. Follow the field-specific instruction for each field listed below.\n"
        "2. Synthesize, interpret, and deduplicate the evidence instead of rewriting the raw entries item by item.\n"
        "3. Stay grounded in the provided annotations and standard domain knowledge; do not invent unsupported facts.\n"
        "4. Avoid database accessions, PubMed IDs, and other raw identifiers unless the field-specific instruction makes them biologically necessary.\n"
        "5. Do not mention that you are summarizing.\n"
        "6. If a field has no details, write a short absence summary.\n"
        "7. Output JSON with this schema:\n"
        '{"primary_accession": "...", "summaries": {"secondary": "...", "interpro": "...", "...": "..."}}'
        "\nField-specific instructions:\n"
        f"{field_instructions}"
        "\nMetadata JSON:\n"
        f"{json.dumps(payload, ensure_ascii=False)}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_stage2_messages(
    record: dict[str, Any],
    summary_record: dict[str, Any],
    config: RunConfig,
) -> list[dict[str, str]]:
    prompt_payload = {
        "primary_accession": record["primary_accession"],
        "release_date": record["release_date"],
        "sequence_available": record["sequence_available"],
        "backbone_available": record["backbone_available"],
        "placeholders": {
            "sequence": config.sequence_placeholder,
            "structure": config.structure_placeholder,
        },
        "summaries": summary_record["summaries"],
    }
    system = (
        "You generate English QA records for protein supervised fine-tuning. "
        "Return strict JSON only. Be faithful to the metadata, keep the wording natural and diverse, "
        "and use the provided sequence and structure placeholders literally. Make the questions meaningfully "
        "different from each other in phrasing, intent, and emphasis."
    )
    user = (
        "Create exactly four QA records for this protein: t2bb, t2s, bb2t, and s2t.\n"
        "Task rules:\n"
        "- t2bb: question is an English structural design request; answer is the structure placeholder only.\n"
        "- t2s: question is an English sequence design request; answer is the sequence placeholder only.\n"
        "- bb2t: question is a natural English query or instruction that includes the structure placeholder; answer is a concise metadata-grounded description.\n"
        "- s2t: question is a natural English query or instruction that includes the sequence placeholder; answer is a concise metadata-grounded description.\n"
        "General rules:\n"
        "1. Generate all four tasks.\n"
        "2. Use the placeholders literally; never ask for or invent real tokens.\n"
        "3. Make the questions diverse across tasks: vary phrasing, task framing, requested emphasis, and level of specificity.\n"
        "4. Avoid repeating the same opening pattern or simply swapping placeholders into the same sentence template.\n"
        "5. Let different questions focus on different aspects when appropriate, such as fold, domain architecture, biochemical function, localization, interactions, or design constraints.\n"
        "6. Keep the outputs useful for training and do not include chain-of-thought.\n"
        "7. Output JSON with schema:\n"
        '{"primary_accession":"...","tasks":[{"task_type":"t2bb","status":"ok","question":"...","answer":"...","evidence_fields":["secondary","domain"]}]}\n'
        "Protein metadata:\n"
        f"{json.dumps(prompt_payload, ensure_ascii=False)}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def dry_run_stage1(record: dict[str, Any], config: RunConfig) -> dict[str, Any]:
    summaries: dict[str, str] = {}
    for field in config.summary_fields:
        details = record[field].get("details", [])
        if not details:
            summaries[field] = f"No curated {field} annotation is available for this protein."
            continue
        texts = [item.get("text", "").strip() for item in details if item.get("text")]
        summaries[field] = " ; ".join(texts[:2])[:400]
    return {"primary_accession": record["primary_accession"], "summaries": summaries}


def dry_run_stage2(record: dict[str, Any], summary_record: dict[str, Any], config: RunConfig) -> dict[str, Any]:
    description = " ".join(
        summary_record["summaries"][field]
        for field in ("function", "domain", "go", "interpro")
        if summary_record["summaries"].get(field)
    )[:800]
    tasks = []
    for task_type in config.stage2_tasks:
        if task_type in {"t2s", "s2t"} and not record["sequence_available"]:
            tasks.append(
                {
                    "task_type": task_type,
                    "status": "pending_sequence_fill",
                    "question": "",
                    "answer": "",
                    "evidence_fields": [],
                }
            )
            continue
        if task_type == "t2s":
            question = f"Design a protein sequence for accession {record['primary_accession']} that matches this description: {description}"
            answer = config.sequence_placeholder
        elif task_type == "s2t":
            question = (
                "Given the following protein sequence tokens, describe the protein's likely "
                f"function, domains, and relevant properties in English: {config.sequence_placeholder}"
            )
            answer = description
        elif task_type == "t2bb":
            question = f"Design a protein backbone for accession {record['primary_accession']} consistent with this description: {description}"
            answer = config.structure_placeholder
        else:
            question = (
                "Given the following protein structure tokens, describe the protein's likely "
                f"architecture, domains, and biological characteristics in English: {config.structure_placeholder}"
            )
            answer = description
        tasks.append(
            {
                "task_type": task_type,
                "status": "ok",
                "question": question,
                "answer": answer,
                "evidence_fields": ["function", "domain", "go", "interpro"],
            }
        )
    return {"primary_accession": record["primary_accession"], "tasks": tasks}
