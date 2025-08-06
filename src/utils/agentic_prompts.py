"""Prompts for multi-agent debate-based capability generation."""

SCIENTIST_AREA_PROMPT = (
    "You are Scientist {role}, an expert in designing capabilities to assess the abilities of foundation models within the {domain} domain.\n\n"
    "Your task is to propose {n} non-overlapping areas that comprehensively cover the domain **{domain}**.\n\n"
    "Consider:\n"
    "- Core theoretical foundations and fundamental concepts\n"
    "- Practical applications and real-world scenarios\n"
    "- Advanced techniques and specialized knowledge\n"
    "- Emerging trends and cutting-edge developments\n\n"
    "Each area should be broad enough to encompass multiple related capabilities while being specific enough to provide clear focus.\n\n"
    "Return JSON with `areas` (list) and `finalized:false`. "
    "Each area element: {{`id,int`,`name,str`,`description,str(<30w)`}}.\n\n"
    "Be thorough and ensure areas don't overlap but collectively cover the full domain scope."
)

SCIENTIST_CAP_PROMPT = (
    "You are Scientist {role}, an expert in designing capabilities to assess the abilities of foundation models.\n\n"
    "For area **{area_name}** (id={area_id}) within the {domain} domain, propose {n} specific capabilities that represent measurable skills.\n\n"
    "Each capability should:\n"
    "- Be testable with concrete problems and clear evaluation criteria\n"
    "- Represent a distinct skill or knowledge area within the broader area\n"
    "- Have well-defined success criteria and failure modes\n"
    "- Progress from basic to advanced concepts within the area\n"
    "- Be designed according to the METR Standard format\n\n"
    "Return JSON with `capabilities` (list) and `finalized:false`. "
    "Each capability element: {{`id,int`,`name,str`,`description,str(<30w)`}}."
)

SCIENTIST_TASK_PROMPT = (
    "You are Scientist {role}, an expert in designing tasks for capability evaluation.\n\n"
    "For capability **{cap_name}** (id={cap_id}) within the {domain} domain, create {n} high-quality evaluation tasks.\n\n"
    "Requirements:\n"
    "- Tasks should test the specific capability thoroughly and comprehensively\n"
    "- Include varying difficulty levels (easy, medium, hard) to assess different skill levels\n"
    "- Provide clear, unambiguous problems with precise, verifiable answers\n"
    "- Cover different aspects and dimensions of the capability\n"
    "- Ensure tasks are diverse in format and approach\n\n"
    "Return JSON with `tasks` (list) and `finalized:false`. "
    "Each task element: {{`id,str`,`problem,str`,`answer,str`,`difficulty,str`}}."
)

MODERATOR_PROMPT = (
    "You are the Moderator facilitating a scientific debate between expert researchers in capability design.\n\n"
    "Your role is to synthesize PROPOSAL_A and PROPOSAL_B into a single, high-quality, deduplicated result that represents the best elements from both proposals.\n\n"
    "Guidelines:\n"
    "- Merge complementary ideas and perspectives\n"
    "- Remove duplicates and redundancies while preserving unique contributions\n"
    "- Preserve the best elements from both proposals\n"
    "- Ensure logical consistency, completeness, and coherence\n"
    "- Maintain the required JSON structure and format\n"
    "- If both scientists approve in the next round, set `finalized:true`\n\n"
    "PROPOSAL_A = ```{proposal_a}```\n\n"
    "PROPOSAL_B = ```{proposal_b}```\n\n"
    "Provide your synthesized result in the same JSON format."
)

SCIENTIST_REVIEW_PROMPT = (
    "You are Scientist {role}, an expert in capability design and evaluation.\n\n"
    "Review the MODERATOR'S DRAFT below with careful attention to:\n"
    "- Completeness and comprehensive coverage of the domain/area/capability\n"
    "- Accuracy and correctness of the proposed elements\n"
    "- Logical organization and coherence of the structure\n"
    "- Missing important elements or gaps in coverage\n"
    "- Quality and feasibility of the proposed capabilities/tasks\n\n"
    "If satisfied with the draft, reply **APPROVE**. Otherwise, provide specific feedback "
    "and an improved JSON version that addresses any identified issues.\n\n"
    "DRAFT = ```{draft}```"
)
