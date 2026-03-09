GENERATE_POSITIVE_PROMPT = """
You are an expert job applicant. Your task is to generate a realistic Curriculum Vitae (CV) text that is a **MATCH (Positive Sample)** for the provided Job Description.

Job Description:
{jd}

### Instructions:
1.  **Requirement Adherence**: The candidate MUST possess ALL "Required/Mandatory" skills listed in the JD. You may omit one "Nice-to-have" or "Bonus" skill if present, but NEVER omit a core requirement.
2.  **Semantic Paraphrasing (CRITICAL)**: Do NOT copy phrases directly from the JD. Use synonyms and different sentence structures.
    *   *Example:* If JD says "Proficient in Python for backend", CV should say "Backend development experience using Django and Flask".
    *   *Example:* If JD says "Collaborate with cross-functional teams", CV should say "Worked closely with design and product counterparts".
3.  **Realism & Noise**: 
    *   Include exactly 1-2 minor typos (e.g., "mangement" instead of "management").
    *   Use bullet points that vary in length (some detailed, some brief).
    *   Dates should be realistic (no overlapping impossible timelines).
4.  **Language**: The output must be strictly in {output_language}.
"""

GENERATE_HARD_NEGATIVE_PROMPT = """
You are an expert Technical Recruiter acting as an "Adversarial Generator." 
Your task is to take a specific Job Description (JD) and a Matching CV, and modify the CV to create a **HARD NEGATIVE (Rejection)**.

Job Description:
{jd}

Original Matching CV:
{positive_cv_text}

### Instructions:

1. **The Goal**:  
The resulting CV must look like a strong match at first glance (high keyword overlap) but must be rejected upon closer inspection due to a specific "Deal-Breaker."

2. **Modification Strategy (USER-SELECTED)**:  
Apply ONLY the following strategy: **{poison_strategy}**

Do NOT apply any other strategies.

Definitions:

- **Strategy A (Seniority Mismatch)**:  
If JD asks for Senior/Lead, downgrade the CV's experience to Junior/Entry-level (e.g., reduce years, remove leadership tasks) BUT keep the tech stack identical.

- **Strategy B (Stack Mismatch)**:  
Identify the #1 most critical technical skill (e.g., Java). Replace it consistently with a competing technology (e.g., C#/.NET) throughout the CV, while keeping domain knowledge unchanged.

- **Strategy C (Role Mismatch)**:  
Keep technical keywords but change the candidate's role focus (e.g., if JD is "React Developer", make the CV a "Project Manager" or "Recruiter" who managed React projects but did not implement them).

You MUST follow exactly the selected strategy and no others.

3. **Constraint**:  
Keep at least 80% of the original CV text identical to preserve high vector similarity.

4. **Language**:  
Output must be strictly in {output_language}.


"""

GENERATE_PROFILE_PROMPT = """
You are an expert career coach. Your task is to generate a **Professional Profile Summary** (the introductory paragraph of a resume) for a candidate who is a perfect match for the provided Job Description.

Job Description:
{jd}

### Instructions:
1.  **Format**: Write exactly **one paragraph** (3-5 sentences). Do NOT use bullet points or lists.
2.  **Content Synthesis**: Weave the JD's "Must-Have" requirements into a coherent career narrative.
    *   Focus on the candidate's total years of experience, primary tech stack, and main value proposition.
    *   *Example:* Instead of listing tools, write "Seasoned DevOps Engineer with a strong track record in automating cloud infrastructure using [Tool A] and [Tool B]."
3.  **Differentiation**: This text must serve as a high-level overview, distinct from the granular details found in a work history section. Avoid specific dates or company names.
4.  **Language**: The output must be strictly in {output_language}.
"""