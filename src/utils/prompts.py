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
1.  **The Goal**: The resulting CV must look like a strong match at first glance (high keyword overlap) but must be rejected upon closer inspection due to a specific "Deal-Breaker."
2.  **Modification Strategy**: Choose EXACTLY ONE of the following strategies to "poison" the CV. Do not apply multiple strategies; we want to test specific failure modes.
    *   **Strategy A (Seniority Mismatch)**: If JD asks for Senior/Lead, downgrade the CV's experience to Junior/Entry-level (e.g., reduce years, remove leadership tasks) BUT keep the tech stack identical.
    *   **Strategy B (Stack Mismatch)**: Identify the #1 most critical technical skill (e.g., Java). Change it to a competitor technology (e.g., C#/.NET) throughout the CV, but keep the domain knowledge (e.g., Backend logic) the same.
    *   **Strategy C (Role Mismatch)**: Keep the technical keywords but change the candidate's focus. (e.g., If JD is "React Developer", make the CV a "Tech Recruiter" or "Project Manager" who *managed* React projects but didn't code).

3.  **Constraint**: Keep at least 80% of the original CV text identical. We want the vector similarity to remain high.
4.  **Language**: The output must be strictly in {output_language}.
"""