GENERATE_POSITIVE_PROMPT = """You are an expert Technical Recruiter and Data Extractor. 
I will provide you with a Job Description (JD). Your task is to invent a "Perfect Match" candidate profile and output it STRICTLY in JSON format.

The output must be in {output_language}.

Job Description:
{jd}

### Instructions:
1. Extract the core requirements, skills, and domain expertise from the JD.
2. Fill out the candidate JSON profile so that this person is a 100% perfect fit for the role.
3. For the `work_history` bullet points, write 3-4 realistic, action-oriented achievements that prove they have the required skills. Use the past tense.
4. Do NOT hallucinate skills that conflict with the JD. If the JD asks for Python, do not make their primary experience in C# unless C# is also mentioned.
5. The output must strictly follow the provided JSON schema. Do not include markdown formatting like ```json or any conversational text.
6. Simulate a realistic candidate profile, include typos (e.g., "Enginner" instead of "Engineer") and grammatical errors.
7. Do NOT mirror the JD's exact phrasing. Describe skills and achievements 
   using different vocabulary than the JD uses (e.g., if JD says 
   "distributed systems", the CV might say "large-scale infrastructure").
"""

GENERATE_HARD_NEGATIVE_PROMPT_A = """
You are an adversarial Technical Recruiter. 
I am going to provide you with a Job Description (JD) and a JSON profile of a candidate who is a PERFECT MATCH for that JD.

Your task is to modify the provided Candidate JSON to create a **HARD NEGATIVE (Rejection)** candidate based STRICTLY on the selected strategy below.

Job Description:
{jd}

Perfect Candidate JSON:
{positive_cv_text}

### Selected Dealbreaker Strategy: {poison_strategy}

### Instructions:
1. Apply the selected strategy logically across the entire JSON.
2. Output the modified profile STRICTLY in the exact same JSON schema. No markdown, no conversational text.
3. The output must be strictly in {output_language}.
"""

GENERATE_HARD_NEGATIVE_PROMPT_B = """
You are an adversarial Technical Recruiter. 
I am going to provide you with a Job Description (JD). 

Your task is to generate a **HARD NEGATIVE (Near-Miss)** candidate from scratch. This candidate must look incredibly strong for the role, but possess a fatal Dealbreaker.

Job Description:
{jd}

### Selected Dealbreaker Strategy: {poison_strategy}

### Instructions for a "Near Miss":
1. The candidate's background must be completely original (invent new companies, universities, projects, and hobbies).
2. To ensure they are a "Hard" Negative, they MUST perfectly match the JD on the following dimensions:
   - They must have the exact same Seniority / Years of Experience (unless the Dealbreaker modifies this).
   - They must work in the exact same Industry/Domain.
   - They must possess all the secondary/bonus tools (e.g., CI/CD, Agile, Cloud) listed in the JD.
3. They MUST explicitly fail the JD based on the selected Dealbreaker Strategy. 
4. Output strictly in the defined JSON schema.
5. The output must be strictly in {output_language}.
"""

DEALBREAKER_A = """
**STRATEGY A: DYNAMIC SENIORITY MISMATCH (THE OPPOSITE LEVEL TRAP)**

Your goal is to make this candidate fundamentally unqualified by drastically flipping their seniority level to the exact OPPOSITE of what the Job Description requires, while keeping their technical knowledge identical.

**Execution Steps:**
1. **Identify & Flip:** Determine the seniority of the provided Perfect Candidate JSON. 
   - If the candidate is Mid/Senior/Lead -> **Downgrade to Junior/Entry-Level.**
   - If the candidate is Junior/Entry-Level -> **Upgrade to Principal/Director/Lead.**
2. **Title & Level:** Update the `headline_title`, `seniority_level`, and `role_title` in `work_history` to explicitly state this new opposite seniority (e.g., prepending "Junior" or "Principal").
3. **Years of Experience (YOE):** 
   - If downgraded to Junior: Reduce `total_yoe` to 0-2 years.
   - If upgraded to Principal: Increase `total_yoe` to 10+ years. 
   - Adjust the `years_in_role` within the `work_history` array to match this new total. (Add or remove past jobs as necessary).
4. **Rewrite Action Verbs (CRITICAL):** Rewrite EVERY bullet point in `work_history` to reflect the new seniority mindset:
   - *If making Junior:* Remove leadership/architecture verbs. Use support verbs (e.g., "Assisted in", "Maintained", "Wrote unit tests for", "Shadowed senior engineers"). Reduce the scope of their impact to bug fixes and minor features.
   - *If making Principal/Lead:* Remove basic individual-contributor tasks. Use high-level verbs (e.g., "Directed", "Architected", "Defined the multi-year technical strategy for", "Managed a department of"). Shift the focus away from writing code to high-level system design and management.
5. **Strict Preservation:** Do NOT change the `core_tech_stack`, `tools_and_frameworks`, or `domain_expertise`. They must still work in the exact same domain using the exact same tools—just at the completely wrong level of responsibility.
"""

DEALBREAKER_B = """
**STRATEGY B: TECH STACK ECOSYSTEM MISMATCH**

Your goal is to make this candidate fundamentally unqualified by swapping their primary technical ecosystem for a competing one, while keeping their domain expertise and seniority exactly the same.

**Execution Steps:**
1. **Identify & Swap:** Identify the #1 most critical programming language/framework required by the JD (e.g., C++, Java, Python, React). Choose a direct, incompatible competitor (e.g., if JD wants Java/Spring, swap to C#/.NET. If JD wants React, swap to Angular).
2. **Update Arrays:** Replace the target language and its associated frameworks in `core_tech_stack` and `tools_and_frameworks`. (e.g., if you swap Python to Ruby, ensure you also swap Django/FastAPI to Ruby on Rails). 
3. **Rewrite Bullet Points (CRITICAL):** Update the `bullet_points` in the `work_history` to reflect the new tech stack. 
   - *Example:* "Developed high-performance C++ engines" MUST become "Developed high-performance C# backend engines".
4. **Title Update:** If the `headline_title` includes the technology (e.g., "C++ Developer"), change it to the new technology ("C# Developer").
5. **Strict Preservation:** Keep the `total_yoe`, `seniority_level`, and `domain_expertise` COMPLETELY IDENTICAL to the positive JSON. The candidate must have achieved the exact same business impact in the exact same industry, just using the wrong tools.
"""

DEALBREAKER_C = """
**STRATEGY C: ROLE FUNCTION MISMATCH (THE ADJACENT PROFESSIONAL)**

Your goal is to change the candidate's core job function. They must no longer be a Hands-On Engineer/Practitioner. Instead, make them a "Scrum Master", "Technical Recruiter", "Product Owner", or "Project Manager" who works *alongside* these technologies but does not write code.

**Execution Steps:**
1. **Change the Role:** Change `headline_title` and the `role_title` in `work_history` to an adjacent non-coding role (e.g., "Technical Recruiter", "Agile Scrum Master", "IT Project Manager").
2. **Preserve Tech Arrays:** Keep the exact same technologies in `core_tech_stack` and `tools_and_frameworks`. (We want the JSON to still contain these keywords to act as a vector trap).
3. **Rewrite Bullet Points (CRITICAL):** Rewrite the `bullet_points` so they include the technical keywords, but grammatically prove the candidate did not do the engineering.
   - *If Recruiter:* "Sourced, interviewed, and hired senior engineers specializing in [TECH STACK]."
   - *If Scrum Master:* "Facilitated Agile sprints for a team of 10 developers building [TECH STACK] microservices in the [DOMAIN] industry."
   - *If Project Manager:* "Managed project timelines and budget for the migration of [TECH STACK] infrastructure."
4. **Strict Preservation:** Keep the `domain_expertise` and the `company_type` identical. They work in the exact same field, just in the wrong job.
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