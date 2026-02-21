# Sentinel — 10 Week Milestone Plan

| Week | Drift Module (Manish) | Bias Module (Aryan) | Adversarial Module (Hardik) | Integration / Shared |
|------|--------------------------|-------------------------|--------------------------------|----------------------|
| W1 | Define drift objectives + select AG News dataset | Select CrowS-Pairs dataset + define bias metrics | Select TruthfulQA dataset + define robustness goals | Create repo + project architecture |
| W2 | Implement drift module skeleton | Implement bias module skeleton | Implement adversarial module skeleton | Define common LLM client interface |
| W3 | Build AG News loader + baseline label stats | Implement CrowS-Pairs loader | Implement TruthfulQA loader | Add config system (YAML) |
| W4 | Implement basic statistical drift metrics (label distribution, avg length) | Implement bias scoring logic | Implement basic truthfulness evaluation | Build combined JSON report structure |
| W5 | Add threshold logic + pass/fail status | Add per-category bias breakdown | Add prompt variation testing | Create unified pipeline runner |
| W6 | Refactor drift for modular metrics | Add second bias benchmark (optional stretch) | Add adversarial prompt variations | Improve report formatting |
| W7 | Implement embedding-based drift (stretch) | Add fairness visualization support | Add jailbreak test prompts | Dockerize project |
| W8 | Run full validation against ≥2 model endpoints | Compare bias metrics across models | Compare robustness across models | Improve CLI UX |
| W9 | Performance improvements + cleanup | Documentation + examples | Documentation + examples | Reproducibility testing |
| W10 | Final validation runs + threshold tuning | Final bias evaluation + interpretation | Final robustness evaluation + interpretation | Final report generation + submission prep |