# Benchmark Specification — v1.1.1 (FROZEN 2026-10-08; `docs/FREEZE_v1.1.json` → amendment `docs/FREEZE_v1.1.1.json`)

**Paper:** *When Is a Typed Decision Model Enough? Prevalence Shift, Label Budgets, and Decision Training*
**Amendment v1.1.1 (2026-10-08, kisi final result se pehle; approved by the authors):** teen independent audits (prediction, training, analysis) ke baad: Exp 2/PA1 no-shift control resampled, PA bootstrap B = 10,000, DeBERTa failed-run threshold candidate-restricted, Jev transport errors kabhi block nahi, `exp1` command, H-A2 ka scope tang. Tafseel `docs/deviations.md`. Decoder runs (`final_determinism`, `final_local`) v1.1 code par hain aur in tabdeeliyon se mutasir nahi (un ka code nahi badla); `verify-results --freeze docs/FREEZE_v1.1.json`. DeBERTa, TF-IDF aur Jev runs v1.1.1 par.
**Status v1.1, FROZEN 2026-10-08 (approved by the authors):** is file, configs, data manifests aur code ke sha256 `docs/FREEZE_v1.1.json` mein; is ke baad har tabdeeli `docs/deviations.md` mein. Pilot (sec 17) poora; saare ⏳ items resolve ya Stage 2/3 amendment ko assign; authors' decisions 2026-10-08. Baqi: Jev cost (item 12). OSF/preregistration nahi hogi (authors' decision 2026-10-08): internal freeze (sec 16). Pilot ki tafseel internal (`docs/pilot_notes.md`), paper ka hissa nahi. Approval ke baad: freeze + preregistration (sec 16).
**Pichla status:** v1.0.3-draft, 2026-10-06 (v1.0.3: stages; core 7 models; gpt-oss aur GLiNER secondary; Pillar A ki implicit-prior framing). Abhi **protocol draft** hai, final nahi. v1.0.1: papers se verification fixes; v1.0.2: methodology review (label-shift aur label-budget papers ke saath) ke fixes. Dono `docs/spec_verification.md` mein. **Experiment numbering v1.0.2 mein sequential ki gayi** (purana → naya: 1→1, 8→2, 4→3, 5→4, 2→5, 3→6); `research_idea.md` mein purane numbers hain. Pilot (section 17) ke baad v1.1 mein update hoga; phir **freeze** (section 16; internal, OSF nahi). Freeze ke baad har tabdeeli `docs/deviations.md` mein tareekh aur wajah ke saath likhni hai.
**Background:** `docs/research_idea.md` (v13), `docs/related_work_gap_analysis.md`, `docs/model_review.md`, `docs/dataset_review.md`.

Is document mein **"⏳ pilot"** ka matlab: value pilot ke baad pakki hogi. **"🔒"** ka matlab: freeze ke baad nahi badlegi.

---

## 0. Paper ek page mein

### 0.1 Idea
Typed decision models (TDMs), jinhein "System One" models bhi kaha jata hai (Jev, Tev1, Nimble, GLiNER2.5-Decide), text generate nahi karte: caller ke diye options par seedha probability distribution dete hain. Yeh saste aur tez hain. Sawal yeh hai ke **asli deployment mein yeh kab kaafi hain**:
- jab asli data ki class distribution test data se alag ho (**prevalence shift**), aur
- jab company ke paas kuch labels hon jin se apna model train ho sake (**label budget**),
- aur jo "decision training" vendors bech rahe hain, woh plain LLM ke muqable mein asal mein kya deti hai.

**Title:** *When Is a Typed Decision Model Enough? Prevalence Shift, Label Budgets, and Decision Training*
**Terminology:** paper mein "Typed Decision Model (TDM)"; abstract ki pehli line mein ek dafa: *"Typed decision models (TDMs), marketed as "System One" models, ..."*
**Venue:** TMLR (pehle arXiv preprint).

### 0.2 Kyun (gap)
Jev 15 Sep 2026 ko release hua, aur teen hafton mein 30+ papers aaye. Sab se qareebi kaam:

| Paper | Kya kiya | Kya chhoda (hamara gap) |
|---|---|---|
| **Rafe & Das** (arXiv 2610.00346) | 8 TDMs vs LLMs vs trained classifiers; calibration, OOS, option naming/count, cascades, Nimble ↔ Qwen pair | Prevalence shift nahi; label budget sirf "labels hon to classifier jeetta hai"; pair mein readout alag; sirf CLINC150 subset |
| **Deußer et al.** (arXiv 2609.37647) | Sirf Jev, 37 datasets, calibration, AURC | Doosre TDMs nahi; trained classifiers nahi; shift nahi |
| **Li & Wagle** (arXiv 2610.02076) | Frozen Qwen3.5-4B pehle hi Jev-style decision model jaisa; fine-tuning ka faida targeted | Vendor TDMs (Tev1, Nimble) aur hamare datasets nahi; hum in ka hypothesis test karte hain |
| **Tang & Zheng audit** (arXiv 2609.32160) | 28 papers ka audit + 14-item checklist | Open problems list kiye: **"calibration under deployment shift"** aur **"attributing gains"**. Hum yahi do bharte hain |

### 0.2b Paper ka markaz: Pillar A
Rafe & Das (2026) se overlap ka khatra sab se kam **Pillar A (prevalence shift)** mein hai; kisi bhi related paper ne yeh nahi kiya, aur audit isay open problem kehta hai. Is liye paper ka **markaz Pillar A** hoga: abstract, introduction aur main figures pehle isi par. Pillar B (budgets, decision training) doosra contribution; Exp 1, 5, 6 supporting (appendix ki taraf). Agar page limit ya scope ka masla ho to pehle Exp 5/6 aur exploratory analyses appendix mein jayenge, Pillar A kabhi nahi.

**Pillar A ki asal framing:** sirf "EM madad karta hai" nahi (yeh label-shift literature mein pehle se maloom hai jab model source prior par calibrated ho). Naya sawal yeh hai ke **zero-shot models (TDMs, LLMs) ka koi training prior hota hi nahi**; hum un ka **implicit prior** dev set se nikaalte hain. Kya label-shift correction is implicit prior ke saath bhi kaam karta hai, aur trained classifier (jis ka asli source prior maloom hai) ke muqable mein kaisa? Abstract aur introduction isi sawal par.

### 0.2c Stages (2026-10-06 faisla)

| Stage | Kya | Models | Output |
|---|---|---|---|
| **1** | Labels, data processing, candidates, model adapters, pilot, freeze; **Exp 1 + Exp 2**; Exp 4 ki conditions (a) aur (c) (Exp 1 ke runs se) | **Core 7:** Jev, Qwen3.5-4B, Tev1-4B, Qwen3.5-9B, Nimble 9B, DeBERTa-v3-large (full data), TF-IDF + LR | **Preprint v1** (Pillar A + P1/P2) |
| **2** | **Exp 3** (label budgets, DeBERTa/LR curves, Qwen QLoRA) aur Exp 4 condition (b) | Core 7 | v2 (Pillar B poora) |
| **3** (secondary, waqt ho to) | gpt-oss-20b, GLiNER2.5-Decide; **Exp 5, Exp 6**; exploratory analyses | + 2 secondary | TMLR submission version |

**Kyun:** reviewers ke mutabiq sab se bara khatra execution hai. Kisi bhi primary test (P1, P2, PA1–PA3) mein gpt-oss ya GLiNER nahi, is liye un ko secondary karne se koi confirmatory claim nahi badalta. gpt-oss 12 GB mein CPU offload aur alag scoring library maangta hai; GLiNER poori probability distribution nahi deta aur Pillar A se pehle hi bahar hai.

### 0.3 Contributions (agar experiments support karein)
1. **Pillar A:** TDMs, open LLMs aur trained classifiers ka calibration aur threshold transfer **controlled prevalence shift** ke under, aur bina target labels ke label-shift correction (Exp 2).
2. **Pillar B1:** **label-budget crossover**: kitne labels ke baad DeBERTa ya base LLM + LoRA zero-shot Jev se aage nikalta hai (Exp 3).
3. **Pillar B2:** **decision training kya deti hai**: same backbone aur same readout par (a) koi training nahi, (b) task fine-tune, (c) vendor decision training (Qwen3.5-4B ↔ Tev1; Qwen3.5-9B ↔ Nimble, same base weights + adapter) (Exp 4).
4. Supporting: paraphrase aur non-adversarial distractor robustness, aur ek 12 GB consumer GPU par efficiency.
5. Harness, prompts, label texts aur saare per-example outputs public; audit checklist ke saath poora compliance (sec 1.3).

### 0.4 Kya claim NAHI karna
- "First independent evaluation" (pehle se kai papers hain).
- "Jev well calibrated / sasta hai" bataur naya finding (pehle report ho chuka).
- Cascade/routing framework (Rafe & Das ne kiya; hum drop karte hain).
- "Privacy-preserving": sirf "locally deployable" (motivation: kai organizations data external API par nahi bhej sakti; Jev API reference hai).
- "No contamination" (sec 15 ki wording).

### 0.5 Models aur datasets ek nazar mein
- **TDMs:** Jev 1.13 (API, Cloudflare), Tev1-4B, Nimble 9B, GLiNER2.5-Decide.
- **Open LLMs:** Qwen3.5-4B, Qwen3.5-9B (Tev1 aur Nimble ke base), gpt-oss-20b.
- **Trained classifiers:** DeBERTa-v3-large, TF-IDF + LR.
- **Datasets (5, sab public, human-labeled):** CLINC150 (in-scope), HWU64, Financial PhraseBank, HateXplain, TREC-6.
- Frontier generative APIs scope se bahar.
- Tafseel: section 2 (models), section 4 (datasets).

### 0.6 Main limitations (paper mein)
Frontier LLMs nahi; sirf English text; prevalence shift resampling se (label-shift assumption by construction, covariate shift nahi); GLiNER label-shift correction se bahar; Jev ka version pin nahi ho sakta (har call par log); TREC-6 aur FPB ke chhote test sets; vendor training data hamare control mein nahi; contamination puri tarah rule out nahi.

---

## 1. Scope

### 1.1 Pillars, research questions aur hypotheses 🔒

| Pillar | Experiment | Confirmatory (primary) | Exploratory |
|---|---|---|---|
| **A** | Exp 2: prevalence shift | **RQ-A1 (two-sided):** kya prevalence shift ke saath Jev aur trained classifier (DeBERTa) ke ECE ka badlav alag hai? Direction pehle se fix nahi; literature ke mutabiq asar architecture se zyada calibration aur prior estimation par depend kar sakta hai (Alexandari et al. 2020; Garg et al. 2020). **H-A2 (directional):** EM prior adjustment (Saerens et al. 2002) un models ka shift ke baad ECE kam karta hai jo poori probability distribution dete hain. **v1.1.1 (audit 2026-10-08, results se pehle):** **confirmatory test (PA3) sirf Jev par plain EM (calibration step ke baghair) hai**; doosre models ke Exp 2 nateeje descriptive. Yeh Alexandari et al. (2020) ki BCTS+EM finding ki replication nahi (un ke mutabiq calibration ke baghair EM kamzor hai); BCTS+EM Exp 2 mein descriptive arm. PA3 two-sided p use karta hai (directional hypothesis ke liye conservative). | Shift ke baad threshold ka excess risk har model par; LLMs aur doosre TDMs |
| **B1** | Exp 3: label budgets | **Estimation (hypothesis test nahi):** har dataset par crossover budget (sec 10.3) aur us ka CI | "Crossover high-K datasets (CLINC150, HWU64) par zyada hai": **sirf dataset-level description**, kyunke K task se confounded hai |
| **B2** | Exp 4: decision training | **H-B2:** vendor decision fine-tune base model ke muqable mein Macro-F1/ECE badalta hai. **Asal test P2 (Nimble ↔ Qwen3.5-9B):** same base weights + adapter, training distribution documented. P1 (Tev1 ↔ Qwen3.5-4B) "closest available comparison", causal nahi | Faide ka K/task ke saath taluq (moderation): exploratory, causal claim nahi |
| Supporting | Exp 1, 5, 6 | — | Confirmatory nahi |

**Framing rules:**
- Tev1 pair ko **causal ya controlled experiment nahi** kehna: Tev1 full fine-tune hai, us ki exact base revision aur recipe confirm nahi, aur K = 18–22 (label groups) us ke training distribution (zyadatar ≤ 8 options) se bahar hai. Agar P1 mein CLINC/HWU par faida kam aaye to yeh OOD ka asar bhi ho sakta hai; is ko pehle se "uninterpretable for H-B2 moderation" maana gaya hai.
- Nimble pair zyada saaf hai: same base weights, same quantization, sirf adapter ka farq.

### 1.2 Uniform principle 🔒
Har primary evaluation har primary dataset par chalegi. Paper wording: *"a common evaluation protocol is applied across all five datasets."*

**Exceptions (pehle se declared):** (1) GLiNER poori probability distribution nahi deta, is liye label-shift correction (Exp 2) se bahar; (2) koi model kisi dataset ke K candidates support na kare to "not supported" (sec 13); (3) Exp 3 ke curve ka aakhri point "full train" har dataset mein alag size ka hai, aur 5,000 wala point sirf un datasets par jahan train ≥ 5,000.

### 1.3 Audit checklist (Tang & Zheng 2026) compliance
B1 ✅ (base LLM label-probability readout) · B2 N/A (koi cascade nahi) · B3 ✅ · T1 ✅ (model spread) · T2 ✅ · T3 ✅ exploratory · T4 ✅ · R1 ✅ · R2 ✅ · R3 ✅ (`deviations.md`) · C1 ✅ · C2 ✅ (Acc@coverage) · C3 ✅ (dev thresholds) · C4 ✅ (caching on/off). Paper ke appendix mein yeh table.

---

## 2. Models 🔒 (revisions pilot ke baad freeze)

| ID (code mein) | Group | HF repo | Revision | License | Precision / backend | Readout |
|---|---|---|---|---|---|---|
| `jev` | TDM (API) | Cloudflare Workers AI `typesafe/jev` | Response `model` field (abhi `jev-1.13.0`), har call par log | TypeSafe terms | API | Native `probabilities` |
| `tev1-4b` | TDM | `togethercomputer/Tev1-4B-experimental` | `0b7becf017daa0e5eb222f8ce7483c8c8259c52f` | Card (`0b7becf0`, check 2026-10-08): "release license … being finalized"; base Apache-2.0. Hum weights redistribute nahi karte; paper mein license status disclose | bf16, transformers | Letter scoring (sec 8) |
| `nimble-9b` | TDM | Base `Qwen/Qwen3.5-9B` + adapter `bespokelabs/Bespoke-Nimble-9B` (tag `original-2676`) | Base `c202236235762e1c871ad0ccb60c8ee5ba337b9a`; adapter `fe33f82b94b17ed4e325932f37d4e63b8a7abe29` | Apache-2.0 | 8-bit (bitsandbytes), transformers + PEFT (12 GB mein fit) | Letter scoring |
| `gliner-decide` *(secondary, Stage 3)* | TDM | `fastino/GLiNER2.5-Decide` | `5a7adf72a23b4d311abae6ce050d7f0012bb3416` | Apache-2.0 | fp32/fp16, `gliner2` library | Native top-label probability (internal per-label scores: Stage 3 amendment) |
| `qwen3.5-4b` | LLM | `Qwen/Qwen3.5-4B` | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | Apache-2.0 | bf16, transformers | Letter scoring (Tev1 prefix) |
| `qwen3.5-9b` | LLM | `Qwen/Qwen3.5-9B` | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` | Apache-2.0 | 8-bit, same as Nimble | Letter scoring (Nimble prefix) |
| `gpt-oss-20b` *(secondary, Stage 3)* | LLM | `ggml-org/gpt-oss-20b-GGUF` (`gpt-oss-20b-MXFP4.gguf`) | `ef9b12f2ff56c69cf32153a02784e7a3c88bf524` | Apache-2.0 | MXFP4, llama-cpp-python, `--n-cpu-moe` (Stage 3 amendment) | Letter scoring after `final` channel |
| `deberta-v3-large` | Classifier | `microsoft/deberta-v3-large` | `64a8c8eab3e352a784c658aef62be1662607476f` | MIT | fp16 training | Softmax |
| `tfidf-lr` | Classifier | scikit-learn | version log | BSD | CPU | Softmax |

**Notes:**
- **Nimble pair:** Nimble LoRA adapter hai, is liye `qwen3.5-9b` aur `nimble-9b` **bilkul same base weights, same quantization** par chalenge; farq sirf adapter ka. Yeh pair ko Rafe & Das se zyada controlled banata hai. (v1.1.1 disclosure: Nimble ka vendor setup bf16 hai; hum dono 9B models 8-bit mein chalate hain (12 GB). Pair ka muqabla barabar hai, lekin Nimble ke absolute numbers vendor setup se alag ho sakte hain; paper mein saaf likhna.)
- **Nimble version (v1.0.3 tak):** ~~`main @ bd792f44` (12,026 rows, T = 1.0), Rafe & Das se comparability ke liye~~ → **2026-10-07 badla** (agla bullet). Rafe & Das wala version (`main`) ab `nimble-9b-main` ke tor par **Stage 1 ka planned exploratory run** hai, taake un se seedha muqabla bhi bacha rahe. Dono revisions ka shipped T = 1.0 = hamara raw T. v2 ka T = 2.179 transferred tha (fit nahi), v3 CC BY-NC.
- **Nimble revision (faisla 2026-10-07): P2 = tag `original-2676` (`fe33f82b94b17ed4e325932f37d4e63b8a7abe29`).** Wajah: `main @ bd792f44` (release "v3-12026", 24 Sep) ke `schema_config.json` `data_audit.components` mein public data hai: Banking77 2,000, MultiNLI 1,000, BoolQ 500, AG News 500, DBpedia 500, **TREC 500**, yani P2 ka TREC-6 cell "training mein dekha" hota (rule: primary comparisons sirf unseen cells par). `original-2676`: 2,676 synthetic rows, koi public data nahi, Apache-2.0, max_length 2,048. Verification: `parallel_schema.py` dono revisions mein byte-identical (sha256 `a0a0f94d…` = contract ka `prompt_code_sha256`); hamara prompt us ke `prepare_prompts` se 100 examples par token-identical; candidate letter ids same; sab se lamba prompt 729 tokens (sab dev+test ≤ 2,048); adapter ki `tokenizer.json` file base se byte-different lekin **16,239 dev+test prompts par token ids identical**, chat template aur A–Z same; readout same (candidate logits softmax, T = 1). `main` (`nimble-9b-main`) sirf **exploratory** (Holm ke bahar, Stage 1 mein optional): "public classification data ke saath training" ka asar.
- **Tev1 pair:** Tev1 full fine-tune hai; base revision (check 2026-10-08): card mein `base_model: Qwen/Qwen3.5-4B` bina revision; Tev1 ka `config.json` hamare `qwen3.5-4b` revision `851bf6e8` ke config se har field mein identical. Exact base commit published nahi; paper mein yahi likhna. **Confound:** Tev1 training mein zyadatar ≤ 8 options (sirf research-taxonomy hissa 24 tak); CLINC150/HWU64 ka K = 18–22 (label groups) us ke training distribution se bahar hai. P1 mein yeh likhna; H-B2 ka K ke saath taluq sirf exploratory (sec 1.1).
- gpt-oss reasoning effort: `low` primary, `high` secondary. **`high` ek alag readout hai:** reasoning ke baad final channel par letter probability aksar ~1.0 hoti hai, is liye `high` ka ECE/AURC `low` se comparable nahi; paper mein alag row aur alag naam.
- Thinking mode: Qwen/Tev1/Nimble par `enable_thinking=False` 🔒.

---

## 3. Hardware, environment aur run workflow

- **Development:** laptop (RTX 4050 Laptop 6 GB). Sirf dev aur chhote checks; **koi latency number report nahi** 🔒.
- **Run machine:** 12 GB GPU. Saare reported runs. **NVIDIA GeForce RTX 4070 Ti (12 GB)**, Windows 11; exact hardware/software har final run ke environment record se (versions `requirements-run.txt` mein pinned).
- **Har run apne saath yeh record karega** (`results/<run_id>/environment.json`): GPU naam, VRAM, driver, CUDA, CPU, RAM, OS, Python aur saari library versions, har model ki revision/hash, quantization, `jev` ka `model` field, run start/end UTC.
- **Data:** run machine par `s1bench download` + `s1bench verify` (sha256 `data/MANIFEST.json` se match hona zaroori).
- **Secrets:** `.env` (Cloudflare) kabhi `results/` mein copy ya package nahi hogi 🔒.

---

## 4. Datasets 🔒

### 4.1 Sources (sab `configs/datasets.yaml` aur `data/MANIFEST.json` mein pinned)

| ID | Raw source | Use |
|---|---|---|
| `clinc150` | `clinc/clinc_oos` @ `155b9c71`, config `plus` | Sirf in-scope (label ≠ oos id 42) |
| `hwu64` | `DeepPavlov/hwu64` @ `0dd289cc` | Train 8,954 / test 1,076 |
| `fpb` | `atrost/financial_phrasebank` @ `fc7fb491` (+ original zip for verification) | 3,100 / 776 / 970 |
| `hatexplain` | GitHub `hate-alert/HateXplain` @ `01d74227` | Official `post_id_divisions.json` |
| `trec6` | CogComp original files | Coarse labels |

### 4.2 Splits 🔒

| Dataset | Train | Dev (val) | Test | Note |
|---|---|---|---|---|
| CLINC150 | 15,000 | 3,000 | 4,500 | Official |
| HWU64 | 8,954 − 1,076 = 7,878 | 1,076 (train se carve) | 1,076 | Dev: **train ki class distribution** ke mutabiq stratified sample (test information use nahi), seed `20261006`. DialoGLUE test distribution se banata tha; hum jaan boojh kar nahi, kyunke Exp 2 ka source prior isi dev se aata hai |
| FPB | 3,100 | 776 | 970 | `atrost` |
| HateXplain | 15,383 | 1,922 | 1,924 | Official; 919 undecided posts pehle se bahar |
| TREC-6 | 5,452 − dev | 500 (train se carve, stratified, seed `20261006`) | 500 | Official test = TREC 10 |

### 4.3 Cleaning rules 🔒 (order mein)
1. **Text normalization sirf comparison ke liye** (v1.0.4): lowercase, aur har non-letter/non-digit run (kisi bhi script mein) → ek space; is tarah sirf punctuation ke farq wale texts ("i need my account frozen" / "…frozen!") same gine jate hain. Jo text is se khaali ho jaye (sirf symbols/emoji) us par lowercase + whitespace collapse. Model ko original text milta hai. (Pehle sirf lowercase + whitespace tha; leakage audit ne CLINC mein 20 punctuation-only test copies train/dev mein paayi.)
2. **Train/dev ∩ test overlap:** jo train ya dev examples kisi test text se exactly match karein, **train/dev se hataye**. Test kabhi nahi badalta. Counts report.
3. **Conflicting duplicates (same text, alag labels):** saari copies har split se hataayi jati hain (sirf train/dev; test mein mile to report + test mein rakhna, label as-is). Counts report.
4. **Train ∩ dev:** jo train examples kisi dev text se exactly match karein, **train se hataye** (dev unseen rahe; v1.0.3 clarification, implementation ke waqt).
5. **Same-label duplicates** train mein: ek copy rakhni.
6. HateXplain label = 3 annotators ka majority; posts text = `post_tokens` ko space se jodna. **Asserts:** 15,383 + 1,922 + 1,924 = 19,229 = 20,148 − 919; test = 594 hate / 548 offensive / 782 normal; koi undecided post kisi split mein nahi.
7. TREC: paper mein "5,500" (Li & Roth) vs actual 5,452 ka discrepancy note.

### 4.3b Processed counts (v1.0.4, 2026-10-07, `s1bench prepare`, deterministic: rerun par same sha256)

| Dataset | Train | Dev | Test | Hataye gaye (test-overlap / conflicting / dev-overlap / duplicates) | Min per class (train / dev / test) |
|---|---|---|---|---|---|
| CLINC150 | 14,939 | 2,994 | 4,500 | 25 / 2 / 13 / 27 | 88 / 19 / 30 |
| HWU64 | 7,871 | 1,076 | 1,076 | 0 / 4 / 2 / 1 | 29 / 4 / **3** |
| FPB | 3,090 | 775 | 970 | 4 / 0 / 3 / 4 | 382 / 97 / 125 |
| HateXplain | 15,330 | 1,921 | 1,924 | 10 / 31 / 2 / 11 | 4,368 / 548 / 548 |
| TREC-6 | 4,869 | 498 | 500 | 11 / 0 / 8 / 66 | 76 / 8 / **9** (ABBR) |

Test sets bilkul original hain (koi example nahi hataya; v1.0.4 ke baad bhi test files ka sha256 same); kisi test set mein conflicting labels nahi. Near-duplicates (char-TF-IDF cosine ≥ 0.95, paraphrase level) official CLINC/HWU splits ka hissa hain aur hataye nahi jate; dataset property ke tor par report (CLINC 6.6%, HWU 7.4% test items). Har split mein saari classes mojood. Per-dataset tafseel: `data/processed/<dataset>/report.json`; hashes: `data/processed/MANIFEST.json`.

### 4.4 Common record format (`data/processed/<dataset>/<split>.jsonl`) 🔒
```json
{"id": "clinc150:test:000123", "dataset": "clinc150", "split": "test",
 "text": "...", "label": "transfer", "label_index": 17,
 "meta": {"source_id": "...", "features": {}}}
```
`id` stable aur deterministic (dataset + split + original order index). Har processed file ka sha256 ek `data/processed/MANIFEST.json` mein.

### 4.5 Training exposure 🔒 (reporting rule)
`docs/research_idea.md` section 5 ka matrix paper ki table banega. Tev1 recipe aur Nimble ke MASSIVE evaluation se test sets ka exact/near-duplicate overlap check; mile to woh test examples **report** honge aur ek sensitivity analysis (un ke bina) di jayegi.

**Implemented (2026-10-07, `s1bench overlap`):** references = vendor-documented public corpora (`configs/overlap_references.yaml`, pinned, hashes `data/reference/MANIFEST.json`): MASSIVE en-US (Nimble evaluation; HWU64 lineage), Banking77, AG News, SST-5, BoolQ, MultiNLI, DBpedia. Asal Tev1/Nimble training rows public nahi, is liye yeh source-level check hai. Exact = normalized text equal; near = hashed char 3–5-gram cosine ≥ 0.9 (short-text corpora); < 4 tokens wale exact matches alag gine jaate hain, flag nahi. Output: `data/overlap/report.json`, `flagged_<dataset>.json` (sensitivity analysis ke liye).

| Test set | Flagged | Kahan se | Tabsira |
|---|---|---|---|
| HWU64 | **614 / 1,076 (57%)** | MASSIVE en-US (582 exact: train 421, test 94, val 67; 32 near) | Lineage confirm. MASSIVE Nimble ka **evaluation** set hai, training nahi; lekin public (2022) hai, is liye pretraining exposure sab models ke liye mumkin. HWU64 results ke saath non-overlap subset (462) ki sensitivity analysis **zaroori** |
| CLINC150 | 96 / 4,500 | MASSIVE 74, MultiNLI 27 (zyada tar aam chhote jumle: "i do not know", "what time is it in dallas") | Chance matches; Banking77 se overlap ~0 |
| TREC-6 | 4 / 500 | MASSIVE (QA utterances) | Nimble `main` TREC ki 500 rows par train hua (`public-trec: 500`) → is liye P2 `original-2676` par (sec 2); `main` sirf exploratory, TREC-6 cell ⚠ |
| FPB | 1 / 970 | AG News (same press release) | — |
| HateXplain | 0 | — | — |


---

## 5. Label texts (verbalization) 🔒 (draft ke baad)

- File: `data/labels/<dataset>.yaml`, har label: `id` (original), `name` (human-readable), `description` (ek line).
- **Rules:**
  - TREC-6: Li & Roth ki official taxonomy (`data/raw/trec/definition.html`) se descriptions.
  - FPB: investor/stock-price nazariya ("positive influence on the stock price ...").
  - HateXplain: hate vs offensive vs normal ki definitions original paper ke annotation guidelines ke mutabiq (Roy et al. 2023 ki definitions reference).
  - CLINC150/HWU64: snake_case naam readable banana (`pay_bill` → "pay bill"); description intent ka ek line ka matlab. Kisi benchmark model (Qwen, gpt-oss) se generate nahi karna; HWU64 ki Qwen-generated descriptions **use nahi** hongi.
- **Author:** authors likhenge, review aur approve karenge; phir freeze.
- **Sensitivity variant:** "raw names only" (description ke bina), test set ke 500-example subset par, saare zero-shot models.

---

## 6. Candidate sets 🔒

- **Rule (v1.0.4):** labels ≤ 20 → saare labels (FPB, HateXplain: K = 3; TREC-6: K = 6). Labels > 20 (CLINC150, HWU64) → **fixed label groups**: labels ek dafa embedding similarity se round(n/20) balanced groups mein baante jate hain (sizes mein zyada se zyada 1 ka farq): CLINC150 8 groups (18–19), HWU64 3 groups (21–22). Har example ka candidate set = **us ke gold label ka poora group**.
- **Kyun badla (leakage audit 2026-10-07):** purana set "gold + gold ke 10 qareebi labels + 9 random" tha, is liye gold option list ka "markaz" tha: text padhe bina sirf options se (har option ke top-10 padosi set mein hain ya nahi) gold CLINC par 97.6% aur HWU par 90.2% pehchana ja sakta tha (chance 5%). Yeh ishara sirf option list padhne wale models (Jev, Qwen, Tev1, Nimble) ko milta, trained classifiers ko nahi. Fixed groups mein group ke har label ko **bilkul same option list** milti hai, is liye list mein gold ka koi ishara nahi (unit test: frozen files par har gold label ka ek hi set).
- **Grouping:** capacity-constrained k-means (cosine) on `"{name}: {description}"` embeddings; seeded k-means++ (20 restarts, `candidate_seed`), sab se zyada within-group similarity wala restart; `data/candidates/<dataset>/label_groups.json`. Report: within-group mean cosine, nearest neighbour usi group mein: **CLINC 64.7%, HWU 84.4%** (yani CLINC mein kuch labels ka sab se qareebi confuser doosre group mein; per-group accuracy report karni hai).
- **Embedding model:** `sentence-transformers/all-mpnet-base-v2` @ `e8c3b32edf5434bc2275fc9bab85f82640a19130` (Apache-2.0; kisi benchmark model ki family se nahi).
- **Order:** har example ke liye seed se shuffle; letters **A, B, C, …** isi order mein. **Gold position counterbalanced:** har split mein, har set size (group/all labels) ke andar har position barabar baar (±1, ek size ke do groups mila kar ±2) gold hoti hai (sha256(seed:id) order, position = rank mod K). Position bias Jev aur Qwen mein documented hai.
- **Seed:** `candidate_seed = 20261006` (`configs/seeds.yaml`).
- **File:** `data/candidates/<dataset>/<split>.jsonl`: `{"id", "candidates": ["label_id", ...], "gold_position": int}` + sha256 manifest. **Saare models, saare experiments yahi file** 🔒. Frozen artifact: run machine par dobara nahi banengi, sirf hash verify.
- **Rotation check:** har dataset ke test se 500 examples (seeded) par K cyclic rotations (CLINC/HWU par 4 rotations: 0, ⌊K/4⌋, ⌊K/2⌋, ⌊3K/4⌋), sirf exploratory.
- **Supporting full-label run:** CLINC150 (150) aur HWU64 (64) par Jev, DeBERTa, LR (K limit nahi). Literature 150/151-way aur 64-way hai, is liye yeh **sanity reference** bhi hai; paper mein comparability note (hamara primary in-scope + K = 20).
- **Exploratory:** random-only candidate file (500 subset, K = 20) vs groups (similar-label distractors ka asar).

---

## 7. Inputs aur prompts

### 7.1 Principle 🔒
*Same information, same decision space, same test examples, same output requirement; prompt wording model-native ho sakti hai.*

### 7.2 Per-model input 🔒 structure aur exact text

**(v1.1, approved by the authors 2026-10-08)** Exact prompt text alag `configs/prompts/` mein nahi; woh model adapters ke code mein hai (`src/s1bench/infrastructure/models/*.py`), aur un files ke sha256 har run ke `config_snapshot` (`code:` keys) mein record hote hain. Freeze = in files ke hashes freeze manifest mein; badlav = deviation. Wajah: prompt ko code se alag copy karna do sources of truth banata hai.

| Model | Input |
|---|---|
| `jev` | `{"state": <text>, "questions": {"label": {"type": "choice", "instructions": <task question>, "criteria": {"<label name>": "<description>", ...}}}}`; keys **label names** (letters nahi: Jev naam semantically parhta hai), order = shuffled candidate order |
| `tev1-4b` + `qwen3.5-4b` | Tev1 ka training format: system prompt (Tev1 card se verbatim) + user message = `json.dumps({"state", "question", "options": [{"label": letter, "key": <label name ka slug, v1.0.4; FPB par "0/1/2" ids nahi>, "description": "<name>: <description>"}]})`; assistant prefix `<think>\n\n</think>\n\n` |
| `nimble-9b` + `qwen3.5-9b` | Nimble ka `parallel_schema.py` format (`original-2676` aur `main` mein byte-identical): `safe_json({"context", "schema"})` + `Requested field: label`; single-token codes A–V (K ≤ 22) |
| `gpt-oss-20b` | Harmony: system (`Reasoning: low`), developer instruction + Tev1-style JSON; scoring `final` channel ke shuru par |
| `gliner-decide` | `classify_text(text, {"label": {<name>: <description>}})`, `include_confidence=True` |
| `deberta`, `tfidf-lr` | Sirf text; output head saare labels par, scoring candidate set tak restrict |

**Task question (har dataset ki ek line, 🔒 labels ke saath):** maslan CLINC150/HWU64: "Which intent does the user's request express?"; FPB: "From an investor's perspective, what is the sentiment of this financial news sentence?"; HateXplain: "How should this social media post be classified?"; TREC-6: "What type of answer does this question ask for?"

---

## 8. Prediction aur confidence 🔒

| Model | Prediction | Probability vector (candidates par) |
|---|---|---|
| `jev` | argmax `probabilities` (ties: candidate position; Jev ka apna `choice` tie par alag ho sakta hai, count report) | `probabilities` (2 decimals, sum 0.99–1.01; analysis mein renormalize) |
| Decoder models | argmax | Native answer position par candidate letters ke logits par softmax (**T = 1**). v1.0.4: charon local models `Qwen3_5ForConditionalGeneration` class se load (checkpoints ki apni class; Nimble adapter isi class par trained: `CausalLM` class par adapter ke 0/496 keys match hote the aur LoRA chupchaap no-op rehta tha). Adapter load ke baad sakht check: har adapter tensor match ho aur koi `lora_B` zero na ho, warna run ruk jata hai. Letter logits **fp32** mein final hidden state × lm_head ki letter rows se (bf16 lm_head output mein 0.125 ke steps se exact ties banti thi); model ke apne logits se match verify kiya. Memory: load ke baad vision encoder hataya jata hai aur input embeddings ki CPU copy se lookup (untied 9B par GPU table free, ~2 GB); logits bilkul same (verify kiya) |
| `nimble-9b` | argmax | Same (shipped T = 1.0 = raw) |
| `gpt-oss-20b` low/high | argmax | `final` channel ke shuru par letters ka softmax; high mein reasoning ke baad |
| `gliner-decide` | top label | Sirf top-label probability (full vector: Stage 3 amendment); label-shift correction se bahar |
| `deberta`, `tfidf-lr` | argmax | Softmax renormalized over candidates |

- **Secondary readouts (exploratory):** constrained JSON generation (agreement aur output validity), Tev1 full-vocabulary scoring, base Qwen par numeric-ID readout (Li & Wagle), Jev `confidence` field.
- **Output validity** sirf generation path aur native API outputs par.
- **Exp 1 table (v1.1.1):** `s1bench exp1 --run ... --split test --out ...`: decision metrics (Accuracy, Macro-F1, MCC, AURC, Acc@80/50) mein blocked = wrong + abstain (confidence 0, Macro-F1 sirf dataset classes par), probability metrics (ECE, Brier, NLL) sirf ok rows par, failed rows bahar (> 1% = rerun), blocked > 1% par blocked-excluded sensitivity bhi; probabilities candidates par renormalize.
- **Models ko gold kabhi nahi milta** (v1.0.4): runner har model ko label-free view deta hai (example ka label aur candidate set ki gold position khaali); gold prediction ke baad joda jata hai. (v1.1.1 wording: example ka `meta` view mein rehta hai, aur TREC-6 mein us mein fine label hai; koi adapter `meta` nahi padhta (audit ne confirm kiya), is liye kisi number par asar nahi; paper mein "adapters sirf text, labels aur candidates padhte hain" likhna.)

---

## 9. Metrics 🔒

| Metric | Definition |
|---|---|
| Accuracy, Macro-F1, MCC | Candidate set ke andar prediction vs gold; Macro-F1 dataset ke saare classes par |
| **ECE (primary)** | Top-label, **15 equal-width bins**. Jev probabilities 2 decimal tak round aati hain (sum 0.99–1.01): analysis mein renormalize; is se ties zyada (tie-break: candidate position) aur confidence ka resolution 0.01 |
| ECE (secondary) | Classwise; equal-mass bins (sensitivity) |
| **Brier** | `(1/N) Σᵢ Σₖ (pᵢₖ − yᵢₖ)²` candidates par; top-label Brier jahan sirf top-label probability |
| NLL | `−(1/N) Σᵢ log p_{i,gold}`, clip `1e-12` |
| **AURC** | Confidence = max probability; risk = error rate; coverages k/n par selective risks ka mean (Geifman & El-Yaniv), barabar confidence wale group ko group ki mean error (v1.1.1: "trapezoid" ki jagah asal definition likhi; farq O(1/n)) |
| Selective risk @ threshold | Exp 2 (sec 10.2) |
| **Acc@coverage** (audit C2) | Accuracy among top-80% aur top-50% confident examples (Deußer et al. jaisa) |
| **Model spread** (audit T1) | Har dataset par best − worst Macro-F1; ceiling ke qareeb datasets (TREC-6) par parity claim nahi |
| **Per-class** | Per-class F1 har model; HateXplain confusion matrix (hate ↔ offensive); TREC-6 ABBR (n = 9) alag; ECE bin occupancy |
| Efficiency | p50/p95 latency (per example, warm), throughput, peak VRAM, tokens/example, GPU-seconds/1,000, API cost/1,000. **Local scoring: prefix caching on aur off dono latency** (audit C4; v1.1 implement: `predict --time-prefix-cache --limit 210`, seeded dev subset, pehli 10 warm-up; shared prefix = rendered prompts ka longest common token prefix (system prompt + chat template + JSON ka shuru, kyunke dono formats mein text options se pehle aata hai); ON = prefix cache ek dafa, har example par cache copy (timed) + sirf suffix ka forward; **sirf latency**, reported predictions hamesha bina cache. fp32 mein on/off same (logic verify kiya); bf16/8-bit mein computation path ki wajah se probabilities mein farq aata hai, is liye final runs mein `max_abs_prob_difference_on_vs_off` aur `argmax_differs_on_vs_off` record honge aur agar paper mein yeh limitation likhi to sirf final runs ke numbers ke saath. Hamari predictions ek hi path se aur deterministic (sec 12). **Trained classifiers** (v1.1): batch mein predict karte hain, is liye selected model par alag timed pass: seeded dev subset 210 (10 warm-up), batch size 1, tokenization timer ke bahar (decoders jaisa), DeBERTa GPU + CUDA sync, TF-IDF CPU; `train_summary.json` → `latency`. Jev (Cloudflare) par repeated-request caching nahi, kyunke har input alag |
| Stability | Jev run-to-run flip rate; paraphrase consistency; distractor slope |

---

## 10. Experiments

### 10.1 Exp 1: Zero-shot benchmark
Saare zero-shot models, saare 5 test sets, candidate files ke saath. **Har example ka poora probability vector save** (Exp 2 isi se chalta hai). Jev: full test **1 run**; nondeterminism ke liye 500-example subset par **3 runs**.

### 10.2 Exp 2: Prevalence shift (MAIN A) 🔒

**Literature basis:** Dirichlet shift simulation Lipton et al. (2018, BBSE), Azizzadenesheli et al. (2019, RLLS), Alexandari et al. (2020, BCTS + EM), Garg et al. (2020, MLLS). Un sab mein source prior **uniform** tha (MNIST, CIFAR), Dirichlet(α·1) with α ∈ {0.1, 1, 10}, aur shifted test set **with replacement** se sample hota tha (Lipton p. 6; Alexandari p. 6; Garg p. 9). Hamare datasets imbalanced aur per-class chhote hain, is liye do badlav.

- **Input:** Exp 1 (aur DeBERTa/LR full-data) ke saved probability vectors. **Naya inference nahi.**
- **"No shift" baseline:** original test set jaisa hai waisa (empirical test prior). 🔒 **v1.1.1 (audit 2026-10-08, results se pehle):** is ke saath **resampled no-shift control** (`alpha = noshift_resampled`, 200 draws, with replacement, q = empirical test prior, same CRN): shift ka asar isi se naapna (ECE resampling bias ki wajah se); original set ki row (`inf`) bhi report. Har row mein `n_classes_present`; realized dev coverage har arm ke liye.
- **Target priors:** **Dirichlet(α · K · π_s)**, jahan π_s = source (train) class prior aur K = classes ki tadaad. Uniform π_s par yeh bilkul Dirichlet(α·1) ban jata hai, yani literature ke barabar; imbalanced datasets par target priors π_s ke gird concentrated hain, uniform ke gird nahi. (v1.0.1 ka Dirichlet(α·1) imbalanced datasets par "α = ∞" ko bhi ek shift bana deta tha.) α ∈ **{10, 1, 0.1}** (literature grid); **200 draws** per α; seed `20261006`.
- **Resampling (primary):** literature ki tarah **with replacement**, test set size ke barabar, target prior ke mutabiq class-stratified. Har draw par **effective unique examples** (distinct examples / n) report.
- **Resampling (sensitivity, exploratory):** **without replacement**; har draw ka size n = min(n_test, min_k ⌊n_k / q_k⌋) taake koi class available count se zyada na maange. n < 200 wale draws alag mark. Agar primary aur sensitivity ke nateeje direction mein alag hon to paper mein dono.
- **Shift magnitude:** har draw par KL(target ‖ empirical test prior). Note: yeh test ke empirical prior se naapa jata hai, model ke implied prior se nahi.
- **Source prior (assumption, paper mein likhna):** trained models ke liye train-split prior. Zero-shot models (Jev, LLMs, TDMs) ka koi training prior nahi; un ke liye **dev set par average predicted distribution** (BBSE-style proxy, Lipton et al. 2018). EM ka assumption hai ke model source prior par calibrated hai (Saerens et al. 2002; Garg et al. 2020); zero-shot models ke liye yeh assumption kamzor hai, aur yahi H-A2 test karta hai.
- **Candidate-restricted EM (CLINC150, HWU64):** prior 150/64 classes par; har example ka support sirf us ke candidates (us ke gold label ka group, 18–22).
  - E-step: p̃(y|x) ∝ p(y|x) · q̂(y) / π̂_s(y), sirf candidates par renormalize.
  - M-step: q̂(y) = (1/N) Σᵢ p̃ᵢ(y), candidates ke bahar p̃ = 0.
  - Gold hamesha candidates mein hota hai, is liye candidate restriction shift ka kuch asar khud absorb kar leti hai. Is confound ko dikhane ke liye **full-label setting** (Jev, DeBERTa, LR; 150/64-way) par bhi Exp 2 chalana (exploratory).
- **Threshold transfer:** dev (no shift) par woh confidence threshold τ jo **80% coverage** deta hai. Report teen numbers: (a) no-shift test par τ ka risk; (b) shifted test par τ ka risk; (c) **shift-induced excess = (b) − (a)**. (Dev aur test ka normal farq (a) mein alag ho jata hai.) Exploratory: 5% target-risk threshold (kuch datasets par infeasible).
- **Recalibration arms:** (i) none; (ii) temperature scaling (dev par NLL; Guo et al. 2017); (iii) **EM prior adjustment** (Saerens et al. 2002; max 100 iterations, tol 1e-6) bina target labels; (iv) **BCTS + EM** (Alexandari et al. 2020: class-specific biases ke saath temperature scaling dev par, phir EM); (v) oracle prior (asal target prior; upper bound).
- **Metrics per draw:** Macro-F1, accuracy, ECE (15 bins, Alexandari jaisa), Brier, NLL, AURC, threshold risks (a)–(c), effective unique examples.
- **Scope:** full probability vector wale models; GLiNER sirf "none" arm (top-label) aur limitation.
- **Implementation details frozen in v1.0.4 (analysis audit 2026-10-07):**
  - **Common random numbers:** shifted samples **poore processed test set** (model se independent) par bante hain, phir har model ki rows par map; jo rows kisi model ne kho di (failed/blocked) woh us sample se nikalti hain aur unka hissa (`mean_share_of_sample_lost`) report hota hai. Is tarah har model aur har arm bilkul same shifted examples par.
  - **Thresholds:** har arm ka τ usi arm ko (unshifted) dev par lagane se: none/temperature/BCTS+EM ke ilawa EM arm ka τ dev par EM ke baad, oracle ka τ dev ke empirical prior par oracle adjustment ke baad. Realized dev coverage meta mein.
  - **Implicit prior smoothing:** zero-shot π_s = mean dev prediction + 1e-3, renormalize (warna jo class dev par kabhi predict na ho us ka π ≈ 0 aur EM ratio q/π phat jata hai). PA3 mein bhi wahi.
  - **Log floor EPS = 1e-6** (temperature/BCTS fitting; Jev ki exact zeros). Jev FPB par fitted T EPS par depend karta hai (1e-3 → 1.58, 1e-6 → 2.00, 1e-12 → 2.46): **sensitivity row** `exp2 --eps 1e-3` aur `--eps 1e-12` (EM/oracle arms par asar nahi).
  - **Macro-F1 per draw:** jo classes us sample ke gold ya predictions mein mojood hon un par average (sklearn default; strong shift par kuch draws mein 1–2 classes hi hoti hain); har draw par present classes ki tadaad report.
  - **Without replacement:** realized n = Σ⌊n·q_k⌋ (n se thoda kam), har row mein `n`; n < 200 wale draws analysis mein alag. Design simulation (sirf label counts) ke mutabiq HWU64 aur TREC-6 par chhote α ke kai draws n < 200 dete hain; is liye yeh sensitivity exploratory hi rahegi, primary (with replacement) nahi badla.
  - **Dev checks** (pipeline jaanch, paper result nahi) `--pseudo-split` ke saath: **dev split ke do seeded hisse** fitting/evaluation; test split kabhi nahi.

### 10.3 Exp 3: Label budgets (MAIN B)

**Literature basis:** Le Scao & Rush (2021) ki tarah learning curves multiple runs ke saath aur "kitne data points ke barabar" ka sawal; Edwards & Camacho-Collados (2024): text classification par fine-tuned chhote models aksar in-context LLMs se aage; Mosbach et al. (2021) aur Zhang et al. (2021): chhote data par BERT-type fine-tuning unstable, failed runs aam.

- **Budgets (absolute, sab datasets par):** 100, 250, 500, 1,000, 2,500; **5,000** sirf jahan train ≥ 5,000 (CLINC150, HWU64, HateXplain); aur **full train** (curve ka aakhri point, har dataset ka apna size). Har point ke saath **shots-per-class** bhi report (CLINC150 par 2,500 ≈ 17/intent; full = 100/intent), taake intent literature (5/10-shot) se jor sake. **5 seeds** (subset seeds `1001–1005`); stratified jahan har class ka kam az kam 1 example mumkin ho; Sirf CLINC150 @100 par kuch classes ka example nahi (100/150; v1.1.1 check: HWU64 @100 aur CLINC150 @250 par saari classes) (expected, likhna).
- **DeBERTa-v3-large (stability rules, Mosbach et al. 2021; Zhang et al. 2021):** AdamW **with bias correction**, **LR grid {1e-5, 2e-5}**, har (dataset, budget, seed) par dev par chunav, linear warmup 10% phir decay, weight decay 0.01, dropout 0.1, batch 16, fp16, grad clip 1.0; epochs: budget ≤ 500 par 20, warna 5; chhote budgets par **kam az kam 500 optimizer steps** (epochs ki jagah steps, taake "too few iterations" wala failure na ho); har 100 steps par dev eval, best checkpoint dev Macro-F1 par; **max length 128** (p99 input length ≤ 64 tokens har dataset par).
- **v1.0.4 implementation (training audit 2026-10-07):** reserve seeds `configs/seeds.yaml` se, **har (dataset, budget) ke liye ek mushtarka queue** (har reserve seed ek dafa, kul max 3 restarts; is liye ek (dataset, budget) ke saare seeds **ek hi `train` call** mein); jo seed phir bhi fail ho woh **majority-class predictor** (train class frequencies, candidates tak restrict) ke tor par curve mein; DeBERTa **full data par bhi saare seeds** (pehla seed = Exp 2 ka `deberta-v3-large` baseline, baaqi `@fulls<seed>`); LR grid `train --lr ...`; checkpoint selection **candidate-restricted dev Macro-F1** (paper metric); best state CPU par aur fail/non-selected model GPU se free; saare hyperparameters `train_log.hparams` mein; asal run sirf `--device cuda`; budget ≥ train size → error (5,000 sirf jahan train ≥ 5,000). Max length 128 (95th percentile 15–56 tokens; > 128 sirf 2 HateXplain train rows).
- **Failed-run rule (freeze se pehle tay, Dodge et al. 2020 / Mosbach et al. 2021 ki definition):** agar rakhe gaye (best) checkpoint ki dev accuracy ≤ majority-class predictor ki dev accuracy ho to run "failed". **v1.1.1 (audit 2026-10-08, results se pehle):** dono **ek hi scale par**, yani har example ke candidate set ke andar: majority predictor = train class frequencies candidates tak restrict (yahi neeche wala fallback). Pehle code candidate-restricted accuracy (CLINC/HWU par chance ~5%) ko all-label majority (0.7% / 1.8%) se milata tha, is liye rule wahan kabhi trigger nahi hota. "Training ke aakhir" ki jagah best checkpoint, kyunke wahi model rakha jata hai. LR grid mein non-failed model hamesha failed par tarjeeh paata hai (phir dev Macro-F1), aur har failed run ki dev accuracy restart record mein. Failed run ko **agle reserve seed** (`1006–1008`) se restart, max 3 restarts per budget-dataset; har failed run aur restart report. Restarts ke baad bhi fail ho to woh seed "failed" ke taur par curve mein (majority-class performance ke saath) shamil, drop nahi.
- **TF-IDF + LR:** word 1–2 grams, sublinear tf, lbfgs **tol 1e-8, max_iter 20,000** (v1.0.4: default tol 1e-4 par solver 1–96 iterations mein bina warning ruk jata tha; CLINC @250, C = 1000 par dev F1 0.333 vs converged 0.396), `n_iter` log, non-convergence = error; C dev par select (grid {0.1, 1, 10, 100, 1000, 10⁴}; v1.0.4: 10⁴ add kiya kyunke converged solver ke saath CLINC/FPB/TREC ne 1000 (edge) chuna; 10⁴ ke baad FPB aur TREC ne phir edge (10⁴) chuna, lekin dev curve chapta hai (dev Macro-F1, C = 10³ / 10⁴ / 10⁵: FPB 0.714 / 0.717 / 0.719, TREC 0.837 / 0.839 / 0.840), yani yeh datasets lagbhag bina regularization ke behtar; grid **10⁴ par freeze** (edge selection paper mein note, aage nahi barhana); v1.0.3: grid barhaya kyunke {0.1, 1, 10} par har dataset ne edge (10) chuna; TREC ne phir 100 (edge) chuna).
- **Qwen3.5-4B QLoRA (Exp 4 condition b):** budgets 100, 500, 2,500; **3 seeds**; r = 16, α = 32, LR 1e-4; steps ka rule (approved by the authors 2026-10-08) DeBERTa jaisa: 20 epochs, kam az kam 500 optimizer steps, har 100 steps par candidate-restricted dev Macro-F1, best checkpoint (Stage 2; code abhi nahi, is liye pilot nahi hua: Stage 2 se pehle chhota dev pilot, aur koi badlav deviation ke tor par **results se pehle**); loss = candidate letters par cross-entropy; **same prompt + prefix as Tev1 pair**.
- **Crossover (estimation, Holm mein nahi):** har budget b par **paired difference** Δ(b) = Macro-F1(DeBERTa, b) − Macro-F1(Jev), same test examples par. CI ke liye bootstrap jo **seeds aur test examples dono** resample kare (Jev ki uncertainty bhi shamil). Crossover budget = sab se chhota b jahan Δ(b) ke 95% CI ka lower bound > 0. Agar do budgets ke darmiyan ho to interval report ("between b₁ and b₂"); agar full train tak na ho to "not reached". Secondary: Le Scao & Rush (2021) style linear interpolation par point estimate.
- **Reporting notes (v1.0.4):** (a) trained classifiers dev set ko model selection (C/LR/checkpoint) **aur** calibration/threshold (Exp 2) dono ke liye use karte hain, zero-shot models sirf calibration ke liye: yeh asymmetry paper mein likhni hai (selected dev scores thode optimistic). (b) Chhote budgets par bhi poora dev set selection ke liye (few-shot ke liye unrealistic; Perez et al. 2021) — limitation. (c) Budget par jo classes train mein na hon (sirf CLINC @100: 100/150 classes) un ko TF-IDF probability 0 deta hai aur DeBERTa > 0; is liye un budgets par NLL/ECE ka TF-IDF vs DeBERTa muqabla nahi.

### 10.4 Exp 4: Decision training (MAIN B)
- 4B: (a) `qwen3.5-4b` frozen, (b) `qwen3.5-4b` + task LoRA (Exp 3), (c) `tev1-4b`.
- 9B: (a) `qwen3.5-9b`, (c) `nimble-9b` (same base weights + adapter).
- Same prompt, same prefix, same candidate file, same readout 🔒.
- **Framing:** 9B pair (same base weights + adapter) H-B2 ka asal test; 4B pair "closest available comparison" (sec 1.1). Condition (b) task-specific hai aur (c) generic; dono alag sawal ka jawab dete hain.

### 10.5 Exp 5: Paraphrase (supporting)
500 test examples per dataset (seeded), 3 paraphrases each. Generator: benchmark se bahar aur Qwen/OpenAI family se bahar (Gemma ya Llama family, 12 GB par); exact model aur revision **Stage 3 amendment** mein (Stage 3 runs se pehle, deviations log mein). 10% sample (50 per dataset) par human check; jahan meaning/label badla woh drop. Metric: consistency (prediction same rahi), accuracy change.

### 10.5b Exploratory analyses (Holm ke bahar)
- Jev **neutral option names** (`option_1…`) + descriptions vs semantic names (audit T3), 500 subset.
- HWU64: `general_quirky`, `qa_factoid` ke saath aur baghair.
- FPB: test ke **100%-agreement** examples par alag numbers (Li 2023, Araci se sanity).
- HateXplain: Twitter vs Gab breakdown.
- Learning-curve budgets ko **shots-per-intent** mein (CLINC150, HWU64) literature se jorne ke liye.

### 10.6 Exp 6: Distractor (supporting)
Same 500 examples; faltu text input length ka 0 / 25 / 50 / 100 / 200%; fixed templates: **neutral text, bina identity terms ya slurs** (HateXplain label na badle), aur har dataset ke format mein (CLINC150 lowercase, no punctuation); sentence boundary par fixed-seed position. Truncation rule: model ki max length par right-truncation; report truncated %, aur main curve truncation-free subset par.

---

## 11. Statistics 🔒

### 11.1 Primary comparisons (Holm, **35 tests**, α = 0.05)

| # | Comparison | Metric | Tasks | Tests |
|---|---|---|---|---|
| P1 | `tev1-4b` vs `qwen3.5-4b` (closest available comparison) | Macro-F1, ECE | 5 | 10 |
| P2 | `nimble-9b` vs `qwen3.5-9b` (**H-B2 ka asal test**) | Macro-F1, ECE | 5 | 10 |
| PA1 | `jev` vs `deberta` (full): ΔECE (α = 1 vs no shift), two-sided (RQ-A1) | ΔECE ka farq | 5 | 5 |
| PA2 | `jev` vs `deberta` (full): shift-induced excess risk at α = 1 (80%-coverage threshold) | Excess risk ka farq | 5 | 5 |
| PA3 | `jev`: EM vs none at α = 1 (H-A2) | ECE | 5 | 5 |

**P3 (Jev vs full-data DeBERTa) ab exploratory:** full-data DeBERTa ka jeetna lagbhag yaqeeni hai aur Rafe & Das yeh dikha chuke hain; is ke 5 tests Holm budget kha rahe the. Is ki jagah crossover (Exp 3) zyada informative hai.

### 11.2 Tests
- **Paired bootstrap:** B = 10,000; same examples dono models ke liye resample; two-sided p = 2 · min(P(Δ ≤ 0), P(Δ ≥ 0)).
- **Shift (PA1–PA3), v1.0.3:** estimand = **fixed set of M = 50 target priors** q₁…q_M ~ Dirichlet(α·K·π_s) par average farq (shift distribution design ka hissa hai, sampling error nahi). Uncertainty sirf finite test set se: har bootstrap replicate (B = 1,000) test examples **class ke andar** resample karta hai (v1.0.4: stratified bootstrap, har class ka test count n_c same; unstratified base mein HWU64 ki 3-example class ~5% replicates mein gayab ho kar run crash karti thi), phir usi base sample se saare M shifted samples banata hai; dono models ko bilkul same samples (paired, common random numbers). Pehle design (har replicate mein ek random prior) par dev check mein CIs prior-draw variance se ~10 guna chaure the (PA2: 0.54 vs 0.04), jis se tests ki power na hone ke barabar thi. PA2 ka excess: shifted risk − **usi bootstrap base** ka no-shift risk. Two-sided formula ka sab se chhota non-zero p = 2/B; p = 0 ko "< 2/B" report. **v1.1.1 (audit 2026-10-08, results se pehle):** PA ka **B = 10,000** (`exp2.yaml` `pa.b`; pehle 1,000, jis par 2/B = 0.002 Holm ki pehli seedhi 0.05/35 = 0.0014 se upar tha, yani PA test sirf tab pass hota jab saare replicates ek sign dein). Yeh p CI-inversion p hai (null ke under test nahi). PA CIs mein dev se fit hui cheezein (τ, implicit prior, T, BCTS) fixed maani jati hain; dev ki uncertainty CI mein shamil nahi (paper mein likhna). **PA1 ka no-shift control (v1.1.1):** ΔECE = ECE(shifted) − ECE(**resampled no-shift**), no-shift samples usi with-replacement resampler se q = base sample ka apna prior par (M draws); top-label ECE resampled (duplicated) data par upar ki taraf biased hota hai, aur base sample se seedha muqabla shift effect mein resampling bias mila deta tha. PA2 (risk = per-example losses ka mean, resampling se unbiased) base sample hi rakhta hai.
- **P1/P2 (implemented 2026-10-07):** `s1bench primary-pair --test P1|P2 --dataset ...` (pairs `configs/experiments/primary.yaml`); Δ = metric(TDM) − metric(base) test set par, B = 10,000, dono metrics (Macro-F1, ECE) ke liye same bootstrap indices; sirf woh examples jo dono models ke liye ok hon. Macro-F1 Δ > 0 aur ECE Δ < 0 TDM ke haq mein. **Input checks (primary-pair aur primary-pa):** har prediction file poora split cover kare, failed ≤ 1%, statuses sirf ok/failed/blocked; warna test se inkaar (sirf dev check mein chalta hai, `dev_check: true`). Dev check = B ya n_priors default se alag, `dev_` run, ya PA mein Jev/DeBERTa-large ke ilawa koi model. **Holm poori family par:** `s1bench primary-holm --run <id>`: keys bilkul pehle se tay 35 (`primary.yaml` ke datasets × P1/P2 F1/ECE + PA1–PA3) hon aur koi dev-check file na ho tabhi COMPLETE; partial family par m = 35 (conservative); duplicate ya family se bahar test = error. p < 1/B ko "< 1/B" likha jata hai.
- **Reporting:** mean, 95% percentile CI, effect size (difference), Holm-adjusted p.
- **Exploratory:** baaqi sab comparisons, P3, Friedman, McNemar (accuracy), purane P4–P6; "exploratory" label ke saath, Holm ke bahar.

---

## 12. Runs aur randomness 🔒
- Local decoder models: greedy / deterministic scoring; 1 run (determinism 50 examples par 2 runs se verify: `s1bench predict --repeat-check --limit 50`, probabilities exactly same honi chahiye).
- Jev: 1 full run + 3 runs on 500-example subset per dataset; saare Jev runs **ek 72-ghante window** mein; `model` field badla to stop.
- Training seeds: Exp 3 ke upar.
- Global seed registry: `configs/seeds.yaml` (v1.0.4 mein bani; baaqi configs ke seeds se match).

---

## 13. Exclusion aur failure handling 🔒
| Situation | Rule |
|---|---|
| API error / timeout (Jev) | 4 retries (exponential backoff 1.5·2^k s, rate limit 6·2^k s; unexpected response shape par sirf 1 retry; har koshish ki error `raw.attempt_errors` mein); phir bhi fail to `failed` record; **failed > 1% kisi dataset par → poora dataset rerun** (v1.1.1: transport errors (connection refused/reset, timeout, `OSError`) kabhi "blocked" nahi ginte; pehle Windows ka "actively refused" message "refus" hint se block ban jata) |
| **Refusal / safety block / content filter** (khaas kar HateXplain; Jev via Cloudflare) | Alag category, transient error nahi; 1 dafa retry; phir bhi block to prediction = **wrong + abstain** (AURC mein abstain); per-model/per-dataset rate report. **Scope (2026-10-07, review ke baad saaf kiya):** wrong + abstain Exp 1 ki Accuracy, Macro-F1, MCC aur AURC tables par lagta hai. Probability-based paired tests (P1/P2 ECE, PA1–PA3, Exp 2/EM) mein blocked row ki koi probability nahi, is liye wahan "dono models ke liye ok" examples; har primary file mein blocked/failed counts (`input_checks`) report. Kisi dataset par blocked > 1% ho to un examples ko wrong maan kar sensitivity analysis zaroori |
| Model K candidates support na kare | Us model × dataset ko "not supported" report; impute nahi |
| GLiNER input 512 tokens se lamba | Truncated record + count; analysis mein include |
| Out-of-memory | Smaller batch; phir bhi OOM to precision fallback (pair ke dono models ek saath) aur deviation log |
| Invalid/unparseable output (sirf secondary JSON path) | Output validity mein count; primary readout par asar nahi |
| Jev `model` field badla | Run stop; nayi window mein poora Jev rerun |

---

## 14. Outputs aur packaging
- `results/<run_id>/`: `environment.json`, `config_snapshot/` (configs + label + candidate hashes), `predictions/<model>/<dataset>.jsonl` (`id`, candidate order, probability vector, prediction, latency_ms, tokens, raw native output), `metrics/`, `logs/`.
- `results/<run_id>/MANIFEST.json`: har file ka sha256. Laptop par wapas laa kar `s1bench verify-results --run <id>` se check (implemented 2026-10-07): manifest hashes; `.env`/weights/raw data aur `.env` ki secret **values** kisi file mein na hon (values print nahi hoti); candidate order = `data/candidates`, gold = `data/processed`; full split; probabilities (Jev 2-decimal rounding ki gunjaish) aur argmax; failed > 1% → rerun; Jev ka ek hi version (saare runs mein bhi); har environment file ka `config_snapshot` is machine se match: sirf prediction par asar daalne wali files (`datasets/processing/candidates/models.yaml` + data manifests, farq = error) aur prediction code (`code:` keys, farq = warning, git nahi hai is liye); analysis configs (exp2/primary/efficiency, overlap) shamil nahi, taake baad mein unhein badalna purane runs ko invalid na kare. Hash CRLF→LF normalize. Review (2026-10-07) ke baad aur checks: NaN/inf probabilities, non-Jev models par sum tolerance 1e-4, unknown status, candidate file mein id na ho, seeded subset ka poora size, khaali snapshot, secret patterns (bearer/HF token/Cloudflare account URL), budget models ka train environment, smoke run sirf `dev_` run mein; `train_summary.json` bhi manifest mein.
- **Environment layout (2026-10-07 fix):** `environment/<model>/<dataset>_<split>.json` (har `predict` call ki apni file; pehle ek `environment.json` har call par overwrite hota tha). Training: `environment/<model>/train_<datasets>_<UTC>.json`.
- **Efficiency table:** `s1bench efficiency --run ... --out ...` (p50/p95 warm latency, pehli 10 rows warm-up, tokens, GPU-s/1,000, throughput, peak VRAM, API cost/1,000 `configs/experiments/efficiency.yaml` se). Dev runs aur laptop GPU wali rows `paper_eligible = false`.
- **Kabhi shamil nahi:** `.env`, raw dataset files, model weights.

---

## 15. Contamination aur exposure reporting 🔒
"No contamination" kabhi nahi. Wording: *"We cannot fully exclude pretraining contamination; results should be interpreted as benchmark performance rather than evidence of generalization to unseen distributions."* Exposure matrix table + overlap checks (sec 4.5).

---

## 16. Freeze (internal; preregistration/OSF nahi, authors' decision 2026-10-08)
- **Kya freeze hoga:** sections 1, 2 (revisions), 4–6, 8, 9, 10.2, 11, 12, 13.
- **Kab:** pilot ke baad, full runs (test predictions) se pehle. **Kaise:** `docs/FREEZE_v1.1.json` mein tareekh + neeche wali files ke sha256 (git aur OSF nahi; har final run ki `config_snapshot` in hashes se match honi chahiye). **Commands:** `s1bench freeze --check` (run machine par har final run se pehle: folder = freeze); `s1bench verify-results --run <id> --freeze docs/FREEZE_v1.1.json` (laptop par; koi bhi farq, code samet = error). **Paper wording:** "preregistered" nahi likhna; "protocol fixed before test-set evaluation" jaisa. **Freeze package:** yeh spec (pilot notes ke baghair), `configs/`, data/labels/processed/candidates ke MANIFEST files, aur `config_snapshot` ki code hashes (prompts isi code mein, sec 7.2), aur pre-freeze disclosure (FPB test split pipeline checks mein dekha gaya, 2026-10-07; approved by the authors). `docs/deviations.md` ke pre-freeze rows internal; freeze ke baad har tabdeeli usi file mein aur paper ke deviations appendix mein. **Stage 3 amendment:** gpt-oss/GLiNER/paraphrase ki pakki settings Stage 3 runs se pehle alag amendment (naya freeze file) ke tor par.
- **Deviations:** `docs/deviations.md` mein tareekh, kya badla, kyun, aur results dekhne se pehle ya baad.

---

## 17. Pilot plan (100 examples per dataset, seeded)

| # | Check | Pass criterion |
|---|---|---|
| 1 | Data processing + cleaning rules | Counts expected; manifests ban gaye |
| 2 | Candidate files | Gold position ~uniform; K sahi; same file sab models |
| 3 **(sab se pehle)** | Jev via Cloudflare | Sab calls successful; `model` = `jev-1.13.0`; probabilities sum ≈ 1; asal cost/1,000 |
| 4 | Jev FPB sanity | Pilot accuracy Deußer ke 0.730 se ballpark mein (full run par confirm) |
| 5 | Tev1/Qwen-4B prompt + prefix | Letter mass > 0.9 candidates par; scoring argmax = greedy JSON (mostly) |
| 6 | Nimble adapter (`original-2676` @ fe33f82b; aur exploratory `main` @ bd792f44) + Qwen-9B 8-bit | 12 GB mein fit; same base hash |
| 7 *(Stage 3)* | gpt-oss-20b llama-cpp-python | Prompt logprobs milte hain; final-channel position sahi |
| 8 *(Stage 3)* | GLiNER | 20 labels + descriptions fit; confidence output milta hai |
| 9 | Throughput / time | Full run aur Exp 3 ka time estimate |
| 10 | Environment capture + results packaging | `environment.json` complete; manifest verify pass |
| 11 | Jev/Cloudflare HateXplain par block/refusal | Block rate naapna; rule (sec 13) apply |
| 12 | Tev1 K = 18–22 (groups) par full-vocab letter mass | Candidate letters par mass report (OOD check) |

Pilot ke results **paper mein nahi** jayenge (sirf protocol fix karne ke liye).

Pilot ke nateeje aur tafseel internal file `docs/pilot_notes.md` mein (paper ka hissa nahi).

---

## 18. Open items (v1.1)
1. ~~Label texts~~ → **frozen v1.0 (2026-10-06)**, `data/labels/*.yaml` + `MANIFEST.json`.
2. ~~Tev1 weights license; Tev1 kis Qwen revision se bana~~ → checked 2026-10-08 (sec 2): license abhi "being finalized" (disclose); config `851bf6e8` se identical, exact base commit published nahi.
3. ~~Nimble prompt~~ → laptop par confirm: `original-2676` aur `main` dono ka prompt token-identical (sec 2).
4. GLiNER per-label scores → **Stage 3 amendment**.
5. Paraphrase generator model → **Stage 3 amendment**.
6. ~~DeBERTa max length aur epochs~~ → 128; 5/20 epochs + min 500 steps (sec 10.3). QLoRA steps rule frozen (sec 10.3), Stage 2 se pehle chhota pilot.
7. ~~Run machine specs~~ → sec 3.
8. ~~OSF account~~ → OSF/preregistration nahi (authors' decision 2026-10-08); internal freeze (sec 16).
9. ~~Exp 2 without-replacement~~ → simulated (sec 10.2); exploratory hi rahega.
10. ~~DeBERTa LR aur min-steps~~ → grid {1e-5, 2e-5} freeze; min-steps rule check ho gaya.
11. ~~Efficiency~~ → implement (authors' decision 2026-10-08) aur check ho gaya (sec 9).
12. **Jev cost/1,000 (authors):** Cloudflare dashboard se (2026-10-08 ke 500 calls, ~275k input tokens) → `configs/experiments/efficiency.yaml`. Paper mein final runs ki billing se.
13. ~~Proposed items~~ → approved by the authors 2026-10-08: sec 7.2 (prompts = code hashes), sec 10.3 (DeBERTa 5/20 epochs; QLoRA steps rule).

---

## Appendix: Sanity reference bands (dataset reviews se)

| Dataset | Reference (fine-tuned, full data) | Zero-shot reference | Note |
|---|---|---|---|
| CLINC150 | ~96.9–97.2 accuracy (150-way in-scope; Larson 2019, Casanueva 2020) | Jev 0.895 (151-way + OOS, Deußer); 0.913 (150, Rafe & Das); 0.975 (20 random options, Rafe & Das) | Hamara K = 20 hard-negative, seedha compare nahi |
| HWU64 | ~92.1–92.9 accuracy (64-way; Casanueva 2020, Mehri 2020) | — | DialoGLUE-DG numbers mein test text pretraining mein ho sakta hai |
| FPB | FinBERT 0.86 (50%+); FLANG-ELECTRA 0.919 | **Jev 0.730 (same 970 test, Deußer)**; GPT-4 0.83, ChatGPT 0.78 (Li 2023) | Jev sanity check |
| HateXplain | BERT Macro-F1 0.674; best 0.699 | gpt-3.5 0.39, flan-T5 0.59 Macro-F1 (Roy 2023) | α = 0.46 noise ceiling |
| TREC-6 | RoBERTa-large 97.4 | GPT-3 175B 47.4 (0-shot) | Ceiling ke qareeb; parity claims nahi |

Agar hamara DeBERTa-v3-large full-data in bands se bohat neeche aaye to pipeline check karni hai.
