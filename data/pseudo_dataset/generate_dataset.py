"""
Pseudo Dataset Generator for Complex Lymphatic Anomalies (CLA)
==============================================================
Generates a realistic curated corpus of rare disease documents
for benchmarking RAG pipelines. Covers:
  - Gorham-Stout Disease (GSD)
  - Generalized Lymphatic Anomaly (GLA)
  - Central Conducting Lymphatic Anomaly (CCLA)
  - Kaposiform Lymphangiomatosis (KLA)
  - Lymphangioleiomyomatosis (LAM)
  - General CLA epidemiology, diagnosis, and treatment

Usage:
    python generate_dataset.py
    python generate_dataset.py --output custom_path.json --n_extra 20
"""

import json
import argparse
import random
from datetime import datetime, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# Curated pseudo documents – realistic clinical / scientific content
# ---------------------------------------------------------------------------

DOCUMENTS: list[dict] = [
    # -----------------------------------------------------------------------
    # 1. EPIDEMIOLOGY & OVERVIEW
    # -----------------------------------------------------------------------
    {
        "id": "doc_001",
        "title": "Complex Lymphatic Anomalies: An Overview of Classification and Epidemiology",
        "source_type": "review_article",
        "disease_entity": "CLA",
        "year": 2022,
        "journal": "Orphanet Journal of Rare Diseases",
        "authors": ["Smith J", "Lee K", "Patel R"],
        "abstract": (
            "Complex lymphatic anomalies (CLAs) represent a heterogeneous group of rare "
            "disorders characterized by abnormal lymphatic vessel development and function. "
            "CLAs affect approximately 1 in 100,000 individuals, with no significant sex "
            "predilection except for lymphangioleiomyomatosis (LAM), which almost exclusively "
            "affects women of childbearing age. The International Society for the Study of "
            "Vascular Anomalies (ISSVA) 2018 classification groups CLAs into Gorham-Stout "
            "disease (GSD), generalized lymphatic anomaly (GLA), central conducting lymphatic "
            "anomaly (CCLA), kaposiform lymphangiomatosis (KLA), and LAM. Diagnosis is typically "
            "delayed by 3–7 years due to phenotypic overlap with other vascular and bone "
            "disorders. Genomic studies have identified somatic mutations in the PI3K/AKT/mTOR "
            "and RAS/MAPK signaling pathways as major drivers of lymphatic endothelial cell "
            "dysregulation in CLAs."
        ),
        "full_text": (
            "Complex lymphatic anomalies (CLAs) represent a heterogeneous group of rare "
            "disorders characterized by abnormal lymphatic vessel development and function. "
            "CLAs affect approximately 1 in 100,000 individuals, with no significant sex "
            "predilection except for lymphangioleiomyomatosis (LAM), which almost exclusively "
            "affects women of childbearing age.\n\n"
            "The International Society for the Study of Vascular Anomalies (ISSVA) 2018 "
            "classification groups CLAs into five major subtypes: (1) Gorham-Stout disease "
            "(GSD), characterized by progressive osteolysis; (2) generalized lymphatic anomaly "
            "(GLA), featuring multifocal lymphatic malformations involving bone and soft tissue; "
            "(3) central conducting lymphatic anomaly (CCLA), involving dysfunction of the "
            "thoracic duct and cisterna chyli; (4) kaposiform lymphangiomatosis (KLA), an "
            "aggressive multisystem disorder; and (5) lymphangioleiomyomatosis (LAM), driven "
            "by TSC2 mutations.\n\n"
            "Diagnosis is typically delayed by 3–7 years due to phenotypic overlap with other "
            "vascular and bone disorders. Genomic studies have identified somatic mutations in "
            "the PI3K/AKT/mTOR and RAS/MAPK signaling pathways as major drivers. The mutation "
            "PIK3CA (p.H1047R) is found in approximately 25% of GLA cases, while NRAS and KRAS "
            "mutations are identified in KLA. TSC1/TSC2 mutations are the hallmark of LAM, "
            "leading to mTORC1 hyperactivation and abnormal smooth muscle cell proliferation."
        ),
        "keywords": ["CLA", "classification", "epidemiology", "ISSVA", "PI3K", "mTOR", "rare disease"],
    },

    # -----------------------------------------------------------------------
    # 2. GORHAM-STOUT DISEASE
    # -----------------------------------------------------------------------
    {
        "id": "doc_002",
        "title": "Gorham-Stout Disease: Pathogenesis, Clinical Features, and Emerging Therapies",
        "source_type": "review_article",
        "disease_entity": "GSD",
        "year": 2023,
        "journal": "Journal of Bone and Mineral Research",
        "authors": ["Gorham LW", "Wallace AC", "Chen M"],
        "abstract": (
            "Gorham-Stout disease (GSD), also known as vanishing bone disease or massive "
            "osteolysis, is an ultra-rare disorder characterized by progressive replacement "
            "of bone by lymphovascular tissue. The global prevalence is estimated at fewer "
            "than 300 published cases. The pathognomonic feature is the invasion of intraosseous "
            "lymphatic capillaries that progressively destroy cortical and trabecular bone. "
            "Complications include chylous pleural effusion, chylothorax, spinal instability, "
            "and neurological compromise. Treatment options include bisphosphonates, sirolimus, "
            "bevacizumab, interferon-alpha, and radiotherapy. Sirolimus (rapamycin) has shown "
            "particular promise by inhibiting mTORC1 signaling in lymphatic endothelial cells."
        ),
        "full_text": (
            "Gorham-Stout disease (GSD), also known as vanishing bone disease or massive "
            "osteolysis, is an ultra-rare disorder characterized by progressive replacement "
            "of bone by lymphovascular tissue. Fewer than 300 cases have been reported in "
            "the global literature since Gorham and Stout's landmark 1955 description.\n\n"
            "PATHOGENESIS: The cardinal pathological feature is proliferating intraosseous "
            "lymphatic capillaries that secrete osteoclast-activating cytokines including "
            "IL-6 and VEGF-C, leading to progressive destruction of both cortical and "
            "trabecular bone. VEGFR-3 overexpression on aberrant lymphatic endothelium "
            "drives lymphangiogenesis. Somatic PIK3CA mutations activating the PI3K/AKT/mTOR "
            "axis have been identified in a subset of cases.\n\n"
            "CLINICAL FEATURES: GSD can affect any bone but has a predilection for the "
            "shoulder girdle, chest wall, and spine. Chylothorax occurs in 17% of cases "
            "and carries a mortality risk exceeding 50% if untreated. Spinal GSD may cause "
            "paraplegia or quadriplegia. Skull base involvement can cause cranial nerve palsies.\n\n"
            "DIAGNOSIS: Diagnosis relies on the combination of radiographic findings (cortical "
            "erosion, intramedullary lucencies), histopathology showing thin-walled lymphatic "
            "channels replacing bone, and exclusion of malignancy. MRI with lymphangiography "
            "using intranodal injection is increasingly used to map lymphatic anatomy.\n\n"
            "TREATMENT: No randomized clinical trials exist for GSD. Current treatment "
            "strategies include: (1) Sirolimus (rapamycin) – the mTOR inhibitor most widely "
            "used, typical dose 1–3 mg/m²/day targeting trough level 5–15 ng/mL; (2) "
            "Bisphosphonates (zoledronate, pamidronate) to reduce osteoclast activity; "
            "(3) Bevacizumab (anti-VEGF) for refractory cases; (4) Radiotherapy 40–45 Gy "
            "for localized disease; (5) Surgical stabilization for spinal instability. "
            "Combination sirolimus + bisphosphonate therapy is increasingly preferred."
        ),
        "keywords": ["GSD", "vanishing bone", "osteolysis", "chylothorax", "sirolimus", "bisphosphonates", "VEGFR-3"],
    },

    # -----------------------------------------------------------------------
    # 3. GENERALIZED LYMPHATIC ANOMALY
    # -----------------------------------------------------------------------
    {
        "id": "doc_003",
        "title": "Generalized Lymphatic Anomaly: Multiorgan Involvement and PIK3CA Mutations",
        "source_type": "original_research",
        "disease_entity": "GLA",
        "year": 2021,
        "journal": "American Journal of Human Genetics",
        "authors": ["Adams S", "Rodriguez B", "Kim H"],
        "abstract": (
            "Generalized lymphatic anomaly (GLA) is defined by multifocal lymphatic "
            "malformations involving bone, soft tissue, and visceral organs without the "
            "progressive osteolysis seen in GSD. We report somatic PIK3CA mutations in "
            "28 of 45 GLA patients (62%), with p.H1047R being the most common variant. "
            "Organs involved include spleen (78%), bone (67%), mediastinum (52%), and lung "
            "(34%). Sirolimus therapy resulted in stable disease or partial response in 83% "
            "of treated patients over 12 months."
        ),
        "full_text": (
            "Generalized lymphatic anomaly (GLA) is defined by multifocal lymphatic "
            "malformations involving bone, soft tissue, and visceral organs. Unlike GSD, "
            "GLA does not cause progressive osteolysis; bone lesions are typically lytic "
            "but non-expansile.\n\n"
            "GENETICS: We performed next-generation sequencing on lesional tissue from 45 "
            "patients. Somatic PIK3CA mutations were identified in 28/45 (62%), with "
            "p.H1047R (n=15), p.E545K (n=8), and p.E542K (n=5) being the predominant "
            "variants. These gain-of-function mutations hyperactivate the PI3K/AKT/mTOR "
            "cascade in lymphatic endothelial cells.\n\n"
            "CLINICAL PRESENTATION: Spleen involvement (78%) manifests as splenomegaly "
            "and hypersplenism. Bone lesions (67%) are predominantly in the axial skeleton. "
            "Mediastinal involvement (52%) can cause chylous pericardial effusion. Pulmonary "
            "involvement (34%) includes chylous pleural effusion and pulmonary lymphatic "
            "perfusion syndrome (PLPS). A minority of patients (12%) develop protein-losing "
            "enteropathy from intestinal lymphangiectasia.\n\n"
            "TREATMENT OUTCOMES: Sirolimus (target trough 5–15 ng/mL) was administered to "
            "36 patients for median 18 months. Stable disease was achieved in 22/36 (61%) "
            "and partial response in 8/36 (22%). Alpelisib (PI3Kα inhibitor) was used in "
            "4 PIK3CA-mutated patients with promising preliminary results. Response was "
            "assessed by MRI volumetry of lymphatic lesions and serum VEGF-D levels."
        ),
        "keywords": ["GLA", "PIK3CA", "lymphatic anomaly", "sirolimus", "alpelisib", "mTOR", "multifocal"],
    },

    # -----------------------------------------------------------------------
    # 4. KAPOSIFORM LYMPHANGIOMATOSIS
    # -----------------------------------------------------------------------
    {
        "id": "doc_004",
        "title": "Kaposiform Lymphangiomatosis: A Distinct Aggressive Subtype of CLA",
        "source_type": "case_series",
        "disease_entity": "KLA",
        "year": 2020,
        "journal": "Haematologica",
        "authors": ["Fernandez M", "Park J", "Wilson T"],
        "abstract": (
            "Kaposiform lymphangiomatosis (KLA) is a rare and aggressive form of CLA "
            "characterized by abnormal spindle-shaped lymphatic endothelial cells in a "
            "kaposiform pattern, thrombocytopenia, coagulopathy, and hemorrhagic pleural "
            "and pericardial effusions. Mortality rates historically exceed 50%. NRAS and "
            "KRAS activating mutations in the MAPK pathway are identified in the majority "
            "of KLA cases. We describe a cohort of 18 patients treated with trametinib "
            "(MEK inhibitor) with objective response rate of 67%."
        ),
        "full_text": (
            "Kaposiform lymphangiomatosis (KLA) is a rare and aggressive form of CLA "
            "distinguished from GLA and GSD by the presence of spindle-shaped lymphatic "
            "endothelial cells arranged in a kaposiform (slit-like) pattern, admixed with "
            "hemosiderin-laden macrophages.\n\n"
            "PATHOGENESIS: NRAS (codons 12/13/61) and KRAS mutations activating the RAS/"
            "MAPK/ERK pathway are identified in 60–70% of KLA cases. These mutations "
            "promote aberrant lymphatic endothelial proliferation and impair lymphatic "
            "valve formation. The MAPK cascade phosphorylates MEK1/2 and ERK1/2, driving "
            "cell survival and angiogenic signaling.\n\n"
            "CLINICAL FEATURES: KLA predominantly affects the mediastinum, lung, bone, "
            "spleen, and abdomen. The hallmark laboratory findings are thrombocytopenia "
            "(median platelets 42 × 10⁹/L), consumptive coagulopathy (elevated D-dimer, "
            "low fibrinogen), and hemorrhagic effusions rich in chylomicrons. "
            "Hemoptysis occurs in 28% of patients.\n\n"
            "DIAGNOSIS: Histopathology is mandatory. Immunohistochemistry shows lymphatic "
            "endothelial markers (LYVE-1+, PROX1+, D2-40+) in both the dilated channels "
            "and the spindle cells. CD34 is focally positive. Ki-67 proliferation index "
            "is 5–15%, distinguishing KLA from Kaposi sarcoma (Ki-67 >30%).\n\n"
            "TREATMENT: Trametinib (1–2 mg/day in adults) was administered to 18 patients "
            "harboring NRAS/KRAS mutations. Objective response (>25% lesion volume reduction "
            "by MRI) was observed in 12/18 (67%). Sirolimus was added in 6 non-responders. "
            "Vincristine + corticosteroids remain first-line for KLA-associated "
            "coagulopathy/thrombocytopenia during acute crises. Bevacizumab is used "
            "adjunctively for hemorrhagic effusion control."
        ),
        "keywords": ["KLA", "kaposiform", "NRAS", "KRAS", "trametinib", "MEK inhibitor", "coagulopathy", "thrombocytopenia"],
    },

    # -----------------------------------------------------------------------
    # 5. CENTRAL CONDUCTING LYMPHATIC ANOMALY
    # -----------------------------------------------------------------------
    {
        "id": "doc_005",
        "title": "Central Conducting Lymphatic Anomaly: Diagnosis with Dynamic Contrast MR Lymphangiography",
        "source_type": "original_research",
        "disease_entity": "CCLA",
        "year": 2023,
        "journal": "Radiology",
        "authors": ["Itkin M", "Johnson M", "Bhatt S"],
        "abstract": (
            "Central conducting lymphatic anomaly (CCLA) encompasses disorders of the "
            "central lymphatic conduits including thoracic duct aplasia, hypoplasia, and "
            "abnormal lymphatic flow patterns leading to protein-losing enteropathy, "
            "chylous ascites, and pulmonary lymphatic perfusion syndrome. Dynamic "
            "contrast-enhanced MR lymphangiography (DCMRL) following intranodal injection "
            "of gadofosveset trisodium enables non-invasive mapping of central lymphatic "
            "flow, identifying retrograde flow patterns and decompressing fistulae in 94% "
            "of cases. Thoracic duct embolization (TDE) achieves clinical resolution in "
            "72% of patients with chylous leak."
        ),
        "full_text": (
            "Central conducting lymphatic anomaly (CCLA) is defined by malformation or "
            "dysfunction of the central lymphatic conduits: the cisterna chyli, thoracic "
            "duct, and right lymphatic duct. This results in lymphatic hypertension and "
            "retrograde or abnormal flow into pulmonary and intestinal lymphatics.\n\n"
            "PATHOPHYSIOLOGY: In NOONAN syndrome-associated CCLA (PTPN11, SOS1 mutations), "
            "defective lymphatic valve formation leads to reflux of lymph from the thoracic "
            "duct into pulmonary lymphatics, causing pulmonary lymphatic perfusion syndrome "
            "(PLPS) with diffuse pulmonary interstitial edema. In thoracic duct aplasia, "
            "lymph cannot drain into the venous system and decompresses via collateral "
            "channels causing pleural and pericardial effusions.\n\n"
            "IMAGING: Dynamic contrast-enhanced MR lymphangiography (DCMRL) is performed "
            "by bilateral intranodal injection of gadofosveset (0.1 mL/kg) in the inguinal "
            "nodes under fluoroscopic guidance, followed by T1-weighted MR imaging during "
            "the venous phase. DCMRL delineates central lymphatic anatomy, identifies "
            "thoracic duct patency, locates chylous fistulae, and demonstrates retrograde "
            "pulmonary lymphatic flow.\n\n"
            "INTERVENTIONAL PROCEDURES: Thoracic duct embolization (TDE) via transpedal "
            "lymphangiography + coil/glue embolization achieves chylous leak resolution "
            "in 72% of cases. For PLPS, thoracic duct decompression by lymphovenous "
            "anastomosis or balloon dilation of lymphovenous junction obstruction improves "
            "symptoms. Lymphatic reconstructive surgery is emerging as a definitive option."
        ),
        "keywords": ["CCLA", "thoracic duct", "DCMRL", "chylothorax", "thoracic duct embolization", "lymphangiography", "PLPS"],
    },

    # -----------------------------------------------------------------------
    # 6. LYMPHANGIOLEIOMYOMATOSIS (LAM)
    # -----------------------------------------------------------------------
    {
        "id": "doc_006",
        "title": "Lymphangioleiomyomatosis: mTOR-Targeted Therapy and Lung Transplantation Outcomes",
        "source_type": "clinical_trial",
        "disease_entity": "LAM",
        "year": 2022,
        "journal": "New England Journal of Medicine",
        "authors": ["Taveira-DaSilva A", "Moss J", "McCormack FX"],
        "abstract": (
            "Lymphangioleiomyomatosis (LAM) is a rare cystic lung disease caused by "
            "inactivating mutations in TSC1 or TSC2, leading to mTORC1 hyperactivation "
            "in smooth muscle-like LAM cells. LAM affects almost exclusively women, with "
            "a prevalence of approximately 3.4–7.8 per million. Sirolimus stabilizes lung "
            "function (FEV₁ decline: −12 mL/year vs −134 mL/year placebo; MILES trial) "
            "and is FDA-approved for LAM. Lung transplantation is indicated for FEV₁ <30% "
            "predicted, with 5-year survival of 55–65%."
        ),
        "full_text": (
            "Lymphangioleiomyomatosis (LAM) is a rare cystic lung disease characterized "
            "by progressive proliferation of smooth muscle-like LAM cells in the lung "
            "parenchyma, lymphatics, and kidneys (angiomyolipomas). LAM occurs sporadically "
            "or in association with tuberous sclerosis complex (TSC-LAM vs. S-LAM).\n\n"
            "GENETICS: TSC1 (hamartin) or TSC2 (tuberin) biallelic loss-of-function "
            "mutations cause constitutive mTORC1 activation. mTORC1 phosphorylates S6K1 "
            "and 4EBP1, driving abnormal smooth muscle cell proliferation. VEGF-D is "
            "elevated in serum of LAM patients (>800 pg/mL is diagnostic) and correlates "
            "with lymphangioleiomyomas and chylous effusions.\n\n"
            "CLINICAL FEATURES: Dyspnea and spontaneous pneumothorax (occur in 66% of "
            "patients) are the most common presentations. HRCT demonstrates bilateral "
            "diffuse thin-walled cysts. Renal angiomyolipomas occur in 30–50% (higher "
            "in TSC-LAM). Chylothorax and chylous ascites develop in 10–15%.\n\n"
            "TREATMENT: Sirolimus is the standard of care, FDA-approved since 2015 based "
            "on the MILES trial. Typical dosing: 2 mg/day (adjust to trough 5–15 ng/mL). "
            "Everolimus (2.5–10 mg/day) is an alternative mTOR inhibitor. Treatment "
            "stabilizes but does not reverse lung function decline; FEV₁ decline on "
            "sirolimus is −12 mL/year vs −134 mL/year on placebo. Pneumothorax: pleurodesis "
            "is recommended after second episode. Lung transplantation (bilateral) is "
            "considered for FEV₁ <30% predicted with rapid decline; LAM may recur in "
            "transplanted lungs."
        ),
        "keywords": ["LAM", "TSC2", "mTOR", "sirolimus", "everolimus", "cystic lung", "VEGF-D", "pneumothorax", "MILES trial"],
    },

    # -----------------------------------------------------------------------
    # 7. SIROLIMUS IN CLA – SYSTEMATIC REVIEW
    # -----------------------------------------------------------------------
    {
        "id": "doc_007",
        "title": "Sirolimus Therapy Across Complex Lymphatic Anomalies: A Systematic Review and Meta-Analysis",
        "source_type": "systematic_review",
        "disease_entity": "CLA",
        "year": 2024,
        "journal": "Lancet Haematology",
        "authors": ["Ozeki M", "Nozawa A", "Fukao T"],
        "abstract": (
            "We conducted a systematic review of sirolimus (rapamycin) therapy in all CLA "
            "subtypes. Across 47 studies comprising 389 patients, the overall objective "
            "response rate (partial + complete response) was 76% (95% CI 71–81%). Response "
            "rates by subtype: LAM 85%, GLA 79%, KLA 68%, GSD 71%, CCLA 64%. Median time "
            "to response was 3.2 months. The most common adverse effects were stomatitis "
            "(38%), infection (24%), and hyperlipidemia (19%). Drug discontinuation for "
            "toxicity occurred in 8% of patients."
        ),
        "full_text": (
            "Sirolimus (rapamycin) inhibits the mechanistic target of rapamycin complex 1 "
            "(mTORC1) by binding FK-binding protein 12 (FKBP12), thereby suppressing "
            "lymphatic endothelial cell proliferation, survival, and VEGF-C/D production.\n\n"
            "METHODS: We searched PubMed, EMBASE, and Cochrane from 2000–2024. Primary "
            "outcome: objective response rate (ORR) defined as ≥25% reduction in lesion "
            "volume or resolution of effusion. Secondary outcomes: time to response, "
            "adverse effects, and recurrence rate after discontinuation.\n\n"
            "RESULTS: 47 studies (7 prospective, 40 retrospective) with 389 patients "
            "(median age 14.2 years, range 0.2–72). ORR across all CLAs: 76% (295/389). "
            "By subtype: LAM 85%, GLA 79%, GSD 71%, KLA 68%, CCLA 64%. Median time to "
            "first response: 3.2 months (IQR 2–6). Complete response was rare (8%) across "
            "all subtypes. Relapse after sirolimus discontinuation: 62% within 12 months, "
            "supporting indefinite treatment in most patients.\n\n"
            "DOSING: Pediatric dosing: 0.8–2.4 mg/m²/day targeting trough 5–15 ng/mL. "
            "Adult dosing: 1–5 mg/day. Monitoring: CBC, LFTs, lipid panel, trough levels "
            "every 2–4 weeks initially, then every 3 months when stable.\n\n"
            "ADVERSE EFFECTS: Stomatitis (38%) – managed with steroid mouthwash. "
            "Infection (24%) – prophylactic TMP-SMX for PCP recommended. "
            "Hyperlipidemia (19%) – statin therapy if needed. "
            "Impaired wound healing (11%). Pneumonitis (3%) – requires dose reduction."
        ),
        "keywords": ["sirolimus", "rapamycin", "mTOR", "CLA", "treatment", "response rate", "adverse effects"],
    },

    # -----------------------------------------------------------------------
    # 8. BIOMARKERS AND DIAGNOSIS
    # -----------------------------------------------------------------------
    {
        "id": "doc_008",
        "title": "Serum and Imaging Biomarkers for Complex Lymphatic Anomalies",
        "source_type": "original_research",
        "disease_entity": "CLA",
        "year": 2023,
        "journal": "Journal of Clinical Investigation",
        "authors": ["Brouillard P", "Vikkula M", "Boon L"],
        "abstract": (
            "Reliable biomarkers for CLA diagnosis and treatment monitoring are lacking. "
            "We evaluated serum VEGF-D, VEGF-C, Ang-2, and soluble LYVE-1 in 87 CLA "
            "patients and 40 healthy controls. VEGF-D discriminated LAM from other CLAs "
            "with AUC 0.94. VEGF-C was elevated in GLA and KLA (median 3.2× normal). "
            "Ang-2 correlated with disease activity score (r=0.71). On sirolimus therapy, "
            "VEGF-D decreased by 42% from baseline at 6 months, serving as an early "
            "pharmacodynamic biomarker."
        ),
        "full_text": (
            "Reliable biomarkers for CLA diagnosis, disease monitoring, and treatment "
            "response assessment are critically needed given the rarity of these conditions "
            "and the challenges of MRI volumetry in multicentric disease.\n\n"
            "BIOMARKERS STUDIED:\n"
            "• VEGF-D: Vascular endothelial growth factor D is produced by LAM cells "
            "  and stimulates lymphangiogenesis via VEGFR-3. Serum VEGF-D >800 pg/mL "
            "  has sensitivity 73%, specificity 100% for LAM diagnosis. In our cohort, "
            "  VEGF-D was also elevated in 45% of KLA patients (mean 620 pg/mL).\n"
            "• VEGF-C: Elevated in GLA (3.4×) and KLA (2.9×) but not GSD or CCLA. "
            "  VEGF-C reflects lymphangiogenic activity and correlates with lesion burden.\n"
            "• Angiopoietin-2 (Ang-2): Destabilizes endothelial junctions and promotes "
            "  angiogenesis. Ang-2 correlated with disease activity score (r=0.71) and "
            "  was highest in KLA during acute flares (median 8.4 ng/mL vs. 1.2 normal).\n"
            "• Soluble LYVE-1: A lymphatic endothelial cell surface receptor for hyaluronan. "
            "  Elevated in 60% of GLA and GSD patients.\n\n"
            "MRI RESPONSE ASSESSMENT: Lesion volume reduction ≥25% on MRI at 6 months "
            "correlates with clinical improvement (sensitivity 81%, specificity 79%). "
            "Diffusion-weighted MRI (ADC mapping) may detect early treatment response "
            "before volumetric changes. FDG-PET is not routinely recommended but may "
            "help distinguish active from quiescent lesions in GLA."
        ),
        "keywords": ["VEGF-D", "VEGF-C", "angiopoietin-2", "biomarkers", "MRI", "LAM", "CLA", "treatment monitoring"],
    },

    # -----------------------------------------------------------------------
    # 9. PEDIATRIC CLA MANAGEMENT
    # -----------------------------------------------------------------------
    {
        "id": "doc_009",
        "title": "Management of Complex Lymphatic Anomalies in Children: A Multidisciplinary Approach",
        "source_type": "clinical_guideline",
        "disease_entity": "CLA",
        "year": 2023,
        "journal": "Pediatrics",
        "authors": ["Drolet BA", "Gupta A", "Liang MG"],
        "abstract": (
            "Pediatric CLAs present unique diagnostic and therapeutic challenges. This "
            "guideline from the Society for Pediatric Dermatology and the American Academy "
            "of Pediatrics recommends multidisciplinary team (MDT) management including "
            "interventional radiology, hematology-oncology, pulmonology, and genetics. "
            "Genetic testing for PIK3CA (GLA), NRAS/KRAS (KLA), TSC1/TSC2 (LAM), and "
            "PTPN11/SOS1 (Noonan/CCLA) is recommended. Sirolimus is recommended as "
            "first-line systemic therapy for all CLA subtypes in children with systemic "
            "involvement, starting at 0.8 mg/m²/day."
        ),
        "full_text": (
            "Pediatric complex lymphatic anomalies require coordinated care from specialists "
            "including: interventional radiology (MR lymphangiography, sclerotherapy, TDE), "
            "hematology-oncology (sirolimus, trametinib, chemotherapy for KLA), pulmonology "
            "(management of chylothorax, pleural effusion), genetics (germline and somatic "
            "mutation analysis), orthopedics (GSD spinal stabilization), and dietetics "
            "(lymphatic diet: medium-chain triglyceride supplementation for chylous leaks).\n\n"
            "GENETIC TESTING RECOMMENDATIONS:\n"
            "• GLA: Panel including PIK3CA, PIK3R1, AKT1, MTOR\n"
            "• KLA: Panel including NRAS, KRAS, RAF1\n"
            "• LAM: TSC1, TSC2 germline sequencing; urinalysis for renal AMLs\n"
            "• GSD: PIK3CA somatic testing from lesional biopsy\n"
            "• CCLA/Noonan: PTPN11, SOS1, RAF1, RIT1 germline panel\n\n"
            "SIROLIMUS DOSING IN CHILDREN:\n"
            "Starting dose: 0.8 mg/m²/day (≤40 kg) or 1 mg/m²/day (>40 kg)\n"
            "Target trough: 5–15 ng/mL; check every 2 weeks × 3, then monthly × 3, "
            "then every 3 months\n"
            "Maximum dose: 3 mg/m²/day\n"
            "Duration: Indefinite for systemic disease; minimum 2 years for isolated "
            "lesions with complete response\n\n"
            "SPECIAL CONSIDERATIONS:\n"
            "• Neonates: Sirolimus dosing poorly characterized; start 0.05 mg/kg/day\n"
            "• Growth: Annual height/weight monitoring; sirolimus may impair growth in "
            "  prepubertal children – consider dose reduction or drug holidays\n"
            "• Vaccinations: Avoid live vaccines during sirolimus therapy\n"
            "• Surgery: Hold sirolimus 7–14 days pre/post major surgery (impaired wound healing)"
        ),
        "keywords": ["pediatric CLA", "sirolimus dosing", "multidisciplinary", "genetics", "guidelines", "children"],
    },

    # -----------------------------------------------------------------------
    # 10. CHYLOTHORAX MANAGEMENT
    # -----------------------------------------------------------------------
    {
        "id": "doc_010",
        "title": "Chylothorax in Complex Lymphatic Anomalies: Mechanisms, Diagnosis, and Treatment",
        "source_type": "review_article",
        "disease_entity": "CLA",
        "year": 2022,
        "journal": "Chest",
        "authors": ["McGrath EE", "Blades Z", "Anderson PB"],
        "abstract": (
            "Chylothorax is a life-threatening complication affecting 20–40% of patients "
            "with CLA, particularly GSD (17%), GLA (25%), and KLA (45%). Chyle is composed "
            "of chylomicrons, lymphocytes, fat-soluble vitamins, and immunoglobulins. "
            "Large-volume chylothorax causes malnutrition, immunodeficiency, and respiratory "
            "compromise. Management includes dietary modification (low-fat/MCT diet or NPO "
            "with TPN), octreotide, thoracic duct embolization, and sirolimus. Pleurodesis "
            "is reserved for non-CLA-related chylothorax."
        ),
        "full_text": (
            "Chylothorax occurs when lymph (chyle) accumulates in the pleural space due to "
            "thoracic duct disruption, obstruction, or lymphatic-pleural fistula formation.\n\n"
            "PATHOPHYSIOLOGY IN CLA:\n"
            "• GSD: Lymphatic invasion of rib and vertebral bone creates fistulous connections\n"
            "• GLA: Mediastinal lymphatic malformations directly communicate with pleural space\n"
            "• KLA: Hemorrhagic lymph leaks from abnormal kaposiform channels\n"
            "• CCLA: Retrograde thoracic duct flow increases lymphatic pressure beyond "
            "  pleural drainage capacity\n\n"
            "DIAGNOSIS: Pleural fluid triglycerides >110 mg/dL confirms chylothorax. "
            "Lymphocyte predominance (>80%) on cell differential. Sudan III stain positive "
            "for chylomicrons. Lipoprotein electrophoresis confirms chylomicron band.\n\n"
            "DIETARY MANAGEMENT:\n"
            "• Low-fat diet with MCT supplementation (MCT oil does not form chylomicrons, "
            "  absorbed directly via portal vein)\n"
            "• Neonates/infants: specialized MCT-based formula (Portagen, Monogen)\n"
            "• Severe cases: NPO + parenteral nutrition to rest lymphatic flow\n"
            "• Fat-soluble vitamin supplementation (A, D, E, K) is essential\n\n"
            "PHARMACOLOGICAL:\n"
            "• Octreotide (somatostatin analog): 1–10 μg/kg/hour IV infusion or "
            "  10–30 μg TID subcutaneous; reduces splanchnic flow and thoracic duct output\n"
            "• Sirolimus: mTOR inhibition reduces lymphangiogenesis and effusion volume; "
            "  response typically within 4–12 weeks\n\n"
            "PROCEDURAL:\n"
            "• Thoracentesis: diagnostic and therapeutic drainage\n"
            "• Thoracic duct embolization (TDE): 72% success rate in CCLA-related chylothorax\n"
            "• Pleurodesis (chemical or mechanical): NOT recommended in CLA as it does not "
            "  address the underlying lymphatic malformation and risks lymphatic hypertension"
        ),
        "keywords": ["chylothorax", "MCT diet", "octreotide", "thoracic duct embolization", "chyle", "CLA", "pleural effusion"],
    },

    # -----------------------------------------------------------------------
    # 11. CASE REPORT: GSD WITH CHYLOTHORAX
    # -----------------------------------------------------------------------
    {
        "id": "doc_011",
        "title": "Case Report: Gorham-Stout Disease Presenting as Refractory Chylothorax – Response to Combined Sirolimus and Bevacizumab",
        "source_type": "case_report",
        "disease_entity": "GSD",
        "year": 2023,
        "journal": "BMC Rare Diseases",
        "authors": ["Harrison A", "Nguyen P", "Martinez C"],
        "abstract": (
            "A 16-year-old male presented with progressive dyspnea and a 4-liter left "
            "chylothorax. CT chest showed destruction of the 4th–7th left ribs with "
            "pleural lymphatic fistula. Bone biopsy confirmed Gorham-Stout disease. "
            "Sirolimus monotherapy partially reduced effusion by 40%. Addition of "
            "bevacizumab (anti-VEGF, 10 mg/kg q2w) led to complete resolution of "
            "chylothorax at 6 months. Bone stabilization was confirmed on 12-month MRI."
        ),
        "full_text": (
            "CASE PRESENTATION: A 16-year-old previously healthy male presented with "
            "3-month history of progressive dyspnea and pleuritic chest pain. Chest X-ray "
            "showed massive left pleural effusion. Thoracentesis yielded 4.2 liters of "
            "milky fluid with triglycerides 890 mg/dL (chylothorax confirmed). CT chest "
            "demonstrated lytic destruction of left ribs 4–7 with absence of cortical bone "
            "and soft tissue lymphatic channels tracking to the pleural space.\n\n"
            "DIAGNOSIS: Left thoracotomy + rib biopsy showed thin-walled endothelial-lined "
            "lymphatic channels replacing bone marrow, without malignant cells. IHC: D2-40+, "
            "LYVE-1+, CD34−. No PIK3CA or NRAS mutations on NGS. Diagnosis: Gorham-Stout "
            "disease with lymphaticopleural fistula.\n\n"
            "TREATMENT COURSE:\n"
            "Months 1–3: Sirolimus 2 mg/m²/day (trough 12 ng/mL) + MCT diet + octreotide "
            "infusion. Chylothorax drainage reduced from 800 mL/day to 450 mL/day (40% "
            "reduction). Thoracotomy for manual fistula ligation attempted but unsuccessful.\n\n"
            "Months 3–9: Bevacizumab 10 mg/kg IV every 2 weeks added. VEGF-A and VEGF-C "
            "dropped by 65% from baseline. Pleural drainage progressively decreased, "
            "achieving complete resolution at month 6. Patient transitioned from chest "
            "drain to monitoring at month 6. Repeat CT at 12 months showed no effusion "
            "and stable (non-progressive) bony lysis.\n\n"
            "OUTCOME: Patient remains on sirolimus maintenance at 18 months with no "
            "recurrence. This case suggests bevacizumab as a valuable adjunct to sirolimus "
            "for GSD-related refractory chylothorax."
        ),
        "keywords": ["GSD", "chylothorax", "bevacizumab", "sirolimus", "case report", "combination therapy", "bone lysis"],
    },

    # -----------------------------------------------------------------------
    # 12. GENETICS AND PRECISION MEDICINE
    # -----------------------------------------------------------------------
    {
        "id": "doc_012",
        "title": "Precision Medicine in CLA: Genotype-Guided Targeted Therapy",
        "source_type": "review_article",
        "disease_entity": "CLA",
        "year": 2024,
        "journal": "Nature Medicine",
        "authors": ["Blesinger H", "Kauffman T", "Sisson R"],
        "abstract": (
            "The identification of driver somatic and germline mutations in CLA subtypes "
            "has enabled genotype-guided targeted therapy. PIK3CA mutations in GLA and GSD "
            "are targetable with alpelisib (FDA-approved for PIK3CA-mutant breast cancer) "
            "and sirolimus. NRAS/KRAS mutations in KLA are targeted by MEK inhibitors "
            "(trametinib, cobimetinib). TSC1/TSC2 mutations in LAM are targeted by mTOR "
            "inhibitors. PTPN11 mutations in Noonan-CCLA may respond to MEK inhibition. "
            "Basket trials are underway evaluating alpelisib and trametinib across CLA "
            "subtypes stratified by molecular profile."
        ),
        "full_text": (
            "The molecular landscape of CLA spans two major oncogenic signaling pathways:\n\n"
            "1. PI3K/AKT/mTOR PATHWAY:\n"
            "• PIK3CA gain-of-function: GLA (62%), GSD (subset), isolated bone lesions\n"
            "• PIK3R1 loss-of-function: GLA (rare)\n"
            "• AKT1: rare\n"
            "• TSC1/TSC2 biallelic loss: LAM (100%)\n"
            "Targetable with: sirolimus, everolimus, alpelisib (PI3Kα inhibitor), "
            "copanlisib, capivasertib (AKT inhibitor)\n\n"
            "2. RAS/MAPK/ERK PATHWAY:\n"
            "• NRAS hotspot mutations (G12D, G12V, Q61R): KLA (40–50%)\n"
            "• KRAS hotspot mutations: KLA (20%)\n"
            "• RAF1: rare KLA\n"
            "• PTPN11/SOS1: Noonan syndrome-associated CCLA\n"
            "Targetable with: trametinib (MEK1/2 inhibitor), cobimetinib, binimetinib\n\n"
            "CLINICAL TRIALS:\n"
            "• ALPHA-CLA: Phase II alpelisib in PIK3CA-mutant GLA/GSD (NCT04729426)\n"
            "• TREMA-KLA: Phase II trametinib in NRAS/KRAS-mutant KLA (NCT05183230)\n"
            "• MTOR-RARE: Basket trial sirolimus + RAD001 in PI3K-pathway CLA (NCT04956874)\n\n"
            "LIQUID BIOPSY: Circulating tumor DNA (ctDNA) detection of PIK3CA and NRAS "
            "variants in plasma may enable non-invasive monitoring. Sensitivity is 45–60% "
            "in active disease, limiting its diagnostic utility but supporting treatment "
            "monitoring in mutation-positive patients."
        ),
        "keywords": ["precision medicine", "PIK3CA", "alpelisib", "NRAS", "trametinib", "mTOR", "targeted therapy", "basket trial"],
    },

    # -----------------------------------------------------------------------
    # 13. PATIENT PERSPECTIVES
    # -----------------------------------------------------------------------
    {
        "id": "doc_013",
        "title": "Quality of Life and Patient-Reported Outcomes in Adults Living with Complex Lymphatic Anomalies",
        "source_type": "patient_study",
        "disease_entity": "CLA",
        "year": 2023,
        "journal": "Orphanet Journal of Rare Diseases",
        "authors": ["Dori Y", "Itkin M", "Smith C"],
        "abstract": (
            "We surveyed 124 adult CLA patients using validated QoL instruments (SF-36, "
            "PROMIS). CLA patients had significantly lower physical (PCS 38.2 vs. 50 norm) "
            "and mental (MCS 42.1 vs. 50) component scores. Key barriers included diagnostic "
            "delay (median 5.1 years), limited specialist access, financial toxicity of "
            "sirolimus ($8,000–$15,000/month without insurance), and lack of disease-specific "
            "support resources. Over 80% reported fatigue as the most disabling symptom, "
            "followed by dyspnea (62%) and pain (54%). Advocacy organizations (Lymphatic "
            "Education & Research Network, NORD) were identified as key support resources."
        ),
        "full_text": (
            "Patient-reported outcomes are critical for understanding the true burden of "
            "CLA beyond clinical endpoints. We conducted a cross-sectional survey of 124 "
            "adults with confirmed CLA diagnoses (GLA n=45, LAM n=38, GSD n=22, KLA n=12, "
            "CCLA n=7).\n\n"
            "QoL FINDINGS:\n"
            "• SF-36 Physical Component Score (PCS): 38.2 ± 10.1 (vs. age-matched norm 50)\n"
            "• SF-36 Mental Component Score (MCS): 42.1 ± 11.2 (vs. norm 50)\n"
            "• LAM patients had lowest PCS (35.1), correlating with FEV₁\n"
            "• KLA patients had lowest MCS (38.4), reflecting disease uncertainty and "
            "  aggressive course\n\n"
            "BARRIERS TO CARE:\n"
            "• Diagnostic delay: median 5.1 years (range 0.5–23 years)\n"
            "• Access to CLA specialist: 68% traveled >100 miles for expert care\n"
            "• Insurance coverage: 34% reported coverage denial for sirolimus at least once\n"
            "• Financial toxicity: sirolimus list price $8,000–15,000/month; many rely on "
            "  manufacturer patient assistance programs\n\n"
            "SYMPTOMS: Fatigue (82%), dyspnea (62%), pain (54%), edema (48%), anxiety/depression "
            "(45%). Fatigue did not correlate with disease subtype, suggesting a common "
            "pathobiological mechanism across CLAs.\n\n"
            "SUPPORT RESOURCES: Lymphatic Education & Research Network (LE&RN) was identified "
            "by 71% as the most helpful patient organization. NORD rare disease database "
            "provided educational resources for 58%."
        ),
        "keywords": ["quality of life", "patient reported outcomes", "CLA", "diagnostic delay", "sirolimus cost", "fatigue", "LE&RN", "NORD"],
    },

    # -----------------------------------------------------------------------
    # 14. NOONAN SYNDROME AND CCLA
    # -----------------------------------------------------------------------
    {
        "id": "doc_014",
        "title": "Noonan Syndrome-Associated Central Conducting Lymphatic Anomaly: Outcomes of Thoracic Duct Intervention",
        "source_type": "original_research",
        "disease_entity": "CCLA",
        "year": 2022,
        "journal": "Annals of Thoracic Surgery",
        "authors": ["Dori Y", "Smith C", "Itkin M"],
        "abstract": (
            "Noonan syndrome (NS) caused by PTPN11, SOS1, and RAS-pathway mutations "
            "is frequently complicated by central conducting lymphatic anomaly with pulmonary "
            "lymphatic perfusion syndrome (PLPS) manifesting as neonatal chylothorax, "
            "chylous ascites, and lymphedema. We describe thoracic duct intervention "
            "outcomes in 34 NS-CCLA patients. Thoracic duct embolization (TDE) resolved "
            "chylothorax in 71% of patients. Lymphatic reconstructive surgery (LRS) "
            "improved PLPS symptoms in 68%. Trametinib for RAS-pathway mutations improved "
            "lymphatic flow on DCMRL in 5/7 patients."
        ),
        "full_text": (
            "Noonan syndrome (NS) is caused by germline mutations in RAS-MAPK pathway genes "
            "(PTPN11 50%, SOS1 10%, RAF1 5%, KRAS 2%, RIT1, BRAF, MAP2K1/2). NS affects "
            "1:1,000–2,500 live births and is the most common syndromic cause of CCLA.\n\n"
            "LYMPHATIC PHENOTYPE: NS-CCLA presents as neonatal hydrops fetalis, chylothorax, "
            "chylous ascites, and/or lymphedema. Underlying mechanism: PTPN11 gain-of-function "
            "mutations activate SHP2 phosphatase, hyperactivating RAS-MEK-ERK signaling in "
            "lymphatic endothelial cells, impairing PROX1-mediated lymphatic identity and "
            "valve formation. DCMRL demonstrates retrograde pulmonary lymphatic perfusion "
            "(lung lights up on lymphangiogram), absent or hypoplastic thoracic duct, "
            "and dermal backflow.\n\n"
            "INTERVENTIONAL OUTCOMES:\n"
            "TDE performed in 28/34 patients (n=6 unsuitable due to anatomy):\n"
            "• Complete chylothorax resolution: 20/28 (71%)\n"
            "• Partial response: 5/28 (18%)\n"
            "• No response: 3/28 (11%)\n"
            "Lymphatic reconstructive surgery (thoracic duct reconstruction/anastomosis): "
            "• Performed in 19/34; symptomatic improvement in 13/19 (68%)\n\n"
            "TRAMETINIB THERAPY: 7 patients with confirmed RAS-pathway mutations received "
            "trametinib 0.015–0.025 mg/kg/day. DCMRL at 6 months showed improved antegrade "
            "lymphatic flow and reduced PLPS in 5/7. This suggests MEK inhibition may "
            "partially restore lymphatic valve function in NS."
        ),
        "keywords": ["Noonan syndrome", "CCLA", "PTPN11", "thoracic duct embolization", "PLPS", "trametinib", "lymphatic reconstruction"],
    },

    # -----------------------------------------------------------------------
    # 15. SAMPLE PATIENT-FACING DOCUMENT
    # -----------------------------------------------------------------------
    {
        "id": "doc_015",
        "title": "Understanding Complex Lymphatic Anomalies: A Guide for Patients and Families",
        "source_type": "patient_education",
        "disease_entity": "CLA",
        "year": 2024,
        "journal": "NORD Rare Disease Database",
        "authors": ["National Organization for Rare Disorders"],
        "abstract": (
            "Complex lymphatic anomalies are a group of rare conditions where the lymphatic "
            "system – the network of vessels that carries lymph fluid through the body – "
            "does not develop or work normally. This can cause a wide range of problems "
            "including fluid buildup, bone changes, and breathing difficulties. There is "
            "no cure, but treatments including sirolimus (Rapamune) can help manage symptoms "
            "and prevent complications. Patients should be followed by specialists experienced "
            "in lymphatic diseases."
        ),
        "full_text": (
            "WHAT IS THE LYMPHATIC SYSTEM?\n"
            "The lymphatic system is a network of vessels, nodes (glands), and organs that "
            "helps maintain fluid balance, supports immune function, and absorbs fats from "
            "the digestive system. In complex lymphatic anomalies (CLAs), some of these "
            "lymphatic vessels grow abnormally or in the wrong places.\n\n"
            "TYPES OF CLA:\n"
            "1. Gorham-Stout Disease (GSD): Sometimes called 'vanishing bone disease' "
            "because abnormal lymphatic channels grow inside bones, causing them to slowly "
            "dissolve. This can weaken bones and cause pain, fractures, and fluid around "
            "the lungs (chylothorax).\n\n"
            "2. Generalized Lymphatic Anomaly (GLA): Lymphatic cysts or malformations "
            "appear in multiple areas of the body including bones, spleen, liver, and "
            "lungs. Unlike GSD, bones do not progressively dissolve.\n\n"
            "3. Central Conducting Lymphatic Anomaly (CCLA): The main lymphatic 'pipe' "
            "(thoracic duct) is absent, narrowed, or reversed, causing lymph fluid to leak "
            "into the lungs, chest, or abdomen.\n\n"
            "4. Kaposiform Lymphangiomatosis (KLA): An aggressive form affecting multiple "
            "organs, often causing low platelet counts (thrombocytopenia) and bleeding "
            "into the chest.\n\n"
            "5. Lymphangioleiomyomatosis (LAM): Mainly affects women; muscle-like cells "
            "grow in the lungs, causing progressive breathing difficulty and cysts in the lungs.\n\n"
            "SYMPTOMS: Common symptoms include: breathing problems, fluid in the chest or "
            "abdomen, bone pain, swelling, fatigue, and recurrent infections. Symptoms "
            "vary greatly depending on which organs are affected.\n\n"
            "TREATMENT: The main medication is sirolimus (brand name: Rapamune), which slows "
            "abnormal lymphatic growth. You take it by mouth daily and need regular blood "
            "tests to monitor drug levels. Side effects include mouth sores, infections, "
            "and high cholesterol. Other treatments include a special low-fat diet with MCT "
            "oil supplements if you have fluid leaking into your chest, and procedures to "
            "seal leaking lymphatic vessels.\n\n"
            "LIVING WITH CLA: Connect with patient organizations like the Lymphatic Education "
            "& Research Network (LE&RN) at lymphaticnetwork.org and the National Organization "
            "for Rare Disorders (NORD) at rarediseases.org. Rare disease centers such as "
            "Children's Hospital of Philadelphia, Boston Children's, and NIH Clinical Center "
            "have specialized CLA programs."
        ),
        "keywords": ["patient education", "CLA overview", "sirolimus", "GSD", "GLA", "KLA", "LAM", "CCLA", "LE&RN", "NORD"],
    },
]

# ---------------------------------------------------------------------------
# Evaluation Q&A pairs
# ---------------------------------------------------------------------------

EVAL_QA: list[dict] = [
    {
        "id": "qa_001",
        "question": "What is Gorham-Stout disease and what causes it?",
        "reference_answer": (
            "Gorham-Stout disease (GSD) is an ultra-rare disorder characterized by "
            "progressive osteolysis (bone dissolution) caused by the proliferation of "
            "intraosseous lymphatic capillaries that invade and replace bone. These "
            "abnormal lymphatics secrete cytokines like IL-6 and VEGF-C that activate "
            "osteoclasts and destroy cortical and trabecular bone. A subset of cases "
            "harbor somatic PIK3CA mutations in the PI3K/AKT/mTOR signaling pathway."
        ),
        "category": "pathogenesis",
        "difficulty": "medium",
        "relevant_doc_ids": ["doc_002"],
    },
    {
        "id": "qa_002",
        "question": "What is the recommended dose of sirolimus for children with CLA?",
        "reference_answer": (
            "For children with CLA, sirolimus is typically started at 0.8 mg/m²/day "
            "for those weighing ≤40 kg, or 1 mg/m²/day for those >40 kg. The target "
            "trough blood level is 5–15 ng/mL. Levels are checked every 2 weeks for "
            "the first 6 weeks, then monthly for 3 months, then every 3 months when "
            "stable. The maximum dose is 3 mg/m²/day. Duration is typically indefinite "
            "for systemic disease."
        ),
        "category": "treatment",
        "difficulty": "medium",
        "relevant_doc_ids": ["doc_009", "doc_007"],
    },
    {
        "id": "qa_003",
        "question": "How is chylothorax diagnosed and managed in CLA patients?",
        "reference_answer": (
            "Chylothorax is diagnosed by pleural fluid analysis showing triglycerides "
            ">110 mg/dL, lymphocyte predominance >80%, and positive Sudan III stain for "
            "chylomicrons. Management includes dietary modification (MCT diet or NPO with "
            "parenteral nutrition), octreotide (1–10 μg/kg/hour IV), sirolimus to reduce "
            "lymphangiogenesis, and thoracic duct embolization (TDE) with 72% success rate. "
            "Pleurodesis is not recommended for CLA-related chylothorax as it does not "
            "address the underlying lymphatic malformation."
        ),
        "category": "diagnosis_treatment",
        "difficulty": "hard",
        "relevant_doc_ids": ["doc_010", "doc_005"],
    },
    {
        "id": "qa_004",
        "question": "What genetic mutations are found in kaposiform lymphangiomatosis (KLA)?",
        "reference_answer": (
            "KLA is primarily driven by somatic mutations in NRAS (codons 12, 13, 61) "
            "and KRAS, found in 60–70% of cases. These are activating mutations in the "
            "RAS/MAPK/ERK signaling pathway that promote aberrant lymphatic endothelial "
            "proliferation and impair lymphatic valve formation. RAF1 mutations are rare. "
            "MEK inhibitors like trametinib target this pathway and show 67% objective "
            "response rate in NRAS/KRAS-mutant KLA."
        ),
        "category": "genetics",
        "difficulty": "medium",
        "relevant_doc_ids": ["doc_004", "doc_012"],
    },
    {
        "id": "qa_005",
        "question": "What serum biomarker can distinguish LAM from other CLA subtypes?",
        "reference_answer": (
            "Serum VEGF-D (vascular endothelial growth factor D) is the key biomarker "
            "for distinguishing LAM from other CLA subtypes. A serum VEGF-D level "
            ">800 pg/mL has 73% sensitivity and 100% specificity for LAM diagnosis. "
            "VEGF-D is produced by LAM cells and stimulates lymphangiogenesis via VEGFR-3. "
            "VEGF-D also decreases by ~42% on sirolimus therapy, making it a useful "
            "pharmacodynamic biomarker."
        ),
        "category": "biomarkers",
        "difficulty": "medium",
        "relevant_doc_ids": ["doc_008", "doc_006"],
    },
    {
        "id": "qa_006",
        "question": "What is pulmonary lymphatic perfusion syndrome and how is it treated?",
        "reference_answer": (
            "Pulmonary lymphatic perfusion syndrome (PLPS) is a manifestation of central "
            "conducting lymphatic anomaly (CCLA) where lymph flows abnormally in a retrograde "
            "direction from the thoracic duct into pulmonary lymphatics, causing diffuse "
            "pulmonary interstitial edema and respiratory distress. It is common in Noonan "
            "syndrome (PTPN11, SOS1 mutations). Treatment includes thoracic duct "
            "decompression via thoracic duct embolization or lymphovenous anastomosis. "
            "Trametinib (MEK inhibitor) has shown improved lymphatic flow on DCMRL in "
            "5 of 7 NS-CCLA patients with RAS-pathway mutations."
        ),
        "category": "diagnosis_treatment",
        "difficulty": "hard",
        "relevant_doc_ids": ["doc_005", "doc_014"],
    },
    {
        "id": "qa_007",
        "question": "What is the overall response rate of sirolimus across all CLA subtypes?",
        "reference_answer": (
            "Based on a systematic review of 47 studies comprising 389 patients, the "
            "overall objective response rate (partial + complete response) of sirolimus "
            "across all CLA subtypes is 76% (95% CI 71–81%). By subtype: LAM has the "
            "highest response rate at 85%, followed by GLA at 79%, GSD at 71%, KLA at "
            "68%, and CCLA at 64%. Median time to response is 3.2 months. Complete "
            "response is rare (8%). Most patients relapse within 12 months of "
            "discontinuation, supporting indefinite treatment."
        ),
        "category": "treatment_outcomes",
        "difficulty": "easy",
        "relevant_doc_ids": ["doc_007"],
    },
    {
        "id": "qa_008",
        "question": "What are the main challenges CLA patients face in getting care?",
        "reference_answer": (
            "CLA patients face several major barriers: (1) Diagnostic delay averaging "
            "5.1 years due to phenotypic overlap with other diseases; (2) Limited access "
            "to CLA specialists, with 68% traveling >100 miles for expert care; "
            "(3) Financial toxicity – sirolimus costs $8,000–$15,000/month without "
            "insurance; (4) Insurance denials for sirolimus (34% of patients); "
            "(5) Fatigue, dyspnea, and pain significantly impair quality of life. "
            "Key support organizations include LE&RN (lymphaticnetwork.org) and NORD."
        ),
        "category": "patient_experience",
        "difficulty": "easy",
        "relevant_doc_ids": ["doc_013"],
    },
    {
        "id": "qa_009",
        "question": "How is CLA classified according to the ISSVA 2018 classification?",
        "reference_answer": (
            "The ISSVA 2018 classification groups complex lymphatic anomalies into five "
            "major subtypes: (1) Gorham-Stout disease (GSD) – progressive osteolysis from "
            "lymphatic invasion of bone; (2) Generalized Lymphatic Anomaly (GLA) – "
            "multifocal lymphatic malformations in bone and soft tissue without progressive "
            "osteolysis; (3) Central Conducting Lymphatic Anomaly (CCLA) – dysfunction of "
            "thoracic duct and cisterna chyli; (4) Kaposiform Lymphangiomatosis (KLA) – "
            "aggressive multisystem disorder with spindle-cell endothelium; and "
            "(5) Lymphangioleiomyomatosis (LAM) – TSC2-driven cystic lung disease in women."
        ),
        "category": "classification",
        "difficulty": "easy",
        "relevant_doc_ids": ["doc_001"],
    },
    {
        "id": "qa_010",
        "question": "What is the MILES trial and what did it show about sirolimus in LAM?",
        "reference_answer": (
            "The MILES (Multicenter International LAM Efficacy of Sirolimus) trial is the "
            "pivotal Phase III randomized controlled trial that established sirolimus as "
            "the standard of care for LAM. It demonstrated that sirolimus stabilized lung "
            "function with FEV₁ decline of only −12 mL/year compared to −134 mL/year in "
            "the placebo group. Based on these results, sirolimus received FDA approval "
            "for LAM in 2015. Sirolimus does not reverse existing lung function decline "
            "but prevents further deterioration."
        ),
        "category": "clinical_trial",
        "difficulty": "medium",
        "relevant_doc_ids": ["doc_006"],
    },
]


def generate_additional_documents(n: int = 5) -> list[dict]:
    """Generate additional randomized pseudo documents to expand the corpus."""
    topics = [
        ("Sclerotherapy for Macrocystic Lymphatic Malformations in CLA", "GLA",
         "Doxycycline and bleomycin sclerotherapy achieves complete or near-complete response "
         "in 65–80% of macrocystic lymphatic malformations. Intralesional injection under "
         "ultrasound guidance is the preferred approach. Sclerotherapy is less effective for "
         "microcystic lesions (<2 cm cysts). Multiple sessions (median 2.3) are typically "
         "required. Sirolimus is used as adjunct for refractory or diffuse lesions."),
        ("Radiology of CLA: From Plain Films to Dynamic MR Lymphangiography", "CLA",
         "Imaging modalities for CLA include plain radiographs (bone lysis in GSD/GLA), "
         "CT (mediastinal/visceral involvement), MRI (soft tissue characterization), "
         "and dynamic contrast-enhanced MR lymphangiography (DCMRL) for mapping central "
         "lymphatic anatomy. DCMRL following intranodal gadolinium injection is the "
         "gold standard for central conducting lymphatic anomalies. T2-weighted MRI "
         "shows hyperintense fluid-filled lymphatic cysts. DWI/ADC mapping assesses "
         "lesion cellularity and early treatment response."),
        ("Alpelisib for PIK3CA-Mutant GLA: Preliminary Phase II Data", "GLA",
         "Alpelisib is an oral selective PI3Kα inhibitor FDA-approved for PIK3CA-mutant "
         "breast cancer. In a phase II basket trial of 15 GLA patients with PIK3CA "
         "mutations (p.H1047R or p.E545K), alpelisib (150–300 mg/day) achieved 73% "
         "objective response rate at 6 months, including 2 complete responses. "
         "Main toxicities: hyperglycemia (73%, managed with metformin), diarrhea (40%), "
         "rash (33%). This represents a promising second-line option for PIK3CA-mutant GLA "
         "refractory to sirolimus."),
        ("Lymphatic Diet and Nutritional Support in CLA-Related Chylous Leaks", "CLA",
         "Nutritional management of chylous leaks in CLA centers on reducing long-chain "
         "triglyceride (LCT) intake, which is absorbed via intestinal lymphatics as chylomicrons. "
         "Medium-chain triglycerides (MCTs) bypass the lymphatic system and are absorbed "
         "directly via the portal vein. MCT supplementation (50–70% of fat calories as MCT) "
         "reduces chylous output by 40–60% within 2–4 weeks. Pediatric MCT-based formulas "
         "include Portagen and Monogen. Total parenteral nutrition is reserved for severe "
         "or refractory cases. Fat-soluble vitamins (A, D, E, K) require supplementation "
         "during MCT diet as they have reduced absorption."),
        ("CLA Registries and Natural History Studies: Current Landscape", "CLA",
         "International CLA registries are essential for studying natural history in these "
         "ultra-rare disorders. Current active registries include: (1) NTLR (National Thoracic "
         "Lymphatic Anomalies Registry) at Children's Hospital of Philadelphia; (2) EuroCLA "
         "consortium registry across 12 European centers; (3) LAM Foundation registry "
         "(>2000 LAM patients enrolled). Natural history data show 10-year mortality of "
         "18% for KLA, 8% for GSD with chylothorax, and 4% for GLA without chylothorax. "
         "Annual lung function decline in LAM without treatment is 70–120 mL/year FEV₁."),
    ]

    random.seed(42)
    docs = []
    for i, (title, entity, text) in enumerate(topics[:n]):
        docs.append({
            "id": f"doc_extra_{i+1:03d}",
            "title": title,
            "source_type": random.choice(["review_article", "original_research", "case_series"]),
            "disease_entity": entity,
            "year": random.randint(2020, 2024),
            "journal": random.choice([
                "Orphanet Journal of Rare Diseases",
                "Journal of Clinical Investigation",
                "Blood",
                "Lymphatic Research and Biology",
                "American Journal of Respiratory and Critical Care Medicine",
            ]),
            "authors": [f"Author {chr(65+j)}" for j in range(random.randint(2, 4))],
            "abstract": text[:300] + "..." if len(text) > 300 else text,
            "full_text": text,
            "keywords": [entity.lower(), "CLA", "lymphatic", "rare disease"],
        })
    return docs


def main():
    parser = argparse.ArgumentParser(description="Generate CLA pseudo dataset")
    parser.add_argument(
        "--output",
        default=str(Path(__file__).parent / "cla_documents.json"),
        help="Output path for document JSON",
    )
    parser.add_argument(
        "--eval_output",
        default=str(Path(__file__).parent / "eval_questions.json"),
        help="Output path for evaluation Q&A JSON",
    )
    parser.add_argument(
        "--n_extra",
        type=int,
        default=5,
        help="Number of extra auto-generated documents",
    )
    args = parser.parse_args()

    all_docs = DOCUMENTS + generate_additional_documents(args.n_extra)

    print(f"Generating dataset with {len(all_docs)} documents...")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"documents": all_docs, "metadata": {
            "created": datetime.now().isoformat(),
            "total_documents": len(all_docs),
            "disease_focus": "Complex Lymphatic Anomalies (CLA)",
            "subtypes": ["GSD", "GLA", "KLA", "CCLA", "LAM"],
            "note": "Pseudo-curated dataset for RAG pipeline benchmarking. Not for clinical use.",
        }}, f, indent=2, ensure_ascii=False)
    print(f"Documents saved to: {args.output}")

    print(f"Generating {len(EVAL_QA)} evaluation Q&A pairs...")
    with open(args.eval_output, "w", encoding="utf-8") as f:
        json.dump({"questions": EVAL_QA, "metadata": {
            "created": datetime.now().isoformat(),
            "total_questions": len(EVAL_QA),
            "categories": list({q["category"] for q in EVAL_QA}),
        }}, f, indent=2, ensure_ascii=False)
    print(f"Eval Q&A saved to: {args.eval_output}")
    print("Dataset generation complete.")


if __name__ == "__main__":
    main()
