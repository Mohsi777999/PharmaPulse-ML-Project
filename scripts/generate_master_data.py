"""
PharmaPulse - Master Data Generator
Generates categories, suppliers, medicines, and medicine_suppliers CSV files.
Adheres strictly to project constraints:
- 6 categories with exact distribution
- 150 medicines with exact demand pattern distribution
- 10 suppliers with realistic lead times and reliabilities
- 1 preferred supplier per medicine plus backup suppliers
- Random seed: 42
"""

import os
import random
import numpy as np
import pandas as pd

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
RAW_DATA_DIR = os.path.join(PROJECT_DIR, "data", "raw")
os.makedirs(RAW_DATA_DIR, exist_ok=True)

# 1. Categories
CATEGORIES_DATA = [
    {
        "category_id": "CAT001",
        "category_name": "Pain relief and fever",
        "description": "Analgesics, antipyretics, and non-steroidal anti-inflammatory drugs (NSAIDs)",
        "target_count": 30
    },
    {
        "category_id": "CAT002",
        "category_name": "Anti-infectives",
        "description": "Antibiotics, antifungals, antivirals, and antimicrobial agents",
        "target_count": 25
    },
    {
        "category_id": "CAT003",
        "category_name": "Respiratory and cold",
        "description": "Antihistamines, decongestants, cough syrups, bronchodilators, and anti-asthmatics",
        "target_count": 25
    },
    {
        "category_id": "CAT004",
        "category_name": "Gastrointestinal",
        "description": "Proton pump inhibitors, antacids, anti-diarrheals, laxatives, and anti-emetics",
        "target_count": 20
    },
    {
        "category_id": "CAT005",
        "category_name": "Chronic care",
        "description": "Cardiovascular, antihypertensive, anti-diabetic, and lipid-lowering therapies",
        "target_count": 25
    },
    {
        "category_id": "CAT006",
        "category_name": "Vitamins and supplements",
        "description": "Multivitamins, minerals, calcium supplements, and nutritional therapies",
        "target_count": 25
    }
]

# 2. Suppliers
SUPPLIERS_DATA = [
    {"supplier_id": "SUP001", "supplier_name": "Premier Pharma Distribution", "lead_time_days": 4, "delivery_reliability": 0.96, "supplier_rating": 4.8, "location": "Karachi", "active_status": "Active"},
    {"supplier_id": "SUP002", "supplier_name": "Indus Healthcare Logistics", "lead_time_days": 7, "delivery_reliability": 0.94, "supplier_rating": 4.6, "location": "Lahore", "active_status": "Active"},
    {"supplier_id": "SUP003", "supplier_name": "Capital MedSupply Corp", "lead_time_days": 5, "delivery_reliability": 0.98, "supplier_rating": 4.9, "location": "Islamabad", "active_status": "Active"},
    {"supplier_id": "SUP004", "supplier_name": "Potohar Pharma Traders", "lead_time_days": 8, "delivery_reliability": 0.91, "supplier_rating": 4.2, "location": "Rawalpindi", "active_status": "Active"},
    {"supplier_id": "SUP005", "supplier_name": "Chenab Medical Wholesalers", "lead_time_days": 10, "delivery_reliability": 0.89, "supplier_rating": 4.1, "location": "Faisalabad", "active_status": "Active"},
    {"supplier_id": "SUP006", "supplier_name": "Khyber Drug Distributors", "lead_time_days": 14, "delivery_reliability": 0.88, "supplier_rating": 4.0, "location": "Peshawar", "active_status": "Active"},
    {"supplier_id": "SUP007", "supplier_name": "Oasis Pharma Network", "lead_time_days": 12, "delivery_reliability": 0.92, "supplier_rating": 4.3, "location": "Multan", "active_status": "Active"},
    {"supplier_id": "SUP008", "supplier_name": "Mehran Health Supplies", "lead_time_days": 6, "delivery_reliability": 0.95, "supplier_rating": 4.7, "location": "Hyderabad", "active_status": "Active"},
    {"supplier_id": "SUP009", "supplier_name": "Chiltan Med Logistics", "lead_time_days": 18, "delivery_reliability": 0.86, "supplier_rating": 3.9, "location": "Quetta", "active_status": "Active"},
    {"supplier_id": "SUP010", "supplier_name": "Punjab National Pharma", "lead_time_days": 9, "delivery_reliability": 0.93, "supplier_rating": 4.4, "location": "Gujranwala", "active_status": "Active"}
]

# Medicine templates by category to generate realistic Pakistani pharmaceutical inventory
MEDICINE_TEMPLATES = {
    "CAT001": [
        ("Panadol", "Paracetamol", "Tablet", "500mg", "tablet", 4.5, 7.0),
        ("Panadol Extra", "Paracetamol + Caffeine", "Tablet", "500mg/65mg", "tablet", 6.0, 9.5),
        ("Panadol CF", "Paracetamol + Pseudoephedrine", "Tablet", "500mg/30mg", "tablet", 8.0, 12.0),
        ("Brufen", "Ibuprofen", "Tablet", "400mg", "tablet", 7.0, 11.0),
        ("Brufen DS", "Ibuprofen", "Suspension", "200mg/5ml", "bottle_120ml", 85.0, 125.0),
        ("Disprin", "Aspirin", "Tablet", "300mg", "tablet", 3.0, 5.0),
        ("Disprin CV", "Aspirin", "Tablet", "75mg", "tablet", 3.5, 5.5),
        ("Ponstan", "Mefenamic Acid", "Tablet", "500mg", "tablet", 9.0, 14.0),
        ("Ponstan Forte", "Mefenamic Acid", "Tablet", "500mg", "tablet", 12.0, 18.0),
        ("Voltral", "Diclofenac Sodium", "Tablet", "50mg", "tablet", 11.0, 16.5),
        ("Voltral SR", "Diclofenac Sodium", "Capsule", "75mg", "capsule", 16.0, 24.0),
        ("Voltral Emulgel", "Diclofenac Diethylamine", "Gel", "1%", "tube_50g", 140.0, 200.0),
        ("Cataflam", "Diclofenac Potassium", "Tablet", "50mg", "tablet", 15.0, 22.0),
        ("Nuberol Forte", "Paracetamol + Orphenadrine", "Tablet", "650mg/50mg", "tablet", 14.0, 21.0),
        ("Muscoril", "Thiocolchicoside", "Capsule", "4mg", "capsule", 35.0, 52.0),
        ("Ansaid", "Flurbiprofen", "Tablet", "100mg", "tablet", 22.0, 32.0),
        ("Brexin", "Piroxicam-beta-cyclodextrin", "Tablet", "20mg", "tablet", 30.0, 44.0),
        ("Feldene", "Piroxicam", "Capsule", "20mg", "capsule", 18.0, 27.0),
        ("Celebrex", "Celecoxib", "Capsule", "200mg", "capsule", 65.0, 95.0),
        ("Arcoxia", "Etoricoxib", "Tablet", "90mg", "tablet", 80.0, 115.0),
        ("Arcoxia 60", "Etoricoxib", "Tablet", "60mg", "tablet", 60.0, 88.0),
        ("Naprosyn", "Naproxen", "Tablet", "500mg", "tablet", 16.0, 24.0),
        ("Synflex", "Naproxen Sodium", "Tablet", "550mg", "tablet", 20.0, 30.0),
        ("Tramal", "Tramadol Hydrochloride", "Capsule", "50mg", "capsule", 25.0, 38.0),
        ("Calpol", "Paracetamol", "Suspension", "120mg/5ml", "bottle_60ml", 60.0, 90.0),
        ("Calpol 6 Plus", "Paracetamol", "Suspension", "250mg/5ml", "bottle_60ml", 75.0, 110.0),
        ("Arinac", "Ibuprofen + Pseudoephedrine", "Tablet", "200mg/30mg", "tablet", 10.0, 15.0),
        ("Arinac Forte", "Ibuprofen + Pseudoephedrine", "Tablet", "400mg/60mg", "tablet", 18.0, 26.0),
        ("Xefo", "Lornoxicam", "Tablet", "8mg", "tablet", 38.0, 55.0),
        ("Mobic", "Meloxicam", "Tablet", "15mg", "tablet", 28.0, 40.0),
    ],
    "CAT002": [
        ("Augmentin 625", "Amoxicillin + Clavulanic Acid", "Tablet", "625mg", "tablet", 38.0, 55.0),
        ("Augmentin 1g", "Amoxicillin + Clavulanic Acid", "Tablet", "1000mg", "tablet", 62.0, 90.0),
        ("Augmentin DS Susp", "Amoxicillin + Clavulanic Acid", "Suspension", "312.5mg/5ml", "bottle_100ml", 180.0, 260.0),
        ("Amoxil", "Amoxicillin", "Capsule", "500mg", "capsule", 14.0, 21.0),
        ("Amoxil Susp", "Amoxicillin", "Suspension", "250mg/5ml", "bottle_60ml", 70.0, 105.0),
        ("Cravit", "Levofloxacin", "Tablet", "500mg", "tablet", 55.0, 80.0),
        ("Cravit 250", "Levofloxacin", "Tablet", "250mg", "tablet", 32.0, 48.0),
        ("Ciproxin", "Ciprofloxacin", "Tablet", "500mg", "tablet", 35.0, 52.0),
        ("Ciproxin 250", "Ciprofloxacin", "Tablet", "250mg", "tablet", 20.0, 30.0),
        ("Zithromax", "Azithromycin", "Capsule", "500mg", "capsule", 75.0, 110.0),
        ("Zithromax 250", "Azithromycin", "Capsule", "250mg", "capsule", 45.0, 68.0),
        ("Zithromax Susp", "Azithromycin", "Suspension", "200mg/5ml", "bottle_30ml", 190.0, 275.0),
        ("Klaricid", "Clarithromycin", "Tablet", "500mg", "tablet", 85.0, 125.0),
        ("Klaricid Susp", "Clarithromycin", "Suspension", "125mg/5ml", "bottle_60ml", 240.0, 340.0),
        ("Cefspan", "Cefixime", "Capsule", "400mg", "capsule", 95.0, 140.0),
        ("Cefspan Susp", "Cefixime", "Suspension", "100mg/5ml", "bottle_30ml", 220.0, 315.0),
        ("Velosef", "Cephradine", "Capsule", "500mg", "capsule", 24.0, 36.0),
        ("Velosef Susp", "Cephradine", "Suspension", "250mg/5ml", "bottle_60ml", 110.0, 160.0),
        ("Flagyl", "Metronidazole", "Tablet", "400mg", "tablet", 6.0, 9.5),
        ("Flagyl Susp", "Metronidazole", "Suspension", "200mg/5ml", "bottle_60ml", 65.0, 95.0),
        ("Septran DS", "Co-trimoxazole", "Tablet", "800mg/160mg", "tablet", 12.0, 18.0),
        ("Diflucan", "Fluconazole", "Capsule", "150mg", "capsule", 120.0, 175.0),
        ("Daktarin Oral Gel", "Miconazole", "Gel", "20mg/g", "tube_40g", 160.0, 230.0),
        ("Canesten Cream", "Clotrimazole", "Cream", "1%", "tube_20g", 110.0, 160.0),
        ("Zovirax", "Acyclovir", "Tablet", "400mg", "tablet", 45.0, 65.0),
    ],
    "CAT003": [
        ("Zyrtec", "Cetirizine", "Tablet", "10mg", "tablet", 9.0, 14.0),
        ("Zyrtec Oral Sol", "Cetirizine", "Syrup", "5mg/5ml", "bottle_60ml", 85.0, 125.0),
        ("Clarityne", "Loratadine", "Tablet", "10mg", "tablet", 12.0, 18.0),
        ("Telfast 120", "Fexofenadine", "Tablet", "120mg", "tablet", 28.0, 42.0),
        ("Telfast 180", "Fexofenadine", "Tablet", "180mg", "tablet", 36.0, 52.0),
        ("Kestine", "Ebastine", "Tablet", "20mg", "tablet", 32.0, 48.0),
        ("Kestine 10", "Ebastine", "Tablet", "10mg", "tablet", 20.0, 30.0),
        ("Avamys", "Fluticasone Furoate", "Nasal Spray", "27.5mcg", "bottle_120d", 420.0, 580.0),
        ("Nasorex", "Mometasone Furoate", "Nasal Spray", "50mcg", "bottle_120d", 350.0, 490.0),
        ("Ventolin Inhaler", "Salbutamol", "Inhaler", "100mcg/puff", "inhaler", 210.0, 295.0),
        ("Ventolin Syrup", "Salbutamol", "Syrup", "2mg/5ml", "bottle_120ml", 70.0, 105.0),
        ("Seretide Evohaler", "Salmeterol + Fluticasone", "Inhaler", "25/125mcg", "inhaler", 750.0, 1050.0),
        ("Symbicort Turbuhaler", "Budesonide + Formoterol", "Inhaler", "160/4.5mcg", "inhaler", 880.0, 1220.0),
        ("Singulair 10", "Montelukast", "Tablet", "10mg", "tablet", 48.0, 70.0),
        ("Singulair 5", "Montelukast", "Chewable", "5mg", "tablet", 38.0, 55.0),
        ("Singulair 4 Sachet", "Montelukast", "Granules", "4mg", "sachet", 32.0, 48.0),
        ("Hydryllin", "Aminophylline + Diphenhydramine", "Syrup", "120ml", "bottle_120ml", 75.0, 110.0),
        ("Cofcol", "Dextromethorphan + Pseudoephedrine", "Syrup", "120ml", "bottle_120ml", 80.0, 118.0),
        ("Pulmonol", "Chlorpheniramine + Ephedrine", "Syrup", "120ml", "bottle_120ml", 70.0, 105.0),
        ("Acefyl Cough", "Acefylline Piperazine", "Syrup", "120ml", "bottle_120ml", 85.0, 125.0),
        ("Sancos", "Pholcodine", "Syrup", "120ml", "bottle_120ml", 95.0, 140.0),
        ("Xynosine", "Xylometazoline", "Nasal Drops", "0.1%", "bottle_15ml", 65.0, 95.0),
        ("Otrivin", "Xylometazoline", "Nasal Spray", "0.1%", "bottle_10ml", 90.0, 130.0),
        ("Actifed", "Triprolidine + Pseudoephedrine", "Tablet", "2.5mg/60mg", "tablet", 8.0, 12.0),
        ("Actifed DM", "Triprolidine + Dextromethorphan", "Syrup", "120ml", "bottle_120ml", 85.0, 125.0),
    ],
    "CAT004": [
        ("Risek 20", "Omeprazole", "Capsule", "20mg", "capsule", 24.0, 36.0),
        ("Risek 40", "Omeprazole", "Capsule", "40mg", "capsule", 40.0, 58.0),
        ("Risek Insta 20", "Omeprazole + Sodium Bicarb", "Sachet", "20mg", "sachet", 35.0, 50.0),
        ("Nexum 20", "Esomeprazole", "Tablet", "20mg", "tablet", 32.0, 48.0),
        ("Nexum 40", "Esomeprazole", "Tablet", "40mg", "tablet", 52.0, 75.0),
        ("Controloc 20", "Pantoprazole", "Tablet", "20mg", "tablet", 28.0, 42.0),
        ("Controloc 40", "Pantoprazole", "Tablet", "40mg", "tablet", 46.0, 68.0),
        ("Gaviscon Liquid", "Sodium Alginate + Potassium Bicarb", "Suspension", "120ml", "bottle_120ml", 140.0, 205.0),
        ("Gaviscon Advance", "Sodium Alginate", "Suspension", "150ml", "bottle_150ml", 210.0, 305.0),
        ("Mucaine", "Oxetacaine + Al Hydroxide", "Suspension", "120ml", "bottle_120ml", 95.0, 140.0),
        ("Motilium", "Domperidone", "Tablet", "10mg", "tablet", 9.0, 14.0),
        ("Motilium Susp", "Domperidone", "Suspension", "5mg/5ml", "bottle_120ml", 85.0, 125.0),
        ("Maxolon", "Metoclopramide", "Tablet", "10mg", "tablet", 5.0, 8.0),
        ("Imodium", "Loperamide", "Capsule", "2mg", "capsule", 7.5, 11.5),
        ("Entamizole", "Diloxanide Furoate + Metronidazole", "Tablet", "250mg/200mg", "tablet", 11.0, 16.5),
        ("Entamizole DS", "Diloxanide Furoate + Metronidazole", "Tablet", "500mg/400mg", "tablet", 18.0, 27.0),
        ("Duphalac", "Lactulose", "Syrup", "120ml", "bottle_120ml", 180.0, 260.0),
        ("Smecta", "Diosmectite", "Sachet", "3g", "sachet", 28.0, 42.0),
        ("ORS Lemon", "Oral Rehydration Salts", "Sachet", "20.5g", "sachet", 15.0, 22.0),
        ("Spasrid", "Drotaverine", "Tablet", "40mg", "tablet", 14.0, 21.0),
    ],
    "CAT005": [
        ("Glucophage 500", "Metformin", "Tablet", "500mg", "tablet", 5.5, 8.5),
        ("Glucophage 850", "Metformin", "Tablet", "850mg", "tablet", 8.0, 12.0),
        ("Glucophage 1000 XR", "Metformin Extended Release", "Tablet", "1000mg", "tablet", 14.0, 21.0),
        ("Getryl 1mg", "Glimepiride", "Tablet", "1mg", "tablet", 9.0, 14.0),
        ("Getryl 2mg", "Glimepiride", "Tablet", "2mg", "tablet", 15.0, 22.0),
        ("Getryl 3mg", "Glimepiride", "Tablet", "3mg", "tablet", 20.0, 30.0),
        ("Galvus 50", "Vildagliptin", "Tablet", "50mg", "tablet", 42.0, 62.0),
        ("Galvus Met 50/500", "Vildagliptin + Metformin", "Tablet", "50mg/500mg", "tablet", 48.0, 70.0),
        ("Januvia 100", "Sitagliptin", "Tablet", "100mg", "tablet", 75.0, 108.0),
        ("Janumet 50/850", "Sitagliptin + Metformin", "Tablet", "50mg/850mg", "tablet", 78.0, 112.0),
        ("Jardiance 10", "Empagliflozin", "Tablet", "10mg", "tablet", 85.0, 122.0),
        ("Jardiance 25", "Empagliflozin", "Tablet", "25mg", "tablet", 115.0, 165.0),
        ("Lopressor 100", "Metoprolol Tartrate", "Tablet", "100mg", "tablet", 12.0, 18.0),
        ("Concor 2.5", "Bisoprolol", "Tablet", "2.5mg", "tablet", 15.0, 22.0),
        ("Concor 5", "Bisoprolol", "Tablet", "5mg", "tablet", 24.0, 35.0),
        ("Norvasc 5", "Amlodipine", "Tablet", "5mg", "tablet", 14.0, 21.0),
        ("Norvasc 10", "Amlodipine", "Tablet", "10mg", "tablet", 24.0, 35.0),
        ("Softvasc 5", "Amlodipine", "Capsule", "5mg", "capsule", 11.0, 16.5),
        ("Cozaar 50", "Losartan Potassium", "Tablet", "50mg", "tablet", 28.0, 42.0),
        ("Hyzaar 50/12.5", "Losartan + Hydrochlorothiazide", "Tablet", "50mg/12.5mg", "tablet", 34.0, 50.0),
        ("Diovan 80", "Valsartan", "Tablet", "80mg", "tablet", 36.0, 54.0),
        ("Exforge 5/80", "Amlodipine + Valsartan", "Tablet", "5mg/80mg", "tablet", 58.0, 85.0),
        ("Lipitor 10", "Atorvastatin", "Tablet", "10mg", "tablet", 30.0, 45.0),
        ("Lipitor 20", "Atorvastatin", "Tablet", "20mg", "tablet", 48.0, 70.0),
        ("Crestor 10", "Rosuvastatin", "Tablet", "10mg", "tablet", 42.0, 62.0),
    ],
    "CAT006": [
        ("Surbex Z", "Zinc + Vitamin B-Complex + C", "Tablet", "High Potency", "tablet", 18.0, 26.0),
        ("Theragran Ultra", "Multivitamins with Minerals", "Tablet", "Multivitamin", "tablet", 22.0, 32.0),
        ("Centrum Silver", "Complete Multivitamin", "Tablet", "Multivitamin", "tablet", 45.0, 65.0),
        ("Centrum Adults", "Complete Multivitamin", "Tablet", "Multivitamin", "tablet", 40.0, 58.0),
        ("CaC-1000 Plus Orange", "Calcium + Vitamin C + D3 + B6", "Effervescent", "1000mg/500mg", "tablet", 25.0, 36.0),
        ("CaC-1000 Plus Lemon", "Calcium + Vitamin C + D3 + B6", "Effervescent", "1000mg/500mg", "tablet", 25.0, 36.0),
        ("Calcee", "Calcium Lactate + Vitamin C", "Effervescent", "500mg/500mg", "tablet", 20.0, 30.0),
        ("Bonex-D", "Calcium Carbonate + Vitamin D3", "Tablet", "600mg/400IU", "tablet", 15.0, 22.0),
        ("Osnate-D", "Ossein Mineral Complex + Vit D", "Tablet", "830mg", "tablet", 19.0, 28.0),
        ("Evion 400", "Vitamin E", "Capsule", "400mg", "capsule", 14.0, 21.0),
        ("Evion 200", "Vitamin E", "Capsule", "200mg", "capsule", 9.0, 14.0),
        ("Neurobion", "Vitamin B1 + B6 + B12", "Tablet", "100/200/200mcg", "tablet", 12.0, 18.0),
        ("Cobalmin 500", "Mecobalamin", "Tablet", "500mcg", "tablet", 16.0, 24.0),
        ("Methycobal 500", "Mecobalamin", "Tablet", "500mcg", "tablet", 22.0, 32.0),
        ("Sunny D 200000", "Cholecalciferol (Vit D3)", "Injection/Oral", "200000IU/1ml", "ampoule", 160.0, 235.0),
        ("Indrop-D 200000", "Cholecalciferol (Vit D3)", "Oral Solution", "200000IU/1ml", "ampoule", 175.0, 255.0),
        ("Sunny D Drops", "Cholecalciferol (Vit D3)", "Drops", "400IU/drop", "bottle_15ml", 280.0, 395.0),
        ("Fefol-Vit", "Ferrous Sulphate + Folic Acid + B", "Capsule", "150mg/0.5mg", "capsule", 15.0, 22.0),
        ("Iberet Folic 500", "Iron + Vitamin C + B-Complex", "Tablet", "525mg", "tablet", 26.0, 38.0),
        ("Maltofer", "Iron Polymaltose Complex", "Tablet", "100mg", "tablet", 28.0, 42.0),
        ("Maltofer Syrup", "Iron Polymaltose Complex", "Syrup", "50mg/5ml", "bottle_120ml", 190.0, 275.0),
        ("C-Cevit", "Vitamin C", "Chewable", "500mg", "tablet", 8.0, 12.0),
        ("Becozinc", "B-Complex + Zinc", "Capsule", "Multivitamin", "capsule", 12.0, 18.0),
        ("Revital", "Ginseng + Multivitamins", "Capsule", "Daily", "capsule", 35.0, 52.0),
        ("Seven Seas Cod Liver", "Cod Liver Oil + Omega 3", "Capsule", "500mg", "capsule", 24.0, 36.0),
    ]
}

def generate_master_data():
    print("Generating Master Data...")
    
    # 1. Categories DataFrame
    df_categories = pd.DataFrame(CATEGORIES_DATA)[["category_id", "category_name", "description"]]
    df_categories.to_csv(os.path.join(RAW_DATA_DIR, "categories.csv"), index=False)
    print(f"Categories created: {len(df_categories)} rows")

    # 2. Suppliers DataFrame
    df_suppliers = pd.DataFrame(SUPPLIERS_DATA)
    df_suppliers.to_csv(os.path.join(RAW_DATA_DIR, "suppliers.csv"), index=False)
    print(f"Suppliers created: {len(df_suppliers)} rows")

    # 3. Demand pattern assignment
    # Exact required counts:
    # Consistent: 40, Seasonal: 35, Growing: 25, Declining: 20, Irregular: 30
    demand_patterns = (
        ["Consistent"] * 40 +
        ["Seasonal"] * 35 +
        ["Growing"] * 25 +
        ["Declining"] * 20 +
        ["Irregular"] * 30
    )
    # Shuffle deterministically
    random.shuffle(demand_patterns)

    # 4. Medicines DataFrame
    medicines = []
    med_idx = 1
    supplier_ids = [s["supplier_id"] for s in SUPPLIERS_DATA]
    sup_lead_times = {s["supplier_id"]: s["lead_time_days"] for s in SUPPLIERS_DATA}

    for cat_id, templates in MEDICINE_TEMPLATES.items():
        for item in templates:
            brand, generic, form, strength, uom, base_cost, base_price = item
            med_id = f"MED{med_idx:03d}"
            
            pattern = demand_patterns[med_idx - 1]
            pref_sup = supplier_ids[(med_idx - 1) % len(supplier_ids)]
            lead_time = sup_lead_times[pref_sup]
            
            # Calibrate base demand by category
            if cat_id in ["CAT001", "CAT005"]:
                est_base_demand = 22.0
            elif cat_id in ["CAT002", "CAT003", "CAT004"]:
                est_base_demand = 16.0
            else:
                est_base_demand = 18.0
                
            # Reorder levels: lead_time demand + safety stock (4 to 8 days demand)
            safety_days = random.randint(4, 7)
            reorder_level = int((lead_time + safety_days) * est_base_demand)
            target_stock_days = int(random.choice([25, 30, 35]))
            
            # Synthetic shelf life per Section 4.E (approximately 60 to 180 days)
            shelf_life_days = int(random.choice([90, 120, 150, 180]))
            
            # Add small realistic variation to unit cost and price
            cost_var = round(float(base_cost * random.uniform(0.95, 1.05)), 2)
            price_var = round(float(base_price * random.uniform(0.96, 1.04)), 2)
            if price_var <= cost_var:
                price_var = round(cost_var * 1.3, 2)
            
            medicines.append({
                "medicine_id": med_id,
                "medicine_name": f"{brand} ({generic}) {strength}",
                "generic_name": generic,
                "brand_name": brand,
                "category_id": cat_id,
                "dosage_form": form,
                "strength": strength,
                "unit_of_measure": uom,
                "unit_cost": cost_var,
                "selling_price": price_var,
                "reorder_level": reorder_level,
                "target_stock_days": target_stock_days,
                "shelf_life_days": shelf_life_days,
                "demand_pattern": pattern,
                "preferred_supplier_id": pref_sup,
                "active_status": "Active"
            })
            med_idx += 1

    df_medicines = pd.DataFrame(medicines)
    df_medicines.to_csv(os.path.join(RAW_DATA_DIR, "medicines.csv"), index=False)
    print(f"Medicines created: {len(df_medicines)} rows")

    # 5. Medicine Suppliers DataFrame
    med_suppliers = []
    ms_idx = 1

    for _, med in df_medicines.iterrows():
        m_id = med["medicine_id"]
        pref_sup_id = med["preferred_supplier_id"]
        base_cost = med["unit_cost"]
        
        pref_sup_info = next(s for s in SUPPLIERS_DATA if s["supplier_id"] == pref_sup_id)
        med_suppliers.append({
            "medicine_supplier_id": f"MS{ms_idx:05d}",
            "medicine_id": m_id,
            "supplier_id": pref_sup_id,
            "is_preferred_supplier": 1,
            "negotiated_unit_cost": round(base_cost, 2),
            "supplier_lead_time_days": pref_sup_info["lead_time_days"]
        })
        ms_idx += 1
        
        available_backups = [s for s in SUPPLIERS_DATA if s["supplier_id"] != pref_sup_id]
        num_backups = random.choice([1, 2])
        chosen_backups = random.sample(available_backups, num_backups)
        
        for b_sup in chosen_backups:
            backup_cost = round(base_cost * random.uniform(1.04, 1.15), 2)
            med_suppliers.append({
                "medicine_supplier_id": f"MS{ms_idx:05d}",
                "medicine_id": m_id,
                "supplier_id": b_sup["supplier_id"],
                "is_preferred_supplier": 0,
                "negotiated_unit_cost": backup_cost,
                "supplier_lead_time_days": b_sup["lead_time_days"]
            })
            ms_idx += 1

    df_med_suppliers = pd.DataFrame(med_suppliers)
    df_med_suppliers.to_csv(os.path.join(RAW_DATA_DIR, "medicine_suppliers.csv"), index=False)
    print(f"Medicine Suppliers created: {len(df_med_suppliers)} rows")

if __name__ == "__main__":
    generate_master_data()
