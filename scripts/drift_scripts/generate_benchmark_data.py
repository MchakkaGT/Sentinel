import csv
import random
from pathlib import Path

# Category templates
TEMPLATES = {
    "business": [
        "Stocks {action} as markets react to {event}.",
        "Quarterly profits {action} analyst expectations.",
        "Merger between {company1} and {company2} announced.",
        "New trade regulations affect {industry} exports.",
        "Central bank {action} interest rates by {percent}."
    ],
    "sports": [
        "Local team {result} championship after {duration}.",
        "Tennis star {result} match in straight sets.",
        "New signing for the regional football club: {player}.",
        "Olympic committee discusses host city for {year}.",
        "Athlete breaks world record in {sport} event."
    ],
    "politics": [
        "New policy proposal sparks debate in {location}.",
        "Election results are final: {party} gains majority.",
        "Diplomatic talks resume to resolve {conflict}.",
        "Prime minister announces new cabinet members.",
        "Senate votes on controversial {bill} legislation."
    ],
    "science": [
        "Breakthrough in {field} research shows major promise.",
        "Discovery of new {object} in deep space announced.",
        "Study explores the impact of {natural_force} on {ecosystem}.",
        "Genetic engineering milestone reached in {lab}.",
        "Scientists warn about the acceleration of {phenomenon}."
    ],
    "tech": [
        "Tech giant launches new {product} model.",
        "Security vulnerability discovered in {software}.",
        "Startup raises ${amount} million for {tech_field} platform.",
        "New {hardware} architecture doubles processing speed.",
        "Cloud provider expands infrastructure in {region}."
    ],
    "entertainment": [  # New category for OOD
        "New blockbuster movie breaks box office records.",
        "Famous singer announces world tour starting in {city}.",
        "Streaming platform releases highly anticipated {genre} series.",
        "Award ceremony honors best achievements in {media}.",
        "Celebrity couple announces their engagement."
    ]
}

VALUES = {
    "action": ["rise", "fall", "steady", "soar", "plummet", "exceed", "miss"],
    "event": ["earnings", "inflation", "jobs report", "fed meeting", "global crisis"],
    "company1": ["TechCorp", "GlobalBank", "AutoMotive", "EnergyFlow"],
    "company2": ["SoftSystems", "FinLeap", "ElectricDrive", "PowerGrid"],
    "industry": ["manufacturing", "technology", "agriculture", "service"],
    "percent": ["0.25%", "0.5%", "1.0%", "0.75%"],
    "result": ["wins", "loses", "secures", "defends", "claims"],
    "duration": ["overtime", "penalty shootout", "hard-fought game", "record time"],
    "player": ["Alex Smith", "Jamie Johnson", "Sam Rivers", "Chris Taylor"],
    "year": ["2028", "2032", "2036"],
    "sport": ["100m sprint", "marathon", "swimming", "cycling"],
    "location": ["the senate", "the capital", "parliament", "city hall"],
    "party": ["The Liberal Party", "The Conservative Party", "Citizens First", "Green Alliance"],
    "conflict": ["trade war", "border dispute", "maritime issue"],
    "bill": ["infrastructure", "tax reform", "climate", "healthcare"],
    "field": ["cancer", "climate change", "quantum", "neurology"],
    "object": ["exoplanet", "black hole", "galaxy", "asteroid"],
    "natural_force": ["temperature rise", "ocean currents", "solar flares"],
    "ecosystem": ["coral reefs", "amazon rainforest", "arctic ice"],
    "lab": ["MIT", "Stanford", "CERN", "Max Planck"],
    "phenomenon": ["glacier melting", "species extinction", "desertification"],
    "product": ["smartphone", "tablet", "laptop", "smartwatch"],
    "software": ["encryption protocol", "operating system", "web browser"],
    "amount": ["50", "120", "200", "450"],
    "tech_field": ["AI", "Blockchain", "Cybersecurity", "Fintech"],
    "hardware": ["GPU", "Processor", "Neural Engine"],
    "region": ["Europe", "Asia", "South America"],
    "city": ["London", "Paris", "New York", "Tokyo"],
    "genre": ["Sci-Fi", "Drama", "Docuseries", "Comedy"],
    "media": ["Film", "Music", "Digital Art"]
}

def generate_sentence(category):
    template = random.choice(TEMPLATES[category])
    return template.format(**{k: random.choice(v) for k, v in VALUES.items() if f"{{{k}}}" in template})

def save_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "text"])
        writer.writeheader()
        writer.writerows(rows)

def main():
    # Increase samples for better accuracy
    SAMPLE_SIZE = 50
    
    # 1. Reference Data (Balanced)
    ref_rows = []
    for cat in ["business", "sports", "politics", "science", "tech"]:
        for _ in range(SAMPLE_SIZE):
            ref_rows.append({"label": cat, "text": generate_sentence(cat)})
    
    save_csv(Path("data/benchmarks/ref_balanced.csv"), ref_rows)

    # 2. No Drift (Same distribution)
    no_drift_rows = []
    for cat in ["business", "sports", "politics", "science", "tech"]:
        for _ in range(SAMPLE_SIZE):
            no_drift_rows.append({"label": cat, "text": generate_sentence(cat)})
    save_csv(Path("data/benchmarks/cur_no_drift.csv"), no_drift_rows)

    # 3. Label Shift (Prior Probability Shift) - Heavy on Sports and Tech
    label_shift_rows = []
    distribution = {"business": 5, "sports": 80, "politics": 5, "science": 5, "tech": 155}
    for cat, count in distribution.items():
        for _ in range(count):
            label_shift_rows.append({"label": cat, "text": generate_sentence(cat)})
    save_csv(Path("data/benchmarks/cur_label_shift.csv"), label_shift_rows)

    # 4. Extreme Feature Shift (Noise) - Nonsense/Gibberish
    noise_rows = []
    words = ["lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "adipiscing", "elit"]
    for _ in range(250):
        text = " ".join(random.choices(words, k=10))
        noise_rows.append({"label": "unknown", "text": text})
    save_csv(Path("data/benchmarks/cur_noise_shift.csv"), noise_rows)

    # 5. OOD Shift (Out-of-Distribution) - New categories
    ood_shift_rows = []
    for cat in ["entertainment"]:
        for _ in range(250):
            ood_shift_rows.append({"label": cat, "text": generate_sentence(cat)})
    save_csv(Path("data/benchmarks/cur_ood_shift.csv"), ood_shift_rows)

    # 6. Mixed Drift (Label Shift + Mild Semantic Shift)
    mixed_rows = []
    for cat in ["business", "sports"]:
        for _ in range(125):
            # Mix in some politics/tech keywords into business/sports to confuse semantics
            text = generate_sentence(cat) + " " + " ".join(random.choices(VALUES["bill"] + VALUES["product"], k=3))
            mixed_rows.append({"label": cat, "text": text})
    save_csv(Path("data/benchmarks/cur_mixed_drift.csv"), mixed_rows)

    print("✅ Enhanced benchmark datasets generated in data/benchmarks/")

if __name__ == "__main__":
    main()
